"""Config flow for dominionsc integration."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from datetime import date
from typing import Any

import voluptuous as vol
from homeassistant.config_entries import (
    SOURCE_REAUTH,
    ConfigEntry,
    ConfigFlow,
    ConfigFlowResult,
    OptionsFlow,
)
from homeassistant.const import CONF_NAME, CONF_PASSWORD, CONF_USERNAME
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.aiohttp_client import async_create_clientsession
from homeassistant.helpers.typing import VolDictType

from dominionsc import (
    ApiException,
    CannotConnect,
    DominionSC,
    DominionSCTFAHandler,
    InvalidAuth,
    MfaChallenge,
    create_cookie_jar,
)

from .const import (
    COMMON_NAME,
    CONF_COST_MODE,
    CONF_FIXED_RATE,
    CONF_LOGIN_DATA,
    COST_MODE_FIXED,
    COST_MODE_NONE,
    COST_MODE_RATE_8,
    DEFAULT_FIXED_RATE,
    DOMAIN,
)
from .rates import TIERED_RATE_REGISTRY, build_cost_mode_choices

CONF_RECALCULATE_HISTORY = "recalculate_history"
CONF_RECALC_START_DATE = "recalc_start_date"
CONF_RECALC_END_DATE = "recalc_end_date"

_LOGGER = logging.getLogger(__name__)

CONF_TFA_CODE = "tfa_code"
CONF_TFA_METHOD = "tfa_method"


async def _validate_login(
    hass: HomeAssistant,
    data: Mapping[str, Any],
) -> None:
    """Validate login data and raise exceptions on failure."""
    api = DominionSC(
        async_create_clientsession(hass, cookie_jar=create_cookie_jar()),
        data[CONF_USERNAME],
        data[CONF_PASSWORD],
        data.get(CONF_LOGIN_DATA),
    )
    _LOGGER.debug("API: async_login")
    await api.async_login()


class DominionSCConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle a config flow for dominionsc."""

    VERSION = 1

    def __init__(self) -> None:
        """Initialize a new DominionSCConfigFlow."""
        self._data: dict[str, Any] = {}
        self._options: dict[str, Any] = {}
        self.tfa_handler: DominionSCTFAHandler | None = None

    @staticmethod
    @callback
    def async_get_options_flow(config_entry: ConfigEntry) -> OptionsFlow:
        """Get the options flow for this handler."""
        return DominionSCOptionsFlow(config_entry)

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle the initial step (credentials)."""
        errors: dict[str, str] = {}

        if user_input is not None:
            self._data.update(user_input)

            self._async_abort_entries_match(
                {
                    CONF_USERNAME: self._data[CONF_USERNAME],
                }
            )

            try:
                await _validate_login(self.hass, self._data)
            except MfaChallenge as exc:
                self.tfa_handler = exc.handler
                _LOGGER.debug("API: async_step_tfa_options")
                return await self.async_step_tfa_options()
            except InvalidAuth:
                errors["base"] = "invalid_auth"
            except CannotConnect:
                errors["base"] = "cannot_connect"
            except ApiException as err:
                _LOGGER.error("API structure error during login: %s", err)
                errors["base"] = "unknown"
            else:
                return await self.async_step_cost_mode()

        schema_dict: VolDictType = {
            vol.Required(CONF_USERNAME): str,
            vol.Required(CONF_PASSWORD): str,
        }

        return self.async_show_form(
            step_id="user",
            data_schema=self.add_suggested_values_to_schema(
                vol.Schema(schema_dict), user_input
            ),
            errors=errors,
        )

    async def async_step_tfa_options(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle TFA options step."""
        errors: dict[str, str] = {}
        assert self.tfa_handler is not None

        if user_input is not None:
            method = user_input[CONF_TFA_METHOD]
            try:
                _LOGGER.debug("API: async_select_tfa_option")
                await self.tfa_handler.async_select_tfa_option(method)
            except CannotConnect:
                errors["base"] = "cannot_connect"
            except ApiException as err:
                _LOGGER.error(
                    "API structure error during TFA option selection: %s", err
                )
                errors["base"] = "unknown"
            else:
                return await self.async_step_tfa_code()

        _LOGGER.debug("API: async_get_tfa_options")
        try:
            tfa_options = await self.tfa_handler.async_get_tfa_options()
        except ApiException as err:
            _LOGGER.error("API structure error getting TFA options: %s", err)
            errors["base"] = "unknown"
            # Show error to user instead of proceeding
            return self.async_show_form(
                step_id="tfa_options",
                data_schema=vol.Schema({}),
                errors=errors,
            )

        if not tfa_options:
            return await self.async_step_tfa_code()
        return self.async_show_form(
            step_id="tfa_options",
            data_schema=self.add_suggested_values_to_schema(
                vol.Schema({vol.Required(CONF_TFA_METHOD): vol.In(tfa_options)}),
                user_input,
            ),
            errors=errors,
        )

    async def async_step_tfa_code(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle TFA code submission step."""
        assert self.tfa_handler is not None
        errors: dict[str, str] = {}
        if user_input is not None:
            code = user_input[CONF_TFA_CODE]
            try:
                _LOGGER.debug("API: async_submit_tfa_code")
                login_data = await self.tfa_handler.async_submit_tfa_code(code)
            except InvalidAuth:
                errors["base"] = "invalid_tfa_code"
            except CannotConnect:
                errors["base"] = "cannot_connect"
            except ApiException as err:
                _LOGGER.error("API structure error during TFA code submission: %s", err)
                errors["base"] = "unknown"
            else:
                self._data[CONF_LOGIN_DATA] = login_data
                if self.source == SOURCE_REAUTH:
                    return self.async_update_reload_and_abort(
                        self._get_reauth_entry(), data=self._data
                    )
                return await self.async_step_cost_mode()

        return self.async_show_form(
            step_id="tfa_code",
            data_schema=self.add_suggested_values_to_schema(
                vol.Schema({vol.Required(CONF_TFA_CODE): str}), user_input
            ),
            errors=errors,
        )

    async def async_step_cost_mode(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Select cost calculation mode during initial setup."""
        if user_input is not None:
            mode = user_input[CONF_COST_MODE]
            if mode == COST_MODE_FIXED:
                return await self.async_step_cost_mode_fixed_rate()
            self._options = {CONF_COST_MODE: mode}
            return self._async_create_dominionsc_entry(self._data)

        mode_choices = build_cost_mode_choices()

        return self.async_show_form(
            step_id="cost_mode",
            data_schema=vol.Schema(
                {
                    vol.Required(CONF_COST_MODE, default=COST_MODE_RATE_8): vol.In(
                        mode_choices
                    ),
                }
            ),
        )

    async def async_step_cost_mode_fixed_rate(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Configure a custom fixed rate during initial setup."""
        if user_input is not None:
            self._options = {
                CONF_COST_MODE: COST_MODE_FIXED,
                CONF_FIXED_RATE: user_input[CONF_FIXED_RATE],
            }
            return self._async_create_dominionsc_entry(self._data)

        return self.async_show_form(
            step_id="cost_mode_fixed_rate",
            data_schema=vol.Schema(
                {
                    vol.Required(
                        CONF_FIXED_RATE, default=DEFAULT_FIXED_RATE
                    ): vol.Coerce(float),
                }
            ),
        )

    @callback
    def _async_create_dominionsc_entry(
        self, data: dict[str, Any], **kwargs: Any
    ) -> ConfigFlowResult:
        """Create the config entry."""
        return self.async_create_entry(
            title=f"{COMMON_NAME} ({data[CONF_USERNAME]})",
            data=data,
            options=self._options,
            **kwargs,
        )

    async def async_step_reauth(
        self, entry_data: Mapping[str, Any]
    ) -> ConfigFlowResult:
        """Handle configuration by re-auth."""
        reauth_entry = self._get_reauth_entry()
        self._data = dict(reauth_entry.data)
        return self.async_show_form(
            step_id="reauth_confirm",
            description_placeholders={CONF_NAME: reauth_entry.title},
        )

    async def async_step_reauth_confirm(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Dialog that informs the user that reauth is required."""
        errors: dict[str, str] = {}
        reauth_entry = self._get_reauth_entry()

        if user_input is not None:
            self._data.update(user_input)

            try:
                await _validate_login(self.hass, self._data)
            except MfaChallenge as exc:
                self.tfa_handler = exc.handler
                return await self.async_step_tfa_options()
            except InvalidAuth:
                errors["base"] = "invalid_auth"
            except CannotConnect:
                errors["base"] = "cannot_connect"
            except ApiException as err:
                _LOGGER.error("API structure error during reauth: %s", err)
                errors["base"] = "unknown"
            else:
                return self.async_update_reload_and_abort(reauth_entry, data=self._data)

        schema_dict: VolDictType = {
            vol.Required(CONF_USERNAME): str,
            vol.Required(CONF_PASSWORD): str,
        }

        return self.async_show_form(
            step_id="reauth_confirm",
            data_schema=self.add_suggested_values_to_schema(
                vol.Schema(schema_dict), self._data
            ),
            errors=errors,
            description_placeholders={CONF_NAME: reauth_entry.title},
        )


class DominionSCOptionsFlow(OptionsFlow):
    """Handle options flow for Dominion Energy SC."""

    def __init__(self, config_entry: ConfigEntry) -> None:
        """Initialize options flow."""
        self._config_entry = config_entry
        self._selected_mode: str | None = None
        self._new_options: dict[str, Any] = {}

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Step 1: Select cost calculation mode."""
        # Guard: block changes while a background recalculation is running.
        coordinator = self.hass.data.get(DOMAIN, {}).get(self._config_entry.entry_id)
        if coordinator is not None and coordinator.recalculation_lock.locked():
            return self.async_show_form(
                step_id="init",
                data_schema=vol.Schema({}),
                errors={"base": "recalculation_in_progress"},
            )

        if user_input is not None:
            self._selected_mode = user_input[CONF_COST_MODE]

            if self._selected_mode == COST_MODE_FIXED:
                return await self.async_step_fixed_rate()
            if self._selected_mode in TIERED_RATE_REGISTRY:
                self._new_options = {CONF_COST_MODE: self._selected_mode}
                return await self.async_step_recalculate_history()
            # No cost calculation - skip history recalculation (nothing to calculate)
            return self.async_create_entry(
                title="", data={CONF_COST_MODE: COST_MODE_NONE}
            )

        current_options = self._config_entry.options

        mode_choices = build_cost_mode_choices()

        return self.async_show_form(
            step_id="init",
            data_schema=vol.Schema(
                {
                    vol.Required(
                        CONF_COST_MODE,
                        default=current_options.get(CONF_COST_MODE, COST_MODE_RATE_8),
                    ): vol.In(mode_choices),
                }
            ),
        )

    async def async_step_fixed_rate(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Step 2a: Configure fixed rate."""
        if user_input is not None:
            self._new_options = {
                CONF_COST_MODE: COST_MODE_FIXED,
                CONF_FIXED_RATE: user_input[CONF_FIXED_RATE],
            }
            return await self.async_step_recalculate_history()

        current_options = self._config_entry.options

        return self.async_show_form(
            step_id="fixed_rate",
            data_schema=vol.Schema(
                {
                    vol.Required(
                        CONF_FIXED_RATE,
                        default=current_options.get(
                            CONF_FIXED_RATE, DEFAULT_FIXED_RATE
                        ),
                    ): vol.Coerce(float),
                }
            ),
        )

    async def async_step_recalculate_history(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Step 3: Ask if the user wants to recalculate historic cost records."""
        if user_input is not None:
            if user_input.get(CONF_RECALCULATE_HISTORY, False):
                return await self.async_step_recalculate_date_range()
            # No recalculation requested — save options and finish
            return self.async_create_entry(title="", data=self._new_options)

        old_mode = self._config_entry.options.get(CONF_COST_MODE, COST_MODE_RATE_8)
        new_mode = self._new_options.get(CONF_COST_MODE, COST_MODE_RATE_8)
        mode_changed = old_mode != new_mode

        return self.async_show_form(
            step_id="recalculate_history",
            data_schema=vol.Schema(
                {
                    vol.Required(
                        CONF_RECALCULATE_HISTORY,
                        default=mode_changed,
                    ): bool,
                }
            ),
            description_placeholders={
                "old_mode": _cost_mode_label(old_mode),
                "new_mode": _cost_mode_label(new_mode),
            },
        )

    async def async_step_recalculate_date_range(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Step 4: Select the date range for historic recalculation."""
        errors: dict[str, str] = {}

        if user_input is not None:
            start_str = user_input.get(CONF_RECALC_START_DATE, "")
            end_str = user_input.get(CONF_RECALC_END_DATE, "")
            try:
                start_date = date.fromisoformat(str(start_str))
                end_date = date.fromisoformat(str(end_str))
            except (ValueError, TypeError):
                errors["base"] = "invalid_date_format"
            else:
                if end_date < start_date:
                    errors["base"] = "end_before_start"
                elif end_date > date.today():
                    errors["base"] = "end_in_future"
                else:
                    # Trigger async recalculation via coordinator
                    coordinator = self.hass.data[DOMAIN][self._config_entry.entry_id]
                    self.hass.async_create_task(
                        coordinator.async_recalculate_historic_costs(
                            start_date=start_date,
                            end_date=end_date,
                            new_options=self._new_options,
                        )
                    )
                    return self.async_create_entry(title="", data=self._new_options)

        today = date.today()
        default_start = date(today.year, today.month, 1)

        return self.async_show_form(
            step_id="recalculate_date_range",
            data_schema=vol.Schema(
                {
                    vol.Required(
                        CONF_RECALC_START_DATE,
                        default=str(default_start),
                    ): str,
                    vol.Required(
                        CONF_RECALC_END_DATE,
                        default=str(today),
                    ): str,
                }
            ),
            errors=errors,
        )


def _cost_mode_label(mode: str) -> str:
    """Return a human-readable label for a cost mode constant."""
    if mode in TIERED_RATE_REGISTRY:
        return TIERED_RATE_REGISTRY[mode].name
    return {
        COST_MODE_NONE: "None",
        COST_MODE_FIXED: "Fixed Rate",
    }.get(mode, mode)
