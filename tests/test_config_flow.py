"""Tests for Dominion Energy SC config flow."""

from unittest.mock import AsyncMock, MagicMock, patch

from dominionsc.exceptions import ApiException, CannotConnect, InvalidAuth, MfaChallenge
from homeassistant import config_entries
from homeassistant.const import CONF_PASSWORD, CONF_USERNAME
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.dominionsc.config_flow import (
    CONF_RECALC_END_DATE,
    CONF_RECALC_START_DATE,
    CONF_RECALCULATE_HISTORY,
    CONF_TFA_CODE,
    CONF_TFA_METHOD,
    DominionSCConfigFlow,
    DominionSCOptionsFlow,
    _validate_login,
)
from custom_components.dominionsc.const import (
    CONF_COST_MODE,
    CONF_EXTENDED_BACKFILL,
    CONF_EXTENDED_COST_BACKFILL,
    CONF_FIXED_RATE,
    CONF_LOGIN_DATA,
    COST_MODE_FIXED,
    COST_MODE_NONE,
    COST_MODE_RATE_6,
    COST_MODE_RATE_8,
    DOMAIN,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_entry(hass: HomeAssistant, user_input: dict) -> MockConfigEntry:
    """Register a config entry directly, bypassing the flow."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        data=user_input,
        options={CONF_COST_MODE: COST_MODE_RATE_8},
        title=f"Dominion Energy SC ({user_input[CONF_USERNAME]})",
    )
    entry.add_to_hass(hass)
    return entry


async def _login_to_backfill(hass: HomeAssistant, user_input: dict) -> dict:
    """Run the user step with a patched login and return the backfill_options form."""
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    with patch("custom_components.dominionsc.config_flow._validate_login"):
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"], user_input
        )
    assert result["step_id"] == "backfill_options"
    return result


async def _login_to_cost_mode(hass: HomeAssistant, user_input: dict) -> dict:
    """Run through login and backfill_options, return the cost_mode form result."""
    result = await _login_to_backfill(hass, user_input)
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"],
        {CONF_EXTENDED_BACKFILL: False, CONF_EXTENDED_COST_BACKFILL: False},
    )
    assert result["step_id"] == "cost_mode"
    return result


# ---------------------------------------------------------------------------
# Config flow — credentials step
# ---------------------------------------------------------------------------


async def test_user_step_shows_form(hass: HomeAssistant) -> None:
    """Initial step renders the credentials form."""
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "user"
    assert result["errors"] == {}


async def test_user_flow_invalid_auth(
    hass: HomeAssistant,
    mock_dominionsc_api: AsyncMock,
    user_input: dict,
) -> None:
    """Invalid credentials show an error and stay on the user step."""
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    with patch(
        "custom_components.dominionsc.config_flow._validate_login",
        side_effect=InvalidAuth("Invalid credentials"),
    ):
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"], user_input
        )

    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "user"
    assert result["errors"] == {"base": "invalid_auth"}


async def test_user_flow_cannot_connect(
    hass: HomeAssistant,
    mock_dominionsc_api: AsyncMock,
    user_input: dict,
) -> None:
    """Connection failure shows an error and stays on the user step."""
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    with patch(
        "custom_components.dominionsc.config_flow._validate_login",
        side_effect=CannotConnect("Connection error"),
    ):
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"], user_input
        )

    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "user"
    assert result["errors"] == {"base": "cannot_connect"}


async def test_user_flow_api_exception(
    hass: HomeAssistant,
    mock_dominionsc_api: AsyncMock,
    user_input: dict,
) -> None:
    """API exception during login shows an unknown error."""
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    with patch(
        "custom_components.dominionsc.config_flow._validate_login",
        side_effect=ApiException("API error", "https://test.com"),
    ):
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"], user_input
        )

    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "user"
    assert result["errors"] == {"base": "unknown"}


# ---------------------------------------------------------------------------
# Config flow — cost mode step (shown after successful login)
# ---------------------------------------------------------------------------


async def test_user_flow_success_rate_8(
    hass: HomeAssistant,
    mock_dominionsc_api: AsyncMock,
    mock_setup_entry: AsyncMock,
    user_input: dict,
) -> None:
    """Successful login → Rate 8 selection creates the entry with correct options."""
    result = await _login_to_cost_mode(hass, user_input)

    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {CONF_COST_MODE: COST_MODE_RATE_8}
    )

    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["title"] == f"Dominion Energy SC ({user_input[CONF_USERNAME]})"
    assert result["data"] == user_input
    assert result["options"] == {CONF_COST_MODE: COST_MODE_RATE_8}


async def test_user_flow_success_rate_6(
    hass: HomeAssistant,
    mock_dominionsc_api: AsyncMock,
    mock_setup_entry: AsyncMock,
    user_input: dict,
) -> None:
    """Successful login → Rate 6 selection creates the entry with correct options."""
    result = await _login_to_cost_mode(hass, user_input)

    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {CONF_COST_MODE: COST_MODE_RATE_6}
    )

    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["options"] == {CONF_COST_MODE: COST_MODE_RATE_6}


async def test_user_flow_success_none(
    hass: HomeAssistant,
    mock_dominionsc_api: AsyncMock,
    mock_setup_entry: AsyncMock,
    user_input: dict,
) -> None:
    """Selecting no cost calculation creates the entry immediately."""
    result = await _login_to_cost_mode(hass, user_input)

    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {CONF_COST_MODE: COST_MODE_NONE}
    )

    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["options"] == {CONF_COST_MODE: COST_MODE_NONE}


async def test_user_flow_success_fixed_rate(
    hass: HomeAssistant,
    mock_dominionsc_api: AsyncMock,
    mock_setup_entry: AsyncMock,
    user_input: dict,
) -> None:
    """Selecting fixed rate prompts for the value, then creates the entry."""
    result = await _login_to_cost_mode(hass, user_input)

    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {CONF_COST_MODE: COST_MODE_FIXED}
    )
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "cost_mode_fixed_rate"

    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {CONF_FIXED_RATE: 0.12}
    )

    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["options"] == {CONF_COST_MODE: COST_MODE_FIXED, CONF_FIXED_RATE: 0.12}


# ---------------------------------------------------------------------------
# Config flow — backfill options step
# ---------------------------------------------------------------------------


async def test_backfill_options_shows_form(
    hass: HomeAssistant,
    mock_dominionsc_api: AsyncMock,
    user_input: dict,
) -> None:
    """After login, the backfill options form is shown."""
    result = await _login_to_backfill(hass, user_input)
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "backfill_options"


async def test_backfill_both_off(
    hass: HomeAssistant,
    mock_dominionsc_api: AsyncMock,
    mock_setup_entry: AsyncMock,
    user_input: dict,
) -> None:
    """
    Declining both backfill options proceeds to
    cost_mode with no backfill options set.
    """
    result = await _login_to_backfill(hass, user_input)
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"],
        {CONF_EXTENDED_BACKFILL: False, CONF_EXTENDED_COST_BACKFILL: False},
    )
    assert result["step_id"] == "cost_mode"

    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {CONF_COST_MODE: COST_MODE_RATE_8}
    )
    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert CONF_EXTENDED_BACKFILL not in result["options"]
    assert CONF_EXTENDED_COST_BACKFILL not in result["options"]


async def test_backfill_consumption_only(
    hass: HomeAssistant,
    mock_dominionsc_api: AsyncMock,
    mock_setup_entry: AsyncMock,
    user_input: dict,
) -> None:
    """Enabling only consumption backfill sets the flag and proceeds."""
    result = await _login_to_backfill(hass, user_input)
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"],
        {CONF_EXTENDED_BACKFILL: True, CONF_EXTENDED_COST_BACKFILL: False},
    )
    assert result["step_id"] == "cost_mode"

    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {CONF_COST_MODE: COST_MODE_RATE_8}
    )
    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["options"][CONF_EXTENDED_BACKFILL] is True
    assert CONF_EXTENDED_COST_BACKFILL not in result["options"]


async def test_backfill_both_on(
    hass: HomeAssistant,
    mock_dominionsc_api: AsyncMock,
    mock_setup_entry: AsyncMock,
    user_input: dict,
) -> None:
    """Enabling both backfill options sets both flags."""
    result = await _login_to_backfill(hass, user_input)
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"],
        {CONF_EXTENDED_BACKFILL: True, CONF_EXTENDED_COST_BACKFILL: True},
    )
    assert result["step_id"] == "cost_mode"

    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {CONF_COST_MODE: COST_MODE_RATE_8}
    )
    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["options"][CONF_EXTENDED_BACKFILL] is True
    assert result["options"][CONF_EXTENDED_COST_BACKFILL] is True


async def test_backfill_cost_without_consumption_rejected(
    hass: HomeAssistant,
    mock_dominionsc_api: AsyncMock,
    user_input: dict,
) -> None:
    """Requesting cost backfill without consumption backfill shows an error."""
    result = await _login_to_backfill(hass, user_input)
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"],
        {CONF_EXTENDED_BACKFILL: False, CONF_EXTENDED_COST_BACKFILL: True},
    )
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "backfill_options"
    assert result["errors"] == {"base": "invalid_backfill_selection"}


# ---------------------------------------------------------------------------
# Config flow — TFA paths
# ---------------------------------------------------------------------------


async def test_user_flow_with_tfa(
    hass: HomeAssistant,
    mock_dominionsc_api: AsyncMock,
    mock_tfa_handler: AsyncMock,
    mock_setup_entry: AsyncMock,
    user_input: dict,
) -> None:
    """Full TFA flow routes to cost_mode after a successful code submission."""
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )

    with patch(
        "custom_components.dominionsc.config_flow._validate_login",
        side_effect=MfaChallenge("TFA Required", mock_tfa_handler),
    ):
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"], user_input
        )

    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "tfa_options"

    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {CONF_TFA_METHOD: "sms"}
    )
    assert result["step_id"] == "tfa_code"

    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {CONF_TFA_CODE: "123456"}
    )

    # TFA success routes to backfill options first.
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "backfill_options"

    result = await hass.config_entries.flow.async_configure(
        result["flow_id"],
        {CONF_EXTENDED_BACKFILL: False, CONF_EXTENDED_COST_BACKFILL: False},
    )
    assert result["step_id"] == "cost_mode"

    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {CONF_COST_MODE: COST_MODE_RATE_8}
    )

    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert CONF_LOGIN_DATA in result["data"]
    assert result["options"] == {CONF_COST_MODE: COST_MODE_RATE_8}


async def test_tfa_options_cannot_connect(
    hass: HomeAssistant,
    mock_dominionsc_api: AsyncMock,
    mock_tfa_handler: AsyncMock,
    user_input: dict,
) -> None:
    """Connection error during TFA method selection shows an error."""
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    with patch(
        "custom_components.dominionsc.config_flow._validate_login",
        side_effect=MfaChallenge("TFA Required", mock_tfa_handler),
    ):
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"], user_input
        )

    mock_tfa_handler.async_select_tfa_option.side_effect = CannotConnect(
        "Connection error"
    )
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {CONF_TFA_METHOD: "sms"}
    )

    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "tfa_options"
    assert result["errors"] == {"base": "cannot_connect"}


async def test_tfa_options_api_exception(
    hass: HomeAssistant,
    mock_dominionsc_api: AsyncMock,
    mock_tfa_handler: AsyncMock,
    user_input: dict,
) -> None:
    """API exception during TFA method selection shows an error."""
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    with patch(
        "custom_components.dominionsc.config_flow._validate_login",
        side_effect=MfaChallenge("TFA Required", mock_tfa_handler),
    ):
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"], user_input
        )

    mock_tfa_handler.async_select_tfa_option.side_effect = ApiException(
        "API error", "https://test.com"
    )
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {CONF_TFA_METHOD: "sms"}
    )

    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "tfa_options"
    assert result["errors"] == {"base": "unknown"}


async def test_tfa_options_get_options_api_exception(
    hass: HomeAssistant,
    mock_dominionsc_api: AsyncMock,
    mock_tfa_handler: AsyncMock,
    user_input: dict,
) -> None:
    """API exception when fetching TFA options shows an error form."""
    mock_tfa_handler.async_get_tfa_options.side_effect = ApiException(
        "API error", "https://test.com"
    )
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    with patch(
        "custom_components.dominionsc.config_flow._validate_login",
        side_effect=MfaChallenge("TFA Required", mock_tfa_handler),
    ):
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"], user_input
        )

    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "tfa_options"
    assert result["errors"] == {"base": "unknown"}


async def test_tfa_code_invalid(
    hass: HomeAssistant,
    mock_dominionsc_api: AsyncMock,
    mock_tfa_handler: AsyncMock,
    user_input: dict,
) -> None:
    """Invalid TFA code shows an error and stays on the tfa_code step."""
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    mock_tfa_handler.async_get_tfa_options.return_value = []
    with patch(
        "custom_components.dominionsc.config_flow._validate_login",
        side_effect=MfaChallenge("TFA Required", mock_tfa_handler),
    ):
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"], user_input
        )

    assert result["step_id"] == "tfa_code"

    mock_tfa_handler.async_submit_tfa_code.side_effect = InvalidAuth("Invalid TFA code")
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {CONF_TFA_CODE: "wrong"}
    )

    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "tfa_code"
    assert result["errors"] == {"base": "invalid_tfa_code"}


async def test_tfa_code_cannot_connect(
    hass: HomeAssistant,
    mock_dominionsc_api: AsyncMock,
    mock_tfa_handler: AsyncMock,
    user_input: dict,
) -> None:
    """Connection error during TFA code submission shows an error."""
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    mock_tfa_handler.async_get_tfa_options.return_value = []
    with patch(
        "custom_components.dominionsc.config_flow._validate_login",
        side_effect=MfaChallenge("TFA Required", mock_tfa_handler),
    ):
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"], user_input
        )

    mock_tfa_handler.async_submit_tfa_code.side_effect = CannotConnect(
        "Connection error"
    )
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {CONF_TFA_CODE: "123456"}
    )

    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "tfa_code"
    assert result["errors"] == {"base": "cannot_connect"}


async def test_tfa_code_api_exception(
    hass: HomeAssistant,
    mock_dominionsc_api: AsyncMock,
    mock_tfa_handler: AsyncMock,
    user_input: dict,
) -> None:
    """API exception during TFA code submission shows an error."""
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    mock_tfa_handler.async_get_tfa_options.return_value = []
    with patch(
        "custom_components.dominionsc.config_flow._validate_login",
        side_effect=MfaChallenge("TFA Required", mock_tfa_handler),
    ):
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"], user_input
        )

    mock_tfa_handler.async_submit_tfa_code.side_effect = ApiException(
        "API error", "https://test.com"
    )
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {CONF_TFA_CODE: "123456"}
    )

    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "tfa_code"
    assert result["errors"] == {"base": "unknown"}


# ---------------------------------------------------------------------------
# Config flow — duplicate entry
# ---------------------------------------------------------------------------


async def test_duplicate_entry(
    hass: HomeAssistant,
    mock_dominionsc_api: AsyncMock,
    mock_setup_entry: AsyncMock,
    user_input: dict,
) -> None:
    """A second entry for the same username is aborted."""
    # Create the first entry through the full flow.
    result = await _login_to_cost_mode(hass, user_input)
    await hass.config_entries.flow.async_configure(
        result["flow_id"], {CONF_COST_MODE: COST_MODE_RATE_8}
    )

    # Attempt a second entry for the same username.
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    with patch("custom_components.dominionsc.config_flow._validate_login"):
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"], user_input
        )

    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "already_configured"


# ---------------------------------------------------------------------------
# Reauth flow
# ---------------------------------------------------------------------------


async def test_reauth_flow_success(
    hass: HomeAssistant,
    mock_dominionsc_api: AsyncMock,
    mock_setup_entry: AsyncMock,
    user_input: dict,
) -> None:
    """Successful reauth updates the entry without going through cost mode."""
    entry = MockConfigEntry(domain=DOMAIN, data=user_input, title="Test")
    entry.add_to_hass(hass)

    result = await entry.start_reauth_flow(hass)
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "reauth_confirm"

    with patch("custom_components.dominionsc.config_flow._validate_login"):
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"], user_input
        )

    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "reauth_successful"


async def test_reauth_flow_invalid_auth(
    hass: HomeAssistant,
    mock_dominionsc_api: AsyncMock,
    user_input: dict,
) -> None:
    """Invalid credentials during reauth show an error."""
    entry = MockConfigEntry(domain=DOMAIN, data=user_input, title="Test")
    entry.add_to_hass(hass)
    result = await entry.start_reauth_flow(hass)

    with patch(
        "custom_components.dominionsc.config_flow._validate_login",
        side_effect=InvalidAuth("Invalid credentials"),
    ):
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"], user_input
        )

    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "reauth_confirm"
    assert result["errors"] == {"base": "invalid_auth"}


async def test_reauth_flow_cannot_connect(
    hass: HomeAssistant,
    mock_dominionsc_api: AsyncMock,
    user_input: dict,
) -> None:
    """Connection failure during reauth shows an error."""
    entry = MockConfigEntry(domain=DOMAIN, data=user_input, title="Test")
    entry.add_to_hass(hass)
    result = await entry.start_reauth_flow(hass)

    with patch(
        "custom_components.dominionsc.config_flow._validate_login",
        side_effect=CannotConnect("Connection error"),
    ):
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"], user_input
        )

    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "reauth_confirm"
    assert result["errors"] == {"base": "cannot_connect"}


async def test_reauth_flow_api_exception(
    hass: HomeAssistant,
    mock_dominionsc_api: AsyncMock,
    user_input: dict,
) -> None:
    """API exception during reauth shows an unknown error."""
    entry = MockConfigEntry(domain=DOMAIN, data=user_input, title="Test")
    entry.add_to_hass(hass)
    result = await entry.start_reauth_flow(hass)

    with patch(
        "custom_components.dominionsc.config_flow._validate_login",
        side_effect=ApiException("API error", "https://test.com"),
    ):
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"], user_input
        )

    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "reauth_confirm"
    assert result["errors"] == {"base": "unknown"}


async def test_reauth_confirm_no_input(
    hass: HomeAssistant,
    user_input: dict,
) -> None:
    """Reauth confirm with no input re-renders the form without calling login."""
    entry = MockConfigEntry(domain=DOMAIN, data=user_input)
    entry.add_to_hass(hass)
    result = await entry.start_reauth_flow(hass)

    result = await hass.config_entries.flow.async_configure(result["flow_id"], None)

    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "reauth_confirm"


async def test_reauth_flow_with_tfa(
    hass: HomeAssistant,
    mock_dominionsc_api: AsyncMock,
    mock_tfa_handler: AsyncMock,
    user_input: dict,
) -> None:
    """Reauth via TFA completes without entering cost mode selection."""
    entry = MockConfigEntry(domain=DOMAIN, data=user_input, title="Test")
    entry.add_to_hass(hass)
    result = await entry.start_reauth_flow(hass)
    mock_tfa_handler.async_get_tfa_options.return_value = []

    with patch(
        "custom_components.dominionsc.config_flow._validate_login",
        side_effect=MfaChallenge("TFA Required", mock_tfa_handler),
    ):
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"], user_input
        )

    assert result["step_id"] == "tfa_code"

    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {CONF_TFA_CODE: "123456"}
    )

    # Reauth completes without going through cost_mode.
    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "reauth_successful"


# ---------------------------------------------------------------------------
# Options flow — mode selection
# ---------------------------------------------------------------------------


async def test_options_flow_none(
    hass: HomeAssistant,
    mock_setup_entry: AsyncMock,
    user_input: dict,
) -> None:
    """Selecting None saves immediately with no sub-steps."""
    entry = _make_entry(hass, user_input)
    result = await hass.config_entries.options.async_init(entry.entry_id)

    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "init"

    result = await hass.config_entries.options.async_configure(
        result["flow_id"], {CONF_COST_MODE: COST_MODE_NONE}
    )

    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["data"] == {CONF_COST_MODE: COST_MODE_NONE}


async def test_options_flow_fixed_rate(
    hass: HomeAssistant,
    mock_setup_entry: AsyncMock,
    user_input: dict,
) -> None:
    """Selecting fixed rate shows the rate form, then recalculate_history."""
    entry = _make_entry(hass, user_input)
    result = await hass.config_entries.options.async_init(entry.entry_id)

    result = await hass.config_entries.options.async_configure(
        result["flow_id"], {CONF_COST_MODE: COST_MODE_FIXED}
    )
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "fixed_rate"

    result = await hass.config_entries.options.async_configure(
        result["flow_id"], {CONF_FIXED_RATE: 0.15}
    )
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "recalculate_history"

    result = await hass.config_entries.options.async_configure(
        result["flow_id"], {CONF_RECALCULATE_HISTORY: False}
    )

    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["data"] == {CONF_COST_MODE: COST_MODE_FIXED, CONF_FIXED_RATE: 0.15}


async def test_options_flow_rate_8(
    hass: HomeAssistant,
    mock_setup_entry: AsyncMock,
    user_input: dict,
) -> None:
    """Selecting Rate 8 goes straight to recalculate_history."""
    entry = _make_entry(hass, user_input)
    result = await hass.config_entries.options.async_init(entry.entry_id)

    result = await hass.config_entries.options.async_configure(
        result["flow_id"], {CONF_COST_MODE: COST_MODE_RATE_8}
    )
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "recalculate_history"

    result = await hass.config_entries.options.async_configure(
        result["flow_id"], {CONF_RECALCULATE_HISTORY: False}
    )

    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["data"] == {CONF_COST_MODE: COST_MODE_RATE_8}


async def test_options_flow_rate_6(
    hass: HomeAssistant,
    mock_setup_entry: AsyncMock,
    user_input: dict,
) -> None:
    """Selecting Rate 6 goes straight to recalculate_history."""
    entry = _make_entry(hass, user_input)
    result = await hass.config_entries.options.async_init(entry.entry_id)

    result = await hass.config_entries.options.async_configure(
        result["flow_id"], {CONF_COST_MODE: COST_MODE_RATE_6}
    )
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "recalculate_history"

    result = await hass.config_entries.options.async_configure(
        result["flow_id"], {CONF_RECALCULATE_HISTORY: False}
    )

    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["data"] == {CONF_COST_MODE: COST_MODE_RATE_6}


# ---------------------------------------------------------------------------
# Options flow — recalculate history
# ---------------------------------------------------------------------------


async def test_options_flow_recalculate_history_yes(
    hass: HomeAssistant,
    mock_setup_entry: AsyncMock,
    user_input: dict,
) -> None:
    """Accepting recalculation advances to the date-range step."""
    entry = _make_entry(hass, user_input)
    result = await hass.config_entries.options.async_init(entry.entry_id)
    result = await hass.config_entries.options.async_configure(
        result["flow_id"], {CONF_COST_MODE: COST_MODE_RATE_8}
    )

    result = await hass.config_entries.options.async_configure(
        result["flow_id"], {CONF_RECALCULATE_HISTORY: True}
    )

    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "recalculate_date_range"


async def test_options_flow_recalculate_date_range_success(
    hass: HomeAssistant,
    mock_setup_entry: AsyncMock,
    user_input: dict,
) -> None:
    """A valid date range triggers background recalculation and saves options."""
    entry = _make_entry(hass, user_input)

    mock_coordinator = MagicMock()
    mock_coordinator.recalculation_lock.locked.return_value = False
    mock_coordinator.async_recalculate_historic_costs = AsyncMock()
    hass.data.setdefault(DOMAIN, {})[entry.entry_id] = mock_coordinator

    result = await hass.config_entries.options.async_init(entry.entry_id)
    result = await hass.config_entries.options.async_configure(
        result["flow_id"], {CONF_COST_MODE: COST_MODE_RATE_8}
    )
    result = await hass.config_entries.options.async_configure(
        result["flow_id"], {CONF_RECALCULATE_HISTORY: True}
    )
    result = await hass.config_entries.options.async_configure(
        result["flow_id"],
        {CONF_RECALC_START_DATE: "2024-01-21", CONF_RECALC_END_DATE: "2024-02-20"},
    )

    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["data"] == {CONF_COST_MODE: COST_MODE_RATE_8}
    assert mock_coordinator.async_recalculate_historic_costs.called


async def test_options_flow_recalculate_invalid_date_format(
    hass: HomeAssistant,
    mock_setup_entry: AsyncMock,
    user_input: dict,
) -> None:
    """An unparseable date string shows an invalid_date_format error."""
    entry = _make_entry(hass, user_input)

    mock_coordinator = MagicMock()
    mock_coordinator.recalculation_lock.locked.return_value = False
    hass.data.setdefault(DOMAIN, {})[entry.entry_id] = mock_coordinator

    result = await hass.config_entries.options.async_init(entry.entry_id)
    result = await hass.config_entries.options.async_configure(
        result["flow_id"], {CONF_COST_MODE: COST_MODE_RATE_8}
    )
    result = await hass.config_entries.options.async_configure(
        result["flow_id"], {CONF_RECALCULATE_HISTORY: True}
    )
    result = await hass.config_entries.options.async_configure(
        result["flow_id"],
        {CONF_RECALC_START_DATE: "not-a-date", CONF_RECALC_END_DATE: "2024-02-20"},
    )

    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "recalculate_date_range"
    assert result["errors"] == {"base": "invalid_date_format"}


async def test_options_flow_recalculate_end_before_start(
    hass: HomeAssistant,
    mock_setup_entry: AsyncMock,
    user_input: dict,
) -> None:
    """End date before start date shows an end_before_start error."""
    entry = _make_entry(hass, user_input)

    mock_coordinator = MagicMock()
    mock_coordinator.recalculation_lock.locked.return_value = False
    hass.data.setdefault(DOMAIN, {})[entry.entry_id] = mock_coordinator

    result = await hass.config_entries.options.async_init(entry.entry_id)
    result = await hass.config_entries.options.async_configure(
        result["flow_id"], {CONF_COST_MODE: COST_MODE_RATE_8}
    )
    result = await hass.config_entries.options.async_configure(
        result["flow_id"], {CONF_RECALCULATE_HISTORY: True}
    )
    result = await hass.config_entries.options.async_configure(
        result["flow_id"],
        {CONF_RECALC_START_DATE: "2024-02-20", CONF_RECALC_END_DATE: "2024-01-01"},
    )

    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "recalculate_date_range"
    assert result["errors"] == {"base": "end_before_start"}


async def test_options_flow_recalculate_end_in_future(
    hass: HomeAssistant,
    mock_setup_entry: AsyncMock,
    user_input: dict,
) -> None:
    """End date in the future shows an end_in_future error."""
    entry = _make_entry(hass, user_input)

    mock_coordinator = MagicMock()
    mock_coordinator.recalculation_lock.locked.return_value = False
    hass.data.setdefault(DOMAIN, {})[entry.entry_id] = mock_coordinator

    result = await hass.config_entries.options.async_init(entry.entry_id)
    result = await hass.config_entries.options.async_configure(
        result["flow_id"], {CONF_COST_MODE: COST_MODE_RATE_8}
    )
    result = await hass.config_entries.options.async_configure(
        result["flow_id"], {CONF_RECALCULATE_HISTORY: True}
    )
    result = await hass.config_entries.options.async_configure(
        result["flow_id"],
        {CONF_RECALC_START_DATE: "2024-01-01", CONF_RECALC_END_DATE: "2099-12-31"},
    )

    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "recalculate_date_range"
    assert result["errors"] == {"base": "end_in_future"}


# ---------------------------------------------------------------------------
# Options flow — recalculation lock
# ---------------------------------------------------------------------------


async def test_options_flow_recalculation_in_progress(
    hass: HomeAssistant,
    mock_setup_entry: AsyncMock,
    user_input: dict,
) -> None:
    """Opening options while a recalculation is running shows an error."""
    entry = _make_entry(hass, user_input)

    mock_coordinator = MagicMock()
    mock_coordinator.recalculation_lock.locked.return_value = True
    hass.data.setdefault(DOMAIN, {})[entry.entry_id] = mock_coordinator

    result = await hass.config_entries.options.async_init(entry.entry_id)

    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "init"
    assert result["errors"] == {"base": "recalculation_in_progress"}


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


async def test_validate_login_helper(
    hass: HomeAssistant, mock_dominionsc_api: AsyncMock
) -> None:
    """The _validate_login helper instantiates the API and calls async_login."""
    await _validate_login(hass, {CONF_USERNAME: "test", CONF_PASSWORD: "test"})
    assert mock_dominionsc_api.async_login.called


def test_async_get_options_flow(hass: HomeAssistant) -> None:
    """async_get_options_flow returns a DominionSCOptionsFlow instance."""
    entry = MockConfigEntry(domain=DOMAIN, data={})
    flow = DominionSCConfigFlow.async_get_options_flow(entry)
    assert isinstance(flow, DominionSCOptionsFlow)
