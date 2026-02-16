"""Tests for Dominion Energy SC config flow."""

from unittest.mock import AsyncMock, patch

from dominionsc.exceptions import ApiException, CannotConnect, InvalidAuth, MfaChallenge
from homeassistant import config_entries
from homeassistant.const import CONF_PASSWORD, CONF_USERNAME
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.dominionsc.config_flow import (
    CONF_TFA_CODE,
    CONF_TFA_METHOD,
    DominionSCConfigFlow,
    DominionSCOptionsFlow,
    _validate_login,
)
from custom_components.dominionsc.const import (
    CONF_COST_MODE,
    CONF_FIXED_RATE,
    CONF_LOGIN_DATA,
    COST_MODE_FIXED,
    COST_MODE_NONE,
    COST_MODE_RATE_6,
    COST_MODE_RATE_8,
    DOMAIN,
)


async def test_user_flow_success(
    hass: HomeAssistant,
    mock_dominionsc_api: AsyncMock,
    mock_setup_entry: AsyncMock,
    user_input: dict,
) -> None:
    """Test successful user flow without TFA."""
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "user"
    assert result["errors"] == {}

    with patch(
        "custom_components.dominionsc.config_flow._validate_login"
    ) as mock_validate:
        mock_validate.return_value = None
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"],
            user_input,
        )

    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["title"] == f"Dominion Energy SC ({user_input[CONF_USERNAME]})"
    assert result["data"] == user_input
    assert len(mock_setup_entry.mock_calls) == 1


async def test_user_flow_invalid_auth(
    hass: HomeAssistant,
    mock_dominionsc_api: AsyncMock,
    user_input: dict,
) -> None:
    """Test user flow with invalid authentication."""
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )

    with patch(
        "custom_components.dominionsc.config_flow._validate_login",
        side_effect=InvalidAuth("Invalid credentials"),
    ):
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"],
            user_input,
        )

    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "user"
    assert result["errors"] == {"base": "invalid_auth"}


async def test_user_flow_cannot_connect(
    hass: HomeAssistant,
    mock_dominionsc_api: AsyncMock,
    user_input: dict,
) -> None:
    """Test user flow when cannot connect."""
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )

    with patch(
        "custom_components.dominionsc.config_flow._validate_login",
        side_effect=CannotConnect("Connection error"),
    ):
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"],
            user_input,
        )

    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "user"
    assert result["errors"] == {"base": "cannot_connect"}


async def test_user_flow_api_exception(
    hass: HomeAssistant,
    mock_dominionsc_api: AsyncMock,
    user_input: dict,
) -> None:
    """Test user flow with API exception."""
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )

    with patch(
        "custom_components.dominionsc.config_flow._validate_login",
        side_effect=ApiException("API error", "https://test.com"),
    ):
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"],
            user_input,
        )

    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "user"
    assert result["errors"] == {"base": "unknown"}


async def test_user_flow_with_tfa(
    hass: HomeAssistant,
    mock_dominionsc_api: AsyncMock,
    mock_tfa_handler: AsyncMock,
    mock_setup_entry: AsyncMock,
    user_input: dict,
) -> None:
    """Test user flow with TFA challenge."""
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )

    # Trigger TFA challenge
    with patch(
        "custom_components.dominionsc.config_flow._validate_login",
        side_effect=MfaChallenge("TFA Required", mock_tfa_handler),
    ):
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"],
            user_input,
        )

    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "tfa_options"

    # Select TFA method
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"],
        {CONF_TFA_METHOD: "sms"},
    )

    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "tfa_code"

    # Submit TFA code
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"],
        {CONF_TFA_CODE: "123456"},
    )

    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["title"] == f"Dominion Energy SC ({user_input[CONF_USERNAME]})"
    assert CONF_LOGIN_DATA in result["data"]


async def test_tfa_options_cannot_connect(
    hass: HomeAssistant,
    mock_dominionsc_api: AsyncMock,
    mock_tfa_handler: AsyncMock,
    user_input: dict,
) -> None:
    """Test TFA options step with connection error."""
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )

    with patch(
        "custom_components.dominionsc.config_flow._validate_login",
        side_effect=MfaChallenge("TFA Required", mock_tfa_handler),
    ):
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"],
            user_input,
        )

    mock_tfa_handler.async_select_tfa_option.side_effect = CannotConnect(
        "Connection error"
    )

    result = await hass.config_entries.flow.async_configure(
        result["flow_id"],
        {CONF_TFA_METHOD: "sms"},
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
    """Test TFA options step with API exception."""
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )

    with patch(
        "custom_components.dominionsc.config_flow._validate_login",
        side_effect=MfaChallenge("TFA Required", mock_tfa_handler),
    ):
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"],
            user_input,
        )

    mock_tfa_handler.async_select_tfa_option.side_effect = ApiException(
        "API error", "https://test.com"
    )

    result = await hass.config_entries.flow.async_configure(
        result["flow_id"],
        {CONF_TFA_METHOD: "sms"},
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
    """Test TFA options step with API exception when getting options."""
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )

    mock_tfa_handler.async_get_tfa_options.side_effect = ApiException(
        "API error", "https://test.com"
    )

    with patch(
        "custom_components.dominionsc.config_flow._validate_login",
        side_effect=MfaChallenge("TFA Required", mock_tfa_handler),
    ):
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"],
            user_input,
        )

    # Should show error form instead of proceeding to tfa_code
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "tfa_options"
    assert result["errors"] == {"base": "unknown"}


async def test_tfa_code_invalid(
    hass: HomeAssistant,
    mock_dominionsc_api: AsyncMock,
    mock_tfa_handler: AsyncMock,
    user_input: dict,
) -> None:
    """Test TFA code step with invalid code."""
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )

    mock_tfa_handler.async_get_tfa_options.return_value = []

    with patch(
        "custom_components.dominionsc.config_flow._validate_login",
        side_effect=MfaChallenge("TFA Required", mock_tfa_handler),
    ):
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"],
            user_input,
        )

    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "tfa_code"

    mock_tfa_handler.async_submit_tfa_code.side_effect = InvalidAuth("Invalid TFA code")

    result = await hass.config_entries.flow.async_configure(
        result["flow_id"],
        {CONF_TFA_CODE: "wrong"},
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
    """Test TFA code step with connection error."""
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )

    mock_tfa_handler.async_get_tfa_options.return_value = []

    with patch(
        "custom_components.dominionsc.config_flow._validate_login",
        side_effect=MfaChallenge("TFA Required", mock_tfa_handler),
    ):
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"],
            user_input,
        )

    assert result["type"] is FlowResultType.FORM

    mock_tfa_handler.async_submit_tfa_code.side_effect = CannotConnect(
        "Connection error"
    )

    result = await hass.config_entries.flow.async_configure(
        result["flow_id"],
        {CONF_TFA_CODE: "123456"},
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
    """Test TFA code step with API exception."""
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )

    mock_tfa_handler.async_get_tfa_options.return_value = []

    with patch(
        "custom_components.dominionsc.config_flow._validate_login",
        side_effect=MfaChallenge("TFA Required", mock_tfa_handler),
    ):
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"],
            user_input,
        )

    assert result["type"] is FlowResultType.FORM

    mock_tfa_handler.async_submit_tfa_code.side_effect = ApiException(
        "API error", "https://test.com"
    )

    result = await hass.config_entries.flow.async_configure(
        result["flow_id"],
        {CONF_TFA_CODE: "123456"},
    )

    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "tfa_code"
    assert result["errors"] == {"base": "unknown"}


async def test_duplicate_entry(
    hass: HomeAssistant,
    mock_dominionsc_api: AsyncMock,
    mock_setup_entry: AsyncMock,
    user_input: dict,
) -> None:
    """Test that duplicate entries are aborted."""
    # Create first entry
    with patch("custom_components.dominionsc.config_flow._validate_login"):
        result = await hass.config_entries.flow.async_init(
            DOMAIN, context={"source": config_entries.SOURCE_USER}, data=user_input
        )
        assert result["type"] is FlowResultType.CREATE_ENTRY

    # Try to create duplicate
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )

    with patch("custom_components.dominionsc.config_flow._validate_login"):
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"],
            user_input,
        )

    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "already_configured"


async def test_reauth_flow_success(
    hass: HomeAssistant,
    mock_dominionsc_api: AsyncMock,
    mock_setup_entry: AsyncMock,
    user_input: dict,
) -> None:
    """Test successful reauth flow."""
    # Create initial entry
    entry = MockConfigEntry(
        domain=DOMAIN,
        data=user_input,
        title="Test",
    )
    entry.add_to_hass(hass)

    # Start reauth
    result = await entry.start_reauth_flow(hass)
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "reauth_confirm"

    # Complete reauth
    with patch("custom_components.dominionsc.config_flow._validate_login"):
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"],
            user_input,
        )

    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "reauth_successful"


async def test_reauth_flow_invalid_auth(
    hass: HomeAssistant,
    mock_dominionsc_api: AsyncMock,
    user_input: dict,
) -> None:
    """Test reauth flow with invalid credentials."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        data=user_input,
        title="Test",
    )
    entry.add_to_hass(hass)

    result = await entry.start_reauth_flow(hass)

    with patch(
        "custom_components.dominionsc.config_flow._validate_login",
        side_effect=InvalidAuth("Invalid credentials"),
    ):
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"],
            user_input,
        )

    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "reauth_confirm"
    assert result["errors"] == {"base": "invalid_auth"}


async def test_reauth_flow_cannot_connect(
    hass: HomeAssistant,
    mock_dominionsc_api: AsyncMock,
    user_input: dict,
) -> None:
    """Test reauth flow with connection error."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        data=user_input,
        title="Test",
    )
    entry.add_to_hass(hass)

    result = await entry.start_reauth_flow(hass)

    with patch(
        "custom_components.dominionsc.config_flow._validate_login",
        side_effect=CannotConnect("Connection error"),
    ):
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"],
            user_input,
        )

    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "reauth_confirm"
    assert result["errors"] == {"base": "cannot_connect"}


async def test_reauth_flow_api_exception(
    hass: HomeAssistant,
    mock_dominionsc_api: AsyncMock,
    user_input: dict,
) -> None:
    """Test reauth flow with API exception."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        data=user_input,
        title="Test",
    )
    entry.add_to_hass(hass)

    result = await entry.start_reauth_flow(hass)

    with patch(
        "custom_components.dominionsc.config_flow._validate_login",
        side_effect=ApiException("API error", "https://test.com"),
    ):
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"],
            user_input,
        )

    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "reauth_confirm"
    assert result["errors"] == {"base": "unknown"}


async def test_reauth_flow_with_tfa(
    hass: HomeAssistant,
    mock_dominionsc_api: AsyncMock,
    mock_tfa_handler: AsyncMock,
    user_input: dict,
) -> None:
    """Test reauth flow with TFA."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        data=user_input,
        title="Test",
    )
    entry.add_to_hass(hass)
    result = await entry.start_reauth_flow(hass)
    mock_tfa_handler.async_get_tfa_options.return_value = []

    with patch(
        "custom_components.dominionsc.config_flow._validate_login",
        side_effect=MfaChallenge("TFA Required", mock_tfa_handler),
    ):
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"],
            user_input,
        )

    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "tfa_code"

    result = await hass.config_entries.flow.async_configure(
        result["flow_id"],
        {CONF_TFA_CODE: "123456"},
    )

    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "reauth_successful"


async def test_options_flow_none(
    hass: HomeAssistant,
    mock_setup_entry: AsyncMock,
    user_input: dict,
) -> None:
    """Test options flow selecting none."""
    with patch("custom_components.dominionsc.config_flow._validate_login"):
        entry = await hass.config_entries.flow.async_init(
            DOMAIN, context={"source": config_entries.SOURCE_USER}, data=user_input
        )
        config_entry = hass.config_entries.async_get_entry(entry["result"].entry_id)

    result = await hass.config_entries.options.async_init(config_entry.entry_id)
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "init"

    result = await hass.config_entries.options.async_configure(
        result["flow_id"],
        {CONF_COST_MODE: COST_MODE_NONE},
    )

    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["data"] == {CONF_COST_MODE: COST_MODE_NONE}


async def test_options_flow_fixed_rate(
    hass: HomeAssistant,
    mock_setup_entry: AsyncMock,
    user_input: dict,
) -> None:
    """Test options flow with fixed rate."""
    with patch("custom_components.dominionsc.config_flow._validate_login"):
        entry = await hass.config_entries.flow.async_init(
            DOMAIN, context={"source": config_entries.SOURCE_USER}, data=user_input
        )
        config_entry = hass.config_entries.async_get_entry(entry["result"].entry_id)

    result = await hass.config_entries.options.async_init(config_entry.entry_id)

    result = await hass.config_entries.options.async_configure(
        result["flow_id"],
        {CONF_COST_MODE: COST_MODE_FIXED},
    )

    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "fixed_rate"

    result = await hass.config_entries.options.async_configure(
        result["flow_id"],
        {CONF_FIXED_RATE: 0.15},
    )

    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["data"] == {
        CONF_COST_MODE: COST_MODE_FIXED,
        CONF_FIXED_RATE: 0.15,
    }


async def test_options_flow_rate_8(
    hass: HomeAssistant,
    mock_setup_entry: AsyncMock,
    user_input: dict,
) -> None:
    """Test options flow with Rate 8."""
    with patch("custom_components.dominionsc.config_flow._validate_login"):
        entry = await hass.config_entries.flow.async_init(
            DOMAIN, context={"source": config_entries.SOURCE_USER}, data=user_input
        )
        config_entry = hass.config_entries.async_get_entry(entry["result"].entry_id)

    result = await hass.config_entries.options.async_init(config_entry.entry_id)

    result = await hass.config_entries.options.async_configure(
        result["flow_id"],
        {CONF_COST_MODE: COST_MODE_RATE_8},
    )

    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "rate8"

    result = await hass.config_entries.options.async_configure(
        result["flow_id"],
        {},
    )

    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["data"] == {CONF_COST_MODE: COST_MODE_RATE_8}


async def test_options_flow_rate_6(
    hass: HomeAssistant,
    mock_setup_entry: AsyncMock,
    user_input: dict,
) -> None:
    """Test options flow with Rate 6."""
    with patch("custom_components.dominionsc.config_flow._validate_login"):
        entry = await hass.config_entries.flow.async_init(
            DOMAIN, context={"source": config_entries.SOURCE_USER}, data=user_input
        )
        config_entry = hass.config_entries.async_get_entry(entry["result"].entry_id)

    result = await hass.config_entries.options.async_init(config_entry.entry_id)

    result = await hass.config_entries.options.async_configure(
        result["flow_id"],
        {CONF_COST_MODE: COST_MODE_RATE_6},
    )

    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "rate6"

    result = await hass.config_entries.options.async_configure(
        result["flow_id"],
        {},
    )

    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["data"] == {CONF_COST_MODE: COST_MODE_RATE_6}


async def test_validate_login_helper(
    hass: HomeAssistant, mock_dominionsc_api: AsyncMock
) -> None:
    """Test the _validate_login helper directly to cover its logic."""
    await _validate_login(hass, {CONF_USERNAME: "test", CONF_PASSWORD: "test"})

    # Verify the API was instantiated and login called
    assert mock_dominionsc_api.async_login.called


def test_async_get_options_flow(hass: HomeAssistant) -> None:
    """Verify async_get_options_flow returns the correct handler."""
    entry = MockConfigEntry(domain=DOMAIN, data={})
    flow = DominionSCConfigFlow.async_get_options_flow(entry)
    assert isinstance(flow, DominionSCOptionsFlow)


async def test_reauth_confirm_no_input(
    hass: HomeAssistant,
    mock_dominionsc_api: AsyncMock,
    user_input: dict,
) -> None:
    """Test reauth confirm step when called without input."""
    entry = MockConfigEntry(domain=DOMAIN, data=user_input)
    entry.add_to_hass(hass)

    # Trigger reauth
    result = await entry.start_reauth_flow(hass)

    # Manually trigger the confirm step with NO input
    with patch("custom_components.dominionsc.config_flow._validate_login"):
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"],
            None,  # Explicitly pass None
        )

    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "reauth_confirm"
