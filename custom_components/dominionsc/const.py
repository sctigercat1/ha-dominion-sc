"""Constants for the dominionsc integration."""

import re
from typing import Final

DOMAIN = "dominionsc"
COMMON_NAME = "Dominion Energy SC"

CONF_LOGIN_DATA = "login_data"

# Number of days to backfill on first setup or statistics upgrade
# BACKFILL_DAYS = 30

# Options keys for cost configuration
CONF_COST_MODE: Final = "cost_mode"
CONF_FIXED_RATE: Final = "fixed_rate"

# Cost mode options
COST_MODE_NONE: Final = "none"
COST_MODE_FIXED: Final = "fixed"
COST_MODE_RATE_8: Final = "rate_8"
COST_MODE_RATE_6: Final = "rate_6"

# Default cost values
DEFAULT_FIXED_RATE: Final = 0.14164 / 1000  # $/Wh


def clean_service_addr(service_addr_account_no: str) -> str:
    """Normalise a service-address / account number into a safe identifier fragment."""
    return re.sub(r"[\W]+|^(?=\d)", "_", service_addr_account_no).strip("_").lower()
