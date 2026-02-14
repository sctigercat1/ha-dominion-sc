"""Rate calculator logic for dominionsc."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class Season(Enum):
    """Billing season."""

    SUMMER = "summer"  # May-September
    WINTER = "winter"  # October-April


@dataclass(frozen=True)
class TieredRate:
    """Rate with a Wh boundary (e.g., first 800 Wh vs. over 800 Wh)."""

    boundary_wh: float
    rate_under: float  # $/Wh for usage <= boundary
    rate_over: float  # $/Wh for usage > boundary


@dataclass(frozen=True)
class SeasonalTieredRates:
    """Summer and winter tiered rates."""

    summer: TieredRate
    winter: TieredRate


@dataclass(frozen=True)
class RateSchedule:
    """Complete rate schedule for a Dominion Energy SC tariff."""

    name: str
    effective_date: str
    basic_facilities_charge: float  # $/month flat charge
    rates: SeasonalTieredRates


# Dominion Energy South Carolina Rate 8 - Residential Service
# Effective for Bills Rendered On and After July 23, 2025
SC_RATE_8 = RateSchedule(
    name="Rate 8 - Residential Service",
    effective_date="2025-07-23",
    basic_facilities_charge=9.00,
    rates=SeasonalTieredRates(
        summer=TieredRate(
            boundary_wh=800 * 1000, rate_under=0.14599 / 1000, rate_over=0.15983 / 1000
        ),
        winter=TieredRate(
            boundary_wh=800 * 1000, rate_under=0.14599 / 1000, rate_over=0.14045 / 1000
        ),
    ),
)


# Dominion Energy South Carolina Rate 6 - Energy Saver / Conservation Rate
# Effective for Bills Rendered On and After July 23, 2025
SC_RATE_6 = RateSchedule(
    name="Rate 6 - Energy Saver / Conservation Rate",
    effective_date="2025-07-23",
    basic_facilities_charge=9.00,
    rates=SeasonalTieredRates(
        summer=TieredRate(
            boundary_wh=800 * 1000, rate_under=0.14164 / 1000, rate_over=0.15505 / 1000
        ),
        winter=TieredRate(
            boundary_wh=800 * 1000, rate_under=0.14164 / 1000, rate_over=0.13628 / 1000
        ),
    ),
)


def get_season(month: int) -> Season:
    """
    Determine billing season from month number (1-12).

    Summer: May-September (5-9)
    Winter: October-April (10-12, 1-4)
    """
    if 5 <= month <= 9:
        return Season.SUMMER
    return Season.WINTER


def calculate_tiered_cost(
    interval_wh: float,
    cumulative_before: float,
    tiered_rate: TieredRate,
) -> float:
    """
    Calculate cost for a single interval using tiered pricing.

    Handles the case where cumulative usage straddles the tier boundary
    within this interval.

    Args:
        interval_wh: Wh consumed in this interval.
        cumulative_before: Total Wh consumed before this interval in the billing period.
        tiered_rate: The tiered rate to apply.

    Returns:
        Cost in dollars for this interval.

    """
    boundary = tiered_rate.boundary_wh
    cumulative_after = cumulative_before + interval_wh

    if cumulative_after <= boundary:
        # All usage in lower tier
        return interval_wh * tiered_rate.rate_under
    if cumulative_before >= boundary:
        # All usage in upper tier
        return interval_wh * tiered_rate.rate_over

    # Straddles the boundary
    wh_under = boundary - cumulative_before
    wh_over = interval_wh - wh_under
    return wh_under * tiered_rate.rate_under + wh_over * tiered_rate.rate_over


def calculate_sc_rate_interval_cost(
    interval_wh: float,
    interval_dt: datetime,
    cumulative_before: float,
    schedule: RateSchedule,
) -> float:
    """
    Calculate SC rate cost for a single interval.

    Args:
        interval_wh: Wh consumed in this interval.
        interval_dt: Timestamp of the interval (used for season determination).
        cumulative_before: Total Wh consumed before this interval in the billing period.
        schedule: The rate schedule to use.

    Returns:
        Total cost in dollars for this interval.

    """
    if interval_wh <= 0:
        return 0.0

    season = get_season(interval_dt.month)

    # Get the appropriate tiered rate for the season
    tiered_rate = (
        schedule.rates.summer if season == Season.SUMMER else schedule.rates.winter
    )

    # Calculate energy cost with tiered pricing
    return calculate_tiered_cost(interval_wh, cumulative_before, tiered_rate)
