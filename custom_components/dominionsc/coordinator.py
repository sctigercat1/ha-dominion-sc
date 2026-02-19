"""Coordinator to handle dominionsc connections."""

import asyncio
import calendar
import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from string import Template
from typing import Any

from homeassistant.components.recorder import get_instance
from homeassistant.components.recorder.models import (
    StatisticData,
    StatisticMeanType,
    StatisticMetaData,
)
from homeassistant.components.recorder.statistics import (
    async_add_external_statistics,
    get_last_statistics,
    statistics_during_period,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_PASSWORD, CONF_USERNAME, UnitOfEnergy, UnitOfVolume
from homeassistant.core import HomeAssistant, callback
from homeassistant.exceptions import ConfigEntryAuthFailed
from homeassistant.helpers.aiohttp_client import async_create_clientsession
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed
from homeassistant.util import dt as dt_util
from homeassistant.util.unit_conversion import EnergyConverter, VolumeConverter

from dominionsc import (
    DominionSC,
    Forecast,
    create_cookie_jar,
)
from dominionsc.exceptions import ApiException, CannotConnect, InvalidAuth, MfaChallenge

from .const import (
    CONF_COST_MODE,
    CONF_EXTENDED_BACKFILL,
    CONF_EXTENDED_COST_BACKFILL,
    CONF_FIXED_RATE,
    CONF_LOGIN_DATA,
    COST_MODE_FIXED,
    COST_MODE_NONE,
    COST_MODE_RATE_8,
    DEFAULT_FIXED_RATE,
    DOMAIN,
    EXTENDED_BACKFILL_DAYS,
    clean_service_addr,
)
from .rates import TIERED_RATE_REGISTRY, RateSchedule, calculate_sc_rate_interval_cost

_LOGGER = logging.getLogger(__name__)

type DominionSCConfigEntry = ConfigEntry[DominionSCCoordinator]


@dataclass
class DominionSCStatisticMetadata:
    """Metadata for creating statistics."""

    account: str
    consumption_id: str
    cost_id: str | None
    name_prefix: Template
    unit_class: str
    unit: str


@dataclass
class DominionSCAccountData:
    """Class to hold DominionSC account-specific data."""

    account: str
    last_changed: datetime | None


@dataclass
class DominionSCData:
    """Class to hold all DominionSC shared data and individual accounts."""

    accounts: dict[str, DominionSCAccountData]
    forecast: Forecast | None
    service_addr_account_no: str
    last_updated: datetime


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _build_statistic_ids(
    service_addr_account_no: str,
    account: str,
) -> tuple[str, str | None, Template]:
    """
    Construct the statistic IDs and name prefix for a given account.

    Returns:
        (consumption_statistic_id, cost_statistic_id, name_prefix)
        ``cost_statistic_id`` is ``None`` for non-ELECTRIC accounts.

    """
    clean_addr = clean_service_addr(service_addr_account_no)
    id_prefix = (f"{clean_addr}_{account}").lower().replace("-", "_")
    consumption_id = f"{DOMAIN}:{id_prefix}_energy_consumption"
    cost_id = f"{DOMAIN}:{id_prefix}_energy_cost" if account == "ELECTRIC" else None
    name_prefix = Template(f"{account.title()} $stat_type {service_addr_account_no}")
    return consumption_id, cost_id, name_prefix


def _resolve_cost_config(
    options: dict[str, Any],
) -> tuple[str, float, RateSchedule | None]:
    """Return (cost_mode, fixed_rate, rate_schedule) with consistent defaults."""
    cost_mode = options.get(CONF_COST_MODE, COST_MODE_RATE_8)
    fixed_rate: float = options.get(CONF_FIXED_RATE, DEFAULT_FIXED_RATE)
    rate_schedule: RateSchedule | None = TIERED_RATE_REGISTRY.get(cost_mode)
    return cost_mode, fixed_rate, rate_schedule


def _calculate_cost_for_wh(
    interval_wh: float,
    interval_dt: datetime,
    cumulative_wh_before: float,
    cost_mode: str,
    fixed_rate: float,
    rate_schedule: RateSchedule | None,
) -> float:
    """
    Calculate the cost for a single Wh interval under the given cost mode.

    Args:
        interval_wh:          Wh consumed in this interval.
        interval_dt:          Timestamp of the interval (used for season).
        cumulative_wh_before: Total Wh consumed before this interval in the
                              billing period (used for tier boundary tracking).
        cost_mode:            One of the COST_MODE_* constants.
        fixed_rate:           $/kWh fixed rate (only used when cost_mode is FIXED).
        rate_schedule:        RateSchedule instance (only used for tiered modes).
                              Pass ``None`` to produce 0.0 cost (e.g. outside the
                              billing period for tiered rates).

    Returns:
        Cost in dollars for this interval.

    """
    if cost_mode == COST_MODE_NONE:
        return 0.0
    if cost_mode == COST_MODE_FIXED:
        return interval_wh * (fixed_rate / 1000)
    if rate_schedule is not None:
        # Skip intervals that predate the rate schedule's effective date
        # so extended backfills don't apply current rates to old data.
        if interval_dt.date() < rate_schedule.effective_date:
            return 0.0
        return calculate_sc_rate_interval_cost(
            interval_wh,
            interval_dt,
            cumulative_wh_before,
            rate_schedule,
        )
    return 0.0


def _billing_cycle_get_gap(d: date) -> int:
    """Return billing cycle gap for provided start date."""
    gaps = {
        1: 30,
        2: 31,
        3: 30,
        5: 30,
        7: 30,
        8: 31,
        9: 31,
        10: 30,
        11: 31,
        12: 30,
    }
    m = d.month
    if m in gaps:
        return gaps[m]
    near_leap = calendar.isleap(d.year) or calendar.isleap(d.year + 1)
    if m == 4:
        return 31 if near_leap else 30
    if m == 6:
        return 30 if near_leap else 31
    return 30


def _estimate_billing_cycles(
    anchor_start: date,
    anchor_end: date,
    earliest: date,
) -> list[tuple[date, date]]:
    """
    Estimate billing cycle intervals given one known (current) billing cycle.
    NOTE: this is a temporary algorithm.

    Args:
        anchor_start: Start date of the known (current) billing cycle.
        anchor_end:   End date of the known (current) billing cycle.
        earliest:     Generate cycles back to (at least) this date.

    Returns:
        List of (start_date, end_date) tuples, ordered oldest-first.
        The last element is the anchor (current) cycle.

    """
    # --- walk backward from the anchor to *earliest* ---
    starts: list[date] = [anchor_start]
    cur = anchor_start
    while cur > earliest:
        prev_month = cur.month - 1 or 12
        prev_year = cur.year - (1 if cur.month == 1 else 0)
        cur -= timedelta(days=_billing_cycle_get_gap(date(prev_year, prev_month, 1)))
        starts.append(cur)
    starts.reverse()

    # --- build (start, end) pairs; end = next_start - 1 ---
    cycles = [
        (starts[i], starts[i + 1] - timedelta(days=1)) for i in range(len(starts) - 1)
    ]
    cycles.append((anchor_start, anchor_end))
    return cycles


def _find_billing_cycle_for_date(
    target: date,
    billing_cycles: list[tuple[date, date]],
) -> tuple[date, date] | None:
    """Return the billing cycle that contains *target*, or ``None``."""
    for start, end in billing_cycles:
        if start <= target <= end:
            return (start, end)
    return None


class DominionSCCoordinator(DataUpdateCoordinator[DominionSCData]):
    """Handle fetching DominionSC data, updating sensors and inserting statistics."""

    config_entry: DominionSCConfigEntry

    def __init__(
        self,
        hass: HomeAssistant,
        config_entry: DominionSCConfigEntry,
    ) -> None:
        """Initialize the data handler."""
        super().__init__(
            hass,
            _LOGGER,
            config_entry=config_entry,
            name=DOMAIN,
            # Data is updated daily on DominionSC.
            # Refresh every 12h to be at most 12h behind.
            update_interval=timedelta(hours=12),
        )
        self.api = DominionSC(
            async_create_clientsession(hass, cookie_jar=create_cookie_jar()),
            config_entry.data[CONF_USERNAME],
            config_entry.data[CONF_PASSWORD],
            config_entry.data.get(CONF_LOGIN_DATA),
        )
        # Track if backfill has been initiated to prevent race condition
        # where recorder hasn't committed stats yet and backfill runs again
        self._backfill_initiated: dict[str, bool] = {}
        # Lock held during historic cost recalculation so the options flow
        # can detect that a recalculation is still running and block a second one.
        self.recalculation_lock = asyncio.Lock()

        @callback
        def _dummy_listener() -> None:
            pass

        # Force the coordinator to periodically update by registering at least one
        # listener. Needed when the _async_update_data below returns {} for utilities
        # that don't provide forecast, which results to no sensors added, no
        # registered listeners, and thus _async_update_data not periodically
        # getting called which is needed for _insert_statistics.
        self.async_add_listener(_dummy_listener)

    async def _async_update_data(
        self,
    ) -> DominionSCData:
        """Fetch data from API endpoint."""
        try:
            # Login expires after a few minutes.
            # Given the infrequent updating (every 12h)
            # assume previous session has expired and re-login.
            _LOGGER.debug("API: async_login")
            await self.api.async_login()
        except (InvalidAuth, MfaChallenge) as err:
            _LOGGER.error("Error during login: %s", err)
            raise ConfigEntryAuthFailed from err
        except CannotConnect as err:
            _LOGGER.error("Error during login: %s", err)
            raise UpdateFailed from err
        except ApiException as err:
            _LOGGER.error("Error during login: %s", err)
            raise

        accounts, service_addr_account_no = await self.api.async_get_accounts()

        try:
            _LOGGER.debug("API: async_get_forecast")
            forecast = await self.api.async_get_forecast()
        except CannotConnect as err:
            _LOGGER.error("Error getting forecast: %s", err)
            raise UpdateFailed from err
        except ApiException as err:
            _LOGGER.error("Error getting forecast: %s", err)
            raise

        _LOGGER.debug("Updating sensor data with: %s", forecast)

        # Because DominionSC provides historical usage with a delay of a couple of days
        # we need to insert data into statistics.
        last_changed_per_account = await self._insert_statistics(
            accounts, service_addr_account_no, forecast
        )

        # Build account-specific data dictionary
        account_data = {
            account: DominionSCAccountData(
                account=account,
                last_changed=last_changed_per_account.get(account),
            )
            for account in accounts
        }

        # Return combined struct with accounts and shared data
        return DominionSCData(
            accounts=account_data,
            forecast=forecast,
            service_addr_account_no=service_addr_account_no,
            last_updated=dt_util.utcnow(),
        )

    def _push_cost_statistics(
        self,
        cost_statistic_id: str,
        cost_stat_name: str,
        cost_statistics: list[StatisticData],
        operation_type: str,
        final_sum: float,
    ) -> None:
        """
        Build cost ``StatisticMetaData`` and push rows to the recorder.

        Shared by ``_process_and_insert_statistics`` and
        ``_async_recalculate_historic_costs_locked``, which previously duplicated
        this block verbatim.
        """
        cost_metadata = StatisticMetaData(
            mean_type=StatisticMeanType.NONE,
            has_sum=True,
            name=cost_stat_name,
            source=DOMAIN,
            statistic_id=cost_statistic_id,
            unit_class=None,
            unit_of_measurement=None,
        )
        _LOGGER.info(
            "Adding %d hourly cost statistics for %s (%s, sum=%.4f)",
            len(cost_statistics),
            cost_statistic_id,
            operation_type,
            final_sum,
        )
        async_add_external_statistics(self.hass, cost_metadata, cost_statistics)

    async def _insert_statistics(
        self,
        accounts: list[str],
        service_addr_account_no: str,
        forecast: Forecast | None,
    ) -> dict[str, datetime]:
        """Insert DominionSC statistics."""
        last_changed_per_account: dict[str, datetime] = {}
        for account in accounts:
            consumption_statistic_id, cost_statistic_id, name_prefix = (
                _build_statistic_ids(service_addr_account_no, account)
            )

            # Only track cost for electric accounts when a cost mode is active.
            cost_mode, _, _ = _resolve_cost_config(self.config_entry.options)
            if account != "ELECTRIC" or cost_mode == COST_MODE_NONE:
                cost_statistic_id = None

            _LOGGER.debug("Updating Statistics for %s", consumption_statistic_id)

            consumption_unit_class = (
                EnergyConverter.UNIT_CLASS
                if account == "ELECTRIC"
                else VolumeConverter.UNIT_CLASS
            )
            consumption_unit = (
                UnitOfEnergy.WATT_HOUR
                if account == "ELECTRIC"
                else UnitOfVolume.CUBIC_FEET
            )

            # Check if we have existing statistics
            last_stat = await get_instance(self.hass).async_add_executor_job(
                get_last_statistics,
                self.hass,
                1,
                consumption_statistic_id,
                True,
                {"sum"},
            )

            consumption_exists = bool(last_stat.get(consumption_statistic_id))

            # Also check for cost statistics (for electric accounts)
            last_cost_stat = {}
            if cost_statistic_id:
                last_cost_stat = await get_instance(self.hass).async_add_executor_job(
                    get_last_statistics, self.hass, 1, cost_statistic_id, True, {"sum"}
                )

            if not consumption_exists:
                # No statistics - perform initial backfill
                if self._backfill_initiated.get(account, False):
                    # Backfill was already started, waiting for recorder to commit
                    _LOGGER.debug(
                        "Backfill already initiated for %s, "
                        "waiting for recorder to commit",
                        consumption_statistic_id,
                    )
                    continue

                _LOGGER.info(
                    "First statistics update for %s - "
                    "backfilling since last billing cycle.",
                    account,
                )
                self._backfill_initiated[account] = True

            dominionsc_metadata = DominionSCStatisticMetadata(
                account=account,
                consumption_id=consumption_statistic_id,
                cost_id=cost_statistic_id,
                name_prefix=name_prefix,
                unit_class=consumption_unit_class,
                unit=consumption_unit,
            )

            if not consumption_exists:
                await self._backfill_statistics(
                    dominionsc_metadata,
                    last_changed_per_account,
                    forecast,
                )
            else:
                # Statistics exist - perform incremental update
                self._backfill_initiated[account] = False
                _LOGGER.debug(
                    "Found existing statistics for %s, performing incremental update",
                    consumption_statistic_id,
                )

                await self._update_statistics(
                    dominionsc_metadata,
                    last_stat,
                    last_cost_stat,
                    last_changed_per_account,
                    forecast,
                )

        return last_changed_per_account

    async def _backfill_statistics(
        self,
        metadata: DominionSCStatisticMetadata,
        last_changed_per_account: dict[str, datetime],
        forecast: Forecast | None,
    ) -> None:
        """Backfill historical statistics for initial setup."""
        today = date.today()
        billing_cycle_start = forecast.start_date
        data_date = today - timedelta(days=1)  # Yesterday

        extended = self.config_entry.options.get(CONF_EXTENDED_BACKFILL, False)
        extended_cost = self.config_entry.options.get(
            CONF_EXTENDED_COST_BACKFILL, False
        )

        if extended:
            start_date = today - timedelta(days=EXTENDED_BACKFILL_DAYS)
        else:
            start_date = billing_cycle_start

        # When consumption is extended but cost is not, limit cost to the
        # current billing cycle.  When both are extended (or neither is),
        # cost_start_date is None which means no additional restriction.
        cost_start_date: date | None = None
        if extended and not extended_cost:
            cost_start_date = billing_cycle_start

        _LOGGER.debug(
            "Backfilling statistics from %s to %s "
            "(extended=%s, extended_cost=%s, billing_cycle_start=%s)",
            start_date,
            data_date,
            extended,
            extended_cost,
            billing_cycle_start,
        )

        # Call the common processing function with initial values
        await self._process_and_insert_statistics(
            metadata=metadata,
            start_date=start_date,
            data_date=data_date,
            consumption_sum=0.0,  # Start from 0 for backfill
            cost_sum=0.0,  # Start from 0 for backfill
            last_stat_dt=None,  # No previous stat for backfill
            last_changed_per_account=last_changed_per_account,
            forecast=forecast,
            cost_start_date=cost_start_date,
        )

    async def _update_statistics(
        self,
        metadata: DominionSCStatisticMetadata,
        last_stat: dict,
        last_cost_stat: dict,
        last_changed_per_account: dict[str, datetime],
        forecast: Forecast | None,
    ) -> None:
        """Update statistics with new data since last recorded statistic."""
        try:
            # Get the last recorded statistic time and sum
            last_stat_data = last_stat[metadata.consumption_id][0]
            last_stat_start = last_stat_data["start"]
            consumption_sum = float(last_stat_data.get("sum") or 0)

            _LOGGER.debug(
                "Last statistic for %s: start=%s (type=%s), sum=%.3f",
                metadata.consumption_id,
                last_stat_start,
                type(last_stat_start).__name__,
                consumption_sum,
            )

            # Convert to datetime for comparison
            if isinstance(last_stat_start, (int, float)):
                last_stat_dt = datetime.fromtimestamp(last_stat_start, tz=dt_util.UTC)
            else:
                last_stat_dt = last_stat_start

            # Convert to local timezone for date comparison
            local_tz = dt_util.get_default_time_zone()
            last_stat_local = last_stat_dt.astimezone(local_tz)
            last_stat_date = last_stat_local.date()

        except (KeyError, IndexError, TypeError, ValueError) as err:
            _LOGGER.warning(
                "Error parsing last statistic for %s: %s (last_stat=%s)",
                metadata.consumption_id,
                err,
                last_stat,
            )
            return

        # Get the last cost sum (default to 0 if cost stats don't exist yet)
        cost_sum = 0.0
        if metadata.cost_id:
            try:
                cost_sum = float(last_cost_stat[metadata.cost_id][0].get("sum") or 0)
            except (KeyError, IndexError, TypeError, ValueError) as err:
                _LOGGER.debug(
                    "WARNING: Unable to get cost_sum: %s, %s, %s",
                    str(last_cost_stat),
                    metadata.cost_id,
                    err,
                )

        # Determine the date range to fetch
        today = date.today()
        data_date = today - timedelta(
            days=1
        )  # Yesterday is the most recent complete day

        _LOGGER.debug(
            "Date comparison: last_stat_date=%s, data_date=%s",
            last_stat_date,
            data_date,
        )

        # Check if we need to fetch new data
        if last_stat_date >= data_date:
            _LOGGER.debug(
                "Statistics already up to date: last_stat_date=%s >= data_date=%s",
                last_stat_date,
                data_date,
            )
            # Update last_changed with existing data
            last_changed_per_account[metadata.account] = last_stat_dt
            return

        # Fetch data from day after last stat to data_date
        start_date = last_stat_date + timedelta(days=1)
        # start_date = last_stat_date - timedelta(days=2)

        # Limit max pull to BACKFILL_DAYS if very stale data
        oldest_available = forecast.start_date  # today - timedelta(days=BACKFILL_DAYS)
        if start_date < oldest_available:
            _LOGGER.warning(
                "Statistics are very stale (last: %s). "
                "Limiting fetch to last billing cycle. "
                "Some historical data may be lost.",
                last_stat_date,
            )
            start_date = oldest_available

        _LOGGER.info(
            "Fetching statistics update from %s to %s (consumption_sum=%.3f%s)",
            start_date,
            data_date,
            consumption_sum,
            f", cost_sum={cost_sum:.3f}" if metadata.cost_id else "",
        )

        # Call the common processing function
        await self._process_and_insert_statistics(
            metadata=metadata,
            start_date=start_date,
            data_date=data_date,
            consumption_sum=consumption_sum,
            cost_sum=cost_sum,
            last_stat_dt=last_stat_dt,
            last_changed_per_account=last_changed_per_account,
            forecast=forecast,
        )

    async def _process_and_insert_statistics(
        self,
        metadata: DominionSCStatisticMetadata,
        start_date: date,
        data_date: date,
        consumption_sum: float,
        cost_sum: float,
        last_stat_dt: datetime | None,
        last_changed_per_account: dict[str, datetime],
        forecast: Forecast | None,
        cost_start_date: date | None = None,
    ) -> None:
        """
        Process usage data and insert statistics.
        Common logic for backfill and updates.
        """
        # Convert dates to datetimes for API call
        tz = await dt_util.async_get_time_zone(self.api.get_timezone())
        start = datetime.combine(start_date, datetime.min.time()).replace(tzinfo=tz)
        end = datetime.combine(data_date, datetime.min.time()).replace(tzinfo=tz)

        # Track the initial sum to determine if this is a backfill or update
        initial_sum = consumption_sum

        try:
            _LOGGER.debug("API: async_get_usage_reads")
            usage_reads = await self.api.async_get_usage_reads(
                metadata.account, start, end
            )
        except CannotConnect as err:
            _LOGGER.warning("Could not fetch statistics data: %s", err)
            return
        except ApiException as err:
            _LOGGER.warning("Could not fetch statistics data: %s", err)
            return

        if not usage_reads:
            _LOGGER.debug(
                "No interval data for statistics (requested %s to %s). "
                "API may not have data available yet.",
                start_date,
                data_date,
            )
            # Set last_changed to the last statistic time if we have one
            if last_stat_dt:
                last_changed_per_account[metadata.account] = last_stat_dt
            return

        # Calculate daily totals to identify zero-consumption days (API may return
        # zeros when data isn't available yet, e.g., during holidays)
        daily_totals: dict[date, float] = {}
        for usage_read in usage_reads:
            d = usage_read.start_time.date()
            daily_totals.setdefault(d, 0.0)
            daily_totals[d] += usage_read.consumption

        # Filter out intervals from zero-consumption days
        zero_days = {d for d, total in daily_totals.items() if total == 0}
        if zero_days:
            _LOGGER.warning(
                "Skipping %d days with zero consumption (data not yet available): %s",
                len(zero_days),
                sorted(zero_days),
            )
            usage_reads = [
                i for i in usage_reads if i.start_time.date() not in zero_days
            ]

        if not usage_reads:
            _LOGGER.debug("No valid interval data after filtering zero days")
            # Set last_changed to the last statistic time if we have one
            if last_stat_dt:
                last_changed_per_account[metadata.account] = last_stat_dt
            return

        _LOGGER.debug("Received %d intervals for statistics", len(usage_reads))

        # Group intervals by hour for hourly statistics
        # For electric accounts with cost tracking, also calculate costs
        is_electric = metadata.account == "ELECTRIC"
        cost_mode_here, fixed_rate_here, rate_schedule_here = _resolve_cost_config(
            self.config_entry.options
        )
        is_tiered_rate = cost_mode_here in TIERED_RATE_REGISTRY
        cumulative_wh = 0.0

        billing_cycles: list[tuple[date, date]] = []
        if is_tiered_rate:
            billing_cycles = _estimate_billing_cycles(
                anchor_start=forecast.start_date,
                anchor_end=forecast.end_date,
                earliest=start_date,
            )
            _LOGGER.debug(
                "Estimated %d billing cycles from %s to %s for tier tracking",
                len(billing_cycles),
                billing_cycles[0][0] if billing_cycles else "?",
                billing_cycles[-1][1] if billing_cycles else "?",
            )

        current_cycle: tuple[date, date] | None = None

        hourly_consumption: dict[datetime, float] = {}
        hourly_cost: dict[datetime, float] = {}

        for usage_read in sorted(usage_reads, key=lambda i: i.start_time):
            interval_date = usage_read.start_time.date()

            hour_start = usage_read.start_time.replace(
                minute=0, second=0, microsecond=0
            )
            if hour_start not in hourly_consumption:
                hourly_consumption[hour_start] = 0.0

            hourly_consumption[hour_start] += usage_read.consumption

            # Calculate cost for electric accounts with cost tracking
            if is_electric and metadata.cost_id:
                # Skip cost for intervals before cost_start_date when set
                # (e.g. extended consumption backfill without extended cost)
                if cost_start_date is not None and interval_date < cost_start_date:
                    cumulative_wh += usage_read.consumption
                    continue

                # For tiered rates, reset cumulative Wh at billing cycle
                # boundaries so the tier threshold is applied per-cycle.
                if is_tiered_rate and billing_cycles:
                    row_cycle = _find_billing_cycle_for_date(
                        interval_date, billing_cycles
                    )
                    if row_cycle != current_cycle:
                        current_cycle = row_cycle
                        cumulative_wh = 0.0

                if hour_start not in hourly_cost:
                    hourly_cost[hour_start] = 0.0

                hourly_cost[hour_start] += _calculate_cost_for_wh(
                    usage_read.consumption,
                    usage_read.start_time,
                    cumulative_wh,
                    cost_mode_here,
                    fixed_rate_here,
                    rate_schedule_here,
                )

                cumulative_wh += usage_read.consumption

        # Build statistics with cumulative sums
        consumption_statistics: list[StatisticData] = []
        cost_statistics: list[StatisticData] = []

        for hour_start in sorted(hourly_consumption.keys()):
            consumption = hourly_consumption[hour_start]
            consumption_sum += consumption
            consumption_statistics.append(
                StatisticData(start=hour_start, state=consumption, sum=consumption_sum)
            )

        if hourly_cost:
            for hour_start in sorted(hourly_cost.keys()):
                cost = hourly_cost[hour_start]
                if cost > 0:
                    cost_sum += cost
                    cost_statistics.append(
                        StatisticData(start=hour_start, state=cost, sum=cost_sum)
                    )

        if not consumption_statistics:
            return

        # Update last_changed with the latest interval time
        last_changed_per_account[metadata.account] = usage_reads[-1].end_time

        # Create metadata for consumption
        consumption_metadata = StatisticMetaData(
            mean_type=StatisticMeanType.NONE,
            has_sum=True,
            name=metadata.name_prefix.substitute(stat_type="consumption"),
            source=DOMAIN,
            statistic_id=metadata.consumption_id,
            unit_class=metadata.unit_class,
            unit_of_measurement=metadata.unit,
        )

        operation_type = "backfill" if initial_sum == 0.0 else "update"
        _LOGGER.info(
            "Adding %d hourly statistics for %s (%s, sum=%.3f)",
            len(consumption_statistics),
            metadata.consumption_id,
            operation_type,
            consumption_sum,
        )
        async_add_external_statistics(
            self.hass, consumption_metadata, consumption_statistics
        )

        # Add cost statistics for electric accounts
        if is_electric and metadata.cost_id and cost_statistics:
            self._push_cost_statistics(
                metadata.cost_id,
                metadata.name_prefix.substitute(stat_type="cost"),
                cost_statistics,
                operation_type,
                cost_sum,
            )

    async def async_recalculate_historic_costs(
        self,
        start_date: date,
        end_date: date,
        new_options: dict[str, Any],
    ) -> None:
        """
        Recalculate cost statistics for a historic date range using a new cost mode.

        This is the public entry point.  It acquires ``recalculation_lock`` for the
        entire duration so the options flow can detect a running recalculation and
        prevent the user from starting a second one.

        The actual work is delegated to ``_async_recalculate_historic_costs_locked``.
        """
        async with self.recalculation_lock:
            await self._async_recalculate_historic_costs_locked(
                start_date, end_date, new_options
            )

    async def _async_recalculate_historic_costs_locked(
        self,
        start_date: date,
        end_date: date,
        new_options: dict[str, Any],
    ) -> None:
        """
        Recalculate cost statistics from stored consumption data.

        For tiered rates the cumulative Wh counter resets at each estimated
        billing cycle boundary so tier thresholds are applied correctly.
        The running cost ``sum`` is seeded from the last record before the
        window to keep the statistic series continuous.
        """
        new_cost_mode, new_fixed_rate, rate_schedule = _resolve_cost_config(new_options)
        if new_cost_mode == COST_MODE_NONE:
            _LOGGER.info("New cost mode is NONE; skipping recalculation.")
            return

        is_tiered = new_cost_mode in TIERED_RATE_REGISTRY

        _LOGGER.info(
            "Starting historic cost recalculation from %s to %s (mode: %s)",
            start_date,
            end_date,
            new_cost_mode,
        )

        tz = await dt_util.async_get_time_zone(self.api.get_timezone())

        def _to_dt(d: date) -> datetime:
            return datetime.combine(d, datetime.min.time()).replace(tzinfo=tz)

        window_start = _to_dt(start_date)
        window_end = _to_dt(end_date + timedelta(days=1))

        # ── 1. Resolve statistic IDs ──────────────────────────────────────────────
        accounts, service_addr_account_no = await self.api.async_get_accounts()
        if "ELECTRIC" not in accounts:
            _LOGGER.info("No ELECTRIC account found; nothing to recalculate.")
            return

        consumption_id, cost_id, name_prefix = _build_statistic_ids(
            service_addr_account_no, "ELECTRIC"
        )
        assert cost_id is not None

        # ── 2. Estimate billing cycles (tiered rates only) ────────────────────────
        billing_cycles: list[tuple[date, date]] = []
        if is_tiered:
            forecast = await self.api.async_get_forecast()
            billing_cycles = _estimate_billing_cycles(
                forecast.start_date,
                forecast.end_date,
                start_date,
            )

        # ── 3. Fetch consumption rows ─────────────────────────────────────────────
        # For tiered rates, start from the earliest estimated cycle so the
        # cumulative Wh counter is correct even when the window is mid-cycle.
        fetch_start = window_start
        if billing_cycles:
            fetch_start = min(fetch_start, _to_dt(billing_cycles[0][0]))

        consumption_rows: list[dict] = (
            await get_instance(self.hass).async_add_executor_job(
                statistics_during_period,
                self.hass,
                fetch_start,
                window_end,
                {consumption_id},
                "hour",
                None,
                {"state"},
            )
        ).get(consumption_id, [])

        if not consumption_rows:
            _LOGGER.warning("No consumption data in %s-%s.", start_date, end_date)
            return

        # ── 4. Seed running cost sum from last record before the window ───────────
        pre_cost_sum = 0.0
        last_cost = await get_instance(self.hass).async_add_executor_job(
            get_last_statistics,
            self.hass,
            1,
            cost_id,
            True,
            {"sum"},
        )
        if cost_id in last_cost:
            last_ts = last_cost[cost_id][0].get("start", 0)
            if isinstance(last_ts, (int, float)):
                last_ts_dt = datetime.fromtimestamp(last_ts, tz=dt_util.UTC)
            else:
                last_ts_dt = last_ts
            if last_ts_dt < window_start:
                pre_cost_sum = float(last_cost[cost_id][0].get("sum") or 0.0)

        # ── 5. Price each row, resetting cumulative Wh at cycle boundaries ────────
        cost_statistics: list[StatisticData] = []
        running_sum = pre_cost_sum
        current_cycle: tuple[date, date] | None = None
        cumulative_wh = 0.0

        for row in consumption_rows:
            hour_dt = datetime.fromtimestamp(row["start"], tz=dt_util.UTC)
            interval_wh = float(row.get("state") or 0.0)
            row_date = hour_dt.astimezone(tz).date() if tz else hour_dt.date()

            # Reset cumulative Wh at billing cycle boundaries
            if is_tiered and billing_cycles:
                row_cycle = _find_billing_cycle_for_date(row_date, billing_cycles)
                if row_cycle != current_cycle:
                    current_cycle = row_cycle
                    cumulative_wh = 0.0

            cost = _calculate_cost_for_wh(
                interval_wh,
                hour_dt,
                cumulative_wh,
                new_cost_mode,
                new_fixed_rate,
                rate_schedule,
            )
            cumulative_wh += interval_wh

            # Only emit rows within the requested window (earlier rows are
            # only fetched to seed the cumulative Wh counter for mid-cycle starts)
            if cost > 0 and row_date >= start_date:
                running_sum += cost
                cost_statistics.append(
                    StatisticData(start=hour_dt, state=cost, sum=running_sum)
                )

        if not cost_statistics:
            _LOGGER.warning(
                "Recalculation produced no cost data for %s-%s.", start_date, end_date
            )
            return

        # ── 6. Upsert into recorder ──────────────────────────────────────────────
        self._push_cost_statistics(
            cost_id,
            name_prefix.substitute(stat_type="cost"),
            cost_statistics,
            "recalculation",
            running_sum,
        )
        _LOGGER.info("Historic cost recalculation complete.")
