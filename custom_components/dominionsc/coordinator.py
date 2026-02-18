"""Coordinator to handle dominionsc connections."""

import asyncio
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
    UsageRead,
    create_cookie_jar,
)
from dominionsc.exceptions import ApiException, CannotConnect, InvalidAuth, MfaChallenge

from .const import (
    CONF_COST_MODE,
    CONF_FIXED_RATE,
    CONF_LOGIN_DATA,
    COST_MODE_FIXED,
    COST_MODE_NONE,
    COST_MODE_RATE_8,
    DEFAULT_FIXED_RATE,
    DOMAIN,
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

    This is the single canonical implementation of the ID-building logic used
    by both ``_insert_statistics`` and ``_async_recalculate_historic_costs_locked``.

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


def _extract_last_sum(last_stat: dict, statistic_id: str) -> float:
    """
    Safely extract the cumulative ``sum`` from a ``get_last_statistics`` result.

    Returns 0.0 if the statistic does not exist or the value cannot be parsed.
    """
    try:
        return float(last_stat[statistic_id][0].get("sum") or 0)
    except (KeyError, IndexError, TypeError, ValueError):
        return 0.0


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

    This is the single canonical pricing function used by both the live
    ingestion path (via ``_calculate_interval_cost``) and the historic
    recalculation path, so both are guaranteed to produce identical values
    for the same inputs.

    Args:
        interval_wh:          Wh consumed in this interval.
        interval_dt:          Timestamp of the interval (used for season).
        cumulative_wh_before: Total Wh consumed before this interval in the
                              billing period (used for tier boundary tracking).
        cost_mode:            One of the COST_MODE_* constants.
        fixed_rate:           $/Wh fixed rate (only used when cost_mode is FIXED).
        rate_schedule:        RateSchedule instance (only used for tiered modes).
                              Pass ``None`` to produce 0.0 cost (e.g. outside the
                              billing period for tiered rates).

    Returns:
        Cost in dollars for this interval.

    """
    if cost_mode == COST_MODE_NONE:
        return 0.0
    if cost_mode == COST_MODE_FIXED:
        return interval_wh * fixed_rate
    if rate_schedule is not None:
        return calculate_sc_rate_interval_cost(
            interval_wh,
            interval_dt,
            cumulative_wh_before,
            rate_schedule,
        )
    return 0.0


def _resolve_cost_config(
    options: dict[str, Any],
) -> tuple[str, float, RateSchedule | None]:
    """Return (cost_mode, fixed_rate, rate_schedule) with consistent defaults."""
    cost_mode = options.get(CONF_COST_MODE, COST_MODE_RATE_8)
    fixed_rate: float = options.get(CONF_FIXED_RATE, DEFAULT_FIXED_RATE)
    rate_schedule: RateSchedule | None = TIERED_RATE_REGISTRY.get(cost_mode)
    return cost_mode, fixed_rate, rate_schedule


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

    def _calculate_interval_cost(
        self,
        usage_read: UsageRead,
        cumulative_wh_before: float = 0.0,
    ) -> float:
        """
        Calculate cost for a single interval based on the currently configured mode.

        Delegates to the module-level ``_calculate_cost_for_wh`` so that live
        ingestion and historic recalculation share identical pricing logic.

        Args:
            usage_read: The usage read interval data.
            cumulative_wh_before: Cumulative Wh before this interval in the
                billing period. Used for tiered pricing.

        """
        cost_mode, fixed_rate, rate_schedule = _resolve_cost_config(
            self.config_entry.options
        )
        return _calculate_cost_for_wh(
            usage_read.consumption,
            usage_read.start_time,
            cumulative_wh_before,
            cost_mode,
            fixed_rate,
            rate_schedule,
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
        """
        Backfill historical statistics for initial setup.

        Fetches the last BACKFILL_DAYS of data and creates hourly statistics.
        """
        today = date.today()
        start_date = forecast.start_date  # today - timedelta(days=BACKFILL_DAYS)
        data_date = today - timedelta(days=1)  # Yesterday

        _LOGGER.debug("Backfilling statistics from %s to %s", start_date, data_date)

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
            cost_sum = _extract_last_sum(last_cost_stat, metadata.cost_id)

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
        cost_mode_here, _, _ = _resolve_cost_config(self.config_entry.options)
        is_tiered_rate = cost_mode_here in TIERED_RATE_REGISTRY
        cumulative_wh = 0.0

        # Track billing cycle for tiered rates
        # Forecasted data if possible, otherwise monthly fallback
        billing_period_start = forecast.start_date
        billing_period_end = forecast.end_date
        _LOGGER.debug(
            "Using billing cycle for tier resets: %s to %s",
            billing_period_start,
            billing_period_end,
        )

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

            # Only add cost data if within the existing billing cycle
            # and is_tiered_rate, also if fixed rate
            if is_electric and metadata.cost_id:
                if not is_tiered_rate or (
                    billing_period_start <= interval_date <= billing_period_end
                ):
                    if hour_start not in hourly_cost:
                        hourly_cost[hour_start] = 0.0

                    hourly_cost[hour_start] += self._calculate_interval_cost(
                        usage_read,
                        cumulative_wh_before=cumulative_wh,
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
                cost_sum += cost
                cost_statistics.append(
                    StatisticData(start=hour_start, state=cost, sum=cost_sum)
                )

        if not consumption_statistics:
            return

        # Update last_changed with the latest interval time
        last_changed_per_account[metadata.account] = usage_reads[-1].start_time

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
        Recalculate cost statistics from stored consumption data (no API call).

        Prices each hourly consumption ``state`` value using ``_calculate_cost_for_wh``.
        For tiered rates, ``cumulative_wh`` accumulates from epoch without resetting
        since historic billing period boundaries are unavailable.  The running cost
        ``sum`` is seeded from the last record before the window to keep the series
        continuous.
        """
        new_cost_mode, new_fixed_rate, rate_schedule = _resolve_cost_config(new_options)
        if new_cost_mode == COST_MODE_NONE:
            _LOGGER.info("New cost mode is NONE; skipping recalculation.")
            return

        _LOGGER.info(
            "Starting historic cost recalculation from %s to %s (new mode: %s)",
            start_date,
            end_date,
            new_cost_mode,
        )

        tz = await dt_util.async_get_time_zone(self.api.get_timezone())

        # Window bounds: [window_start, window_end) as tz-aware datetimes.
        window_start = datetime.combine(start_date, datetime.min.time()).replace(
            tzinfo=tz
        )
        window_end = datetime.combine(
            end_date + timedelta(days=1), datetime.min.time()
        ).replace(tzinfo=tz)

        # ── 1. Reconstruct statistic IDs via the shared helper ────────────────────
        # async_get_accounts() reads self.accounts in-memory — no network call.
        accounts, service_addr_account_no = await self.api.async_get_accounts()

        if "ELECTRIC" not in accounts:
            _LOGGER.info("No ELECTRIC account found; nothing to recalculate.")
            return

        consumption_statistic_id, cost_statistic_id, name_prefix = _build_statistic_ids(
            service_addr_account_no, "ELECTRIC"
        )
        # cost_statistic_id is always non-None for ELECTRIC (see _build_statistic_ids),
        # but assert here to satisfy the type checker.
        assert cost_statistic_id is not None

        # ── 2. Fetch consumption statistics for the recalculation window ──────────
        window_consumption: dict = await get_instance(self.hass).async_add_executor_job(
            statistics_during_period,
            self.hass,
            window_start,
            window_end,
            {consumption_statistic_id},
            "hour",
            None,
            {"state"},
        )

        consumption_rows: list[dict] = window_consumption.get(
            consumption_statistic_id, []
        )
        if not consumption_rows:
            _LOGGER.warning(
                "No consumption statistics found for %s in range %s - %s; "
                "nothing to recalculate.",
                consumption_statistic_id,
                start_date,
                end_date,
            )
            return

        # ── 3. Seed the tier counter from consumption before the window ───────────
        # Sum all consumption state values before window_start so the tier boundary
        # is crossed at the correct point for mid-cycle windows.  Returns 0.0 if
        # no rows exist before the window (first-ever stat or full-history recalc).
        cumulative_wh_before_window = 0.0
        pre_window_consumption: dict = await get_instance(
            self.hass
        ).async_add_executor_job(
            statistics_during_period,
            self.hass,
            datetime.fromtimestamp(0, tz=dt_util.UTC),  # epoch — all history
            window_start,  # exclusive upper bound
            {consumption_statistic_id},
            "hour",
            None,
            {"state"},
        )
        for row in pre_window_consumption.get(consumption_statistic_id, []):
            val = row.get("state")
            if val is not None:
                cumulative_wh_before_window += float(val)

        _LOGGER.debug(
            "Cumulative Wh before window start %s: %.1f Wh",
            start_date,
            cumulative_wh_before_window,
        )

        # ── 5. Seed the running cost sum from the last record before the window ───
        # Use a bounded query rather than get_last_statistics, which returns the
        # last row of the whole series and would cause a spike at the window start.
        pre_window_cost_sum = 0.0
        pre_window_cost_rows: dict = await get_instance(
            self.hass
        ).async_add_executor_job(
            statistics_during_period,
            self.hass,
            datetime.fromtimestamp(0, tz=dt_util.UTC),  # epoch — all history
            window_start,  # exclusive upper bound
            {cost_statistic_id},
            "hour",
            None,
            {"sum"},
        )
        rows_before = pre_window_cost_rows.get(cost_statistic_id, [])
        if rows_before:
            pre_window_cost_sum = float(rows_before[-1].get("sum") or 0.0)

        # ── 6. Price each hourly consumption row under the new rate ───────────────
        # cumulative_wh is never reset: historic period start dates are unavailable,
        # so a continuous counter is the most accurate approximation possible.
        cost_statistics: list[StatisticData] = []
        running_cost_sum = pre_window_cost_sum
        cumulative_wh = cumulative_wh_before_window

        for row in consumption_rows:
            # `start` is a float Unix timestamp as returned by statistics_during_period
            start_ts: float = row["start"]
            hour_start_dt = datetime.fromtimestamp(start_ts, tz=dt_util.UTC)
            interval_wh: float = float(row.get("state") or 0.0)

            new_cost = _calculate_cost_for_wh(
                interval_wh,
                hour_start_dt,
                cumulative_wh,
                new_cost_mode,
                new_fixed_rate,
                rate_schedule,
            )

            running_cost_sum += new_cost
            cumulative_wh += interval_wh

            cost_statistics.append(
                StatisticData(
                    start=hour_start_dt,
                    state=new_cost,
                    sum=running_cost_sum,
                )
            )

        if not cost_statistics:
            _LOGGER.warning(
                "Recalculation produced no cost data for range %s - %s.",
                start_date,
                end_date,
            )
            return

        # ── 7. Upsert the corrected statistics into the recorder ──────────────────
        self._push_cost_statistics(
            cost_statistic_id,
            name_prefix.substitute(stat_type="cost"),
            cost_statistics,
            "recalculation",
            running_cost_sum,
        )
        _LOGGER.info("Historic cost recalculation complete.")
