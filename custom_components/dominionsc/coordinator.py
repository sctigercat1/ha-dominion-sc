"""Coordinator to handle dominionsc connections."""

import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from string import Template

from homeassistant.components.recorder import get_instance
from homeassistant.components.recorder.models import (
    StatisticData,
    StatisticMeanType,
    StatisticMetaData,
)
from homeassistant.components.recorder.statistics import (
    async_add_external_statistics,
    get_last_statistics,
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
    COST_MODE_NONE,
    COST_MODE_RATE_6,
    COST_MODE_RATE_8,
    DEFAULT_FIXED_RATE,
    DOMAIN,
)
from .rates import SC_RATE_6, SC_RATE_8, calculate_sc_rate_interval_cost

_LOGGER = logging.getLogger(__name__)

type DominionSCConfigEntry = ConfigEntry[DominionSCCoordinator]


@dataclass
class DominionSCStatisticMetadata:
    """Metadata for creating statistics."""

    account: str
    consumption_id: str
    cost_id: str | None
    name_prefix: str
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

        accounts_full = await self.api.async_get_accounts()
        accounts = accounts_full[0]
        service_addr_account_no = accounts_full[1]

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
        Calculate cost for a single interval based on configured mode.

        Args:
            usage_read: The usage read interval data.
            cumulative_wh_before: Cumulative Wh before this interval in the
                billing period. Used for tiered pricing.

        """
        options = self.config_entry.options
        cost_mode = options.get(CONF_COST_MODE, COST_MODE_RATE_8)

        if cost_mode == COST_MODE_NONE:
            # No cost calculation
            return 0.0

        if cost_mode == COST_MODE_RATE_8:
            return calculate_sc_rate_interval_cost(
                usage_read.consumption,
                usage_read.start_time,
                cumulative_wh_before,
                SC_RATE_8,
            )

        if cost_mode == COST_MODE_RATE_6:
            return calculate_sc_rate_interval_cost(
                usage_read.consumption,
                usage_read.start_time,
                cumulative_wh_before,
                SC_RATE_6,
            )

        # Fixed rate
        fixed_rate = options.get(CONF_FIXED_RATE, DEFAULT_FIXED_RATE)  # $/Wh
        return usage_read.consumption * fixed_rate

    async def _insert_statistics(
        self,
        accounts: list[str],
        service_addr_account_no: str,
        forecast: Forecast | None,
    ) -> dict[str, datetime]:
        """Insert DominionSC statistics."""
        last_changed_per_account: dict[str, datetime] = {}
        for account in accounts:
            id_prefix = (f"DominionSC_{account}").lower().replace("-", "_")
            consumption_statistic_id = f"{DOMAIN}:{id_prefix}_energy_consumption"

            cost_statistic_id = None
            cost_mode = self.config_entry.options.get(CONF_COST_MODE, COST_MODE_RATE_8)
            if account == "ELECTRIC" and cost_mode != COST_MODE_NONE:
                cost_statistic_id = f"{DOMAIN}:{id_prefix}_energy_cost"

            _LOGGER.debug("Updating Statistics for %s", consumption_statistic_id)

            name_prefix = Template(
                f"{account.title()} $stat_type {service_addr_account_no}"
            )

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

                # Create metadata bundle
                dominionsc_metadata = DominionSCStatisticMetadata(
                    account=account,
                    consumption_id=consumption_statistic_id,
                    cost_id=cost_statistic_id,
                    name_prefix=name_prefix,
                    unit_class=consumption_unit_class,
                    unit=consumption_unit,
                )

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

                # Create metadata bundle
                dominionsc_metadata = DominionSCStatisticMetadata(
                    account=account,
                    consumption_id=consumption_statistic_id,
                    cost_id=cost_statistic_id,
                    name_prefix=name_prefix,
                    unit_class=consumption_unit_class,
                    unit=consumption_unit,
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
        if metadata.cost_id and last_cost_stat.get(metadata.cost_id):
            try:
                cost_sum = float(last_cost_stat[metadata.cost_id][0].get("sum") or 0)
            except (KeyError, IndexError, TypeError, ValueError):
                _LOGGER.debug("Could not get last cost sum, starting from 0")

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
        is_tiered_rate = self.config_entry.options.get(
            CONF_COST_MODE, COST_MODE_RATE_8
        ) in (COST_MODE_RATE_8, COST_MODE_RATE_6)
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
                if (not is_tiered_rate) or (
                    is_tiered_rate
                    and billing_period_start <= interval_date
                    and billing_period_end >= interval_date
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
            cost_metadata = StatisticMetaData(
                mean_type=StatisticMeanType.NONE,
                has_sum=True,
                name=metadata.name_prefix.substitute(stat_type="cost"),
                source=DOMAIN,
                statistic_id=metadata.cost_id,
                unit_class=None,
                unit_of_measurement=None,
            )

            _LOGGER.info(
                "Adding %d hourly cost statistics for %s (%s, sum=%.3f)",
                len(cost_statistics),
                metadata.cost_id,
                operation_type,
                cost_sum,
            )
            async_add_external_statistics(self.hass, cost_metadata, cost_statistics)
