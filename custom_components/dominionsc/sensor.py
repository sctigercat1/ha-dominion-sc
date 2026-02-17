"""Support for dominionsc sensors."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime
import re

from homeassistant.components.sensor import (
    SensorDeviceClass,
    SensorEntity,
    SensorEntityDescription,
    SensorStateClass,
)
from homeassistant.const import EntityCategory
from homeassistant.core import HomeAssistant
from homeassistant.helpers.device_registry import DeviceEntryType, DeviceInfo
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers.typing import StateType
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .const import DOMAIN
from .coordinator import (
    DominionSCAccountData,
    DominionSCConfigEntry,
    DominionSCCoordinator,
)

# Coordinator is used to centralize the data updates
PARALLEL_UPDATES = 0


@dataclass(frozen=True, kw_only=True)
class DominionSCEntityDescription(SensorEntityDescription):
    """Class describing dominionsc sensors entities."""

    value_fn: Callable[[DominionSCAccountData | DominionSCData], str | float | date | datetime | None]


ACCOUNT_SENSORS: tuple[DominionSCEntityDescription, ...] = (
    DominionSCEntityDescription(
        key="last_changed",
        translation_key="last_changed",
        device_class=SensorDeviceClass.TIMESTAMP,
        entity_category=EntityCategory.DIAGNOSTIC,
        value_fn=lambda data: data.last_changed,
    ),
)

BILLING_SENSORS: tuple[DominionSCEntityDescription, ...] = (
    DominionSCEntityDescription(
        key="cost_to_date",
        translation_key="cost_to_date",
        device_class=SensorDeviceClass.MONETARY,
        entity_category=EntityCategory.DIAGNOSTIC,
        native_unit_of_measurement="USD",
        state_class=SensorStateClass.TOTAL,
        suggested_display_precision=2,
        value_fn=lambda data: data.forecast.cost_to_date if data.forecast else None,
    ),
    DominionSCEntityDescription(
        key="forecasted_cost",
        translation_key="forecasted_cost",
        device_class=SensorDeviceClass.MONETARY,
        entity_category=EntityCategory.DIAGNOSTIC,
        native_unit_of_measurement="USD",
        state_class=SensorStateClass.TOTAL,
        suggested_display_precision=2,
        value_fn=lambda data: data.forecast.forecasted_cost if data.forecast else None,
    ),
    DominionSCEntityDescription(
        key="typical_cost",
        translation_key="typical_cost",
        device_class=SensorDeviceClass.MONETARY,
        entity_category=EntityCategory.DIAGNOSTIC,
        native_unit_of_measurement="USD",
        state_class=SensorStateClass.TOTAL,
        suggested_display_precision=2,
        value_fn=lambda data: data.forecast.typical_cost if data.forecast else None,
    ),
    DominionSCEntityDescription(
        key="start_date",
        translation_key="start_date",
        device_class=SensorDeviceClass.DATE,
        entity_category=EntityCategory.DIAGNOSTIC,
        entity_registry_enabled_default=False,
        value_fn=lambda data: data.forecast.start_date if data.forecast else None,
    ),
    DominionSCEntityDescription(
        key="end_date",
        translation_key="end_date",
        device_class=SensorDeviceClass.DATE,
        entity_category=EntityCategory.DIAGNOSTIC,
        entity_registry_enabled_default=False,
        value_fn=lambda data: data.forecast.end_date if data.forecast else None,
    ),
    DominionSCEntityDescription(
        key="last_updated",
        translation_key="last_updated",
        device_class=SensorDeviceClass.TIMESTAMP,
        entity_category=EntityCategory.DIAGNOSTIC,
        value_fn=lambda data: data.last_updated,
    ),
)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: DominionSCConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up the dominionsc sensor."""
    coordinator = entry.runtime_data
    entities: list[DominionSCSensor] = []

    dominionsc_data = coordinator.data
    accounts_data = dominionsc_data.accounts
    forecast = dominionsc_data.forecast
    service_addr_account_no = dominionsc_data.service_addr_account_no
    clean_service_addr = re.sub(r'[\W]+|^(?=\d)', '_', service_addr_account_no).strip('_').lower()
    
    # Device per service address
    device_id = f"{DOMAIN}_{clean_service_addr}"
    device = DeviceInfo(
        identifiers={(DOMAIN, device_id)},
        name=f"{service_addr_account_no}",
        manufacturer="Dominion Energy SC",
        entry_type=DeviceEntryType.SERVICE,
    )
    
    # Account (electric/gas) sensors
    for account in accounts_data:
        entities.extend(
            DominionSCSensor(
                coordinator,
                sensor,
                account,
                device,
                device_id,
            )
            for sensor in ACCOUNT_SENSORS
        )

    # Common billing sensors
    if forecast is not None:
        entities.extend(
            DominionSCSensor(
                coordinator,
                sensor,
                "billing",
                device,
                device_id,
            )
            for sensor in BILLING_SENSORS
        )

    async_add_entities(entities)


class DominionSCSensor(CoordinatorEntity[DominionSCCoordinator], SensorEntity):
    """Representation of an dominionsc sensor."""

    _attr_has_entity_name = True
    entity_description: DominionSCEntityDescription

    def __init__(
        self,
        coordinator: DominionSCCoordinator,
        description: DominionSCEntityDescription,
        account: str,
        device: DeviceInfo,
        device_id: str,
    ) -> None:
        """Initialize the sensor."""
        super().__init__(coordinator)
        self.entity_description = description
        self._attr_unique_id = f"{device_id}_{account}_{description.key}"
        self._attr_device_info = device
        self._attr_translation_placeholders = {"account": account.title()}
        self.account = account

    @property
    def native_value(self) -> StateType | date | datetime:
        """Return the state."""
        coordinator_data = self.coordinator.data

        # Common billing sensors
        if self.account == "billing":
            return self.entity_description.value_fn(coordinator_data)

        # Account specific sensors
        return self.entity_description.value_fn(coordinator_data.accounts[self.account])
