"""The dominionsc integration."""

from __future__ import annotations

from homeassistant.const import Platform
from homeassistant.core import HomeAssistant

from .const import DOMAIN
from .coordinator import DominionSCConfigEntry, DominionSCCoordinator

PLATFORMS: list[Platform] = [Platform.SENSOR]


async def async_setup_entry(hass: HomeAssistant, entry: DominionSCConfigEntry) -> bool:
    """Set up DominionSC from a config entry."""
    coordinator = DominionSCCoordinator(hass, entry)
    await coordinator.async_config_entry_first_refresh()
    entry.runtime_data = coordinator

    # Store the coordinator so we can access it later
    hass.data.setdefault(DOMAIN, {})[entry.entry_id] = coordinator
    # Registers update_listener to be called when options are changed
    entry.async_on_unload(entry.add_update_listener(update_listener))

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

    return True


async def async_unload_entry(hass: HomeAssistant, entry: DominionSCConfigEntry) -> bool:
    """Unload a config entry."""
    return await hass.config_entries.async_unload_platforms(entry, PLATFORMS)


async def update_listener(hass: HomeAssistant, entry: DominionSCConfigEntry) -> None:
    """Handle options update."""
    # Retrieve the coordinator from hass.data
    coordinator = hass.data[DOMAIN][entry.entry_id]

    # 1. Update internal variables if necessary
    # Example: coordinator.api_client.timeout = entry.options.get("timeout")

    # 2. Trigger the update
    # Note: Use async_request_refresh(), NOT _async_update_data()
    await coordinator.async_request_refresh()
