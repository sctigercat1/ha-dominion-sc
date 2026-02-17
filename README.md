# Home Assistant Integration for Dominion Energy SC

[![hacs_badge](https://img.shields.io/badge/HACS-Custom-41BDF5.svg)](https://github.com/hacs/integration)
[![GitHub Release](https://img.shields.io/github/release/sctigercat1/ha-dominion-sc.svg)](https://github.com/sctigercat1/ha-dominion-sc/releases)
[![License](https://img.shields.io/github/license/sctigercat1/ha-dominion-sc.svg)](LICENSE)

A Home Assistant custom integration for Dominion Energy South Carolina customers to monitor their energy usage and billing information.

## Features

- **Hourly Interval Energy Usage Data**: Track your energy consumption with hourly granularity
- **Billing Information**: Monitor current billing cycle costs and forecasts
- **Energy Dashboard Compatibility**: Seamlessly integrate with Home Assistant's Energy Dashboard
- **Automatic Updates**: Data refreshes every 12 hours
- **Cost Estimation Options**: 
  - Fixed rate pricing
  - South Carolina Rate Schedule 8 (General Service)
  - South Carolina Rate Schedule 6 (Energy Saver/Conservation)
- **Multiple Energy Sources**: Support for both electric and gas services at the same service address

## Installation

### HACS (Recommended)

1. Open HACS in your Home Assistant instance
2. Click on the three dots in the top right corner
3. Select "Custom repositories"
4. Add `https://github.com/sctigercat1/ha-dominion-sc` as an integration repository
5. Click "Explore & Download Repositories"
6. Search for "Dominion Energy SC"
7. Click "Download"
8. Restart Home Assistant

### Manual Installation

1. Download the latest release from the [releases page](https://github.com/sctigercat1/ha-dominion-sc/releases)
2. Extract the `custom_components/dominionsc` folder to your Home Assistant `custom_components` directory
3. Restart Home Assistant

## Configuration

### Initial Setup & Installation Parameters

1. Go to **Settings** → **Devices & Services**
2. Click **Add Integration**
3. Search for "Dominion Energy SC"
4. Enter your Dominion Energy SC account credentials:
   - **Username**: Your Dominion Energy SC online account username
   - **Password**: Your Dominion Energy SC online account password
5. Complete two-factor authentication when prompted
   - **Recommended**: Use SMS-based verification for better reliability
6. Select your cost tracking preferences:
   - **None**: No cost calculation
   - **Fixed Rate**: Enter a custom rate (default: $0.14164/kWh)
   - **Rate Schedule 8**: General Service rate
   - **Rate Schedule 6**: Energy Saver/Conservation rate
7. Click **Submit**

### Configuration Parameters

| Parameter | Description | Required | Default |
|-----------|-------------|----------|---------|
| Username | Dominion Energy SC account username | Yes | - |
| Password | Dominion Energy SC account password | Yes | - |
| Cost Mode | Cost calculation method | No | None |
| Fixed Rate | Custom rate in $/kWh | No | 0.14164 |

## Removal

This integration follows standard integration removal. No extra steps are required.

## Sensors

The integration creates the following sensors:

### Energy Source Sensors

- **Last Published**: Timestamp of when energy data was last published to the Dominion Energy SC website
- **Last Checked**: Timestamp of when the Dominion Energy SC website was last polled for data changes

### Billing Sensors

- **Cost to Date**: Current billing cycle cost in USD
- **Forecasted Cost**: Projected end-of-cycle cost in USD
- **Typical Cost**: Historical average cost for comparison in USD
- **Start Date**: Billing cycle start date
- **End Date**: Billing cycle end date

All monetary sensors display in USD with 2 decimal precision.

## Example Usage

### Viewing Your Data

Energy data can be viewed in two locations:
- **Energy Dashboard**: For electric consumption, electric cost, and gas consumption
- **Sensors**: As described in the Sensors section above

### Energy Dashboard Integration

The integration provides three statistics for the Energy Dashboard:

| Statistic | Description | Unit |
|-----------|-------------|------|
| `dominionsc:electric_consumption` | Hourly electric energy consumption | Wh |
| `dominionsc:electric_cost` | Hourly electric energy cost | USD |
| `dominionsc:gas_consumption` | Hourly gas consumption | ft³ |

**To add statistics to the Energy Dashboard:**

1. Go to **Settings** → **Dashboards** → **Energy**
2. Under **Electricity grid**, click **Add consumption**
3. Search for your service address or "electric" / "gas"
4. Select the appropriate consumption statistic as described in the table above
5. For cost tracking (optional), select **Use an entity tracking the total costs**
6. Select the appropriate cost statistic
7. For gas consumption (if applicable), navigate to **Gas consumption** and repeat the process with the gas consumption statistic

### Creating Automations

Example automation to notify when forecasted costs exceed a threshold:

```yaml
automation:
  - alias: "High Energy Cost Alert"
    trigger:
      - platform: numeric_state
        entity_id: sensor.dominionsc_billing_forecasted_cost
        above: 150
    action:
      - service: notify.mobile_app
        data:
          message: "Your forecasted energy cost is ${{ states('sensor.dominionsc_billing_forecasted_cost') }}"
```

## Troubleshooting

### Authentication Issues

If you receive authentication errors:
1. Verify your credentials at [dominionenergy.com](https://www.dominionenergy.com)
2. If you recently changed your password, remove and re-add the integration
3. Ensure two-factor authentication is properly configured
   - **Recommended**: Use SMS-based verification for better reliability
4. Check the Home Assistant logs for specific error messages

### Data Not Updating

- The integration polls every 12 hours by default
- Manual refresh: Go to **Settings** → **Devices & Services** → **Dominion Energy SC** → Click the three dots → **Reload**
- Check your internet connection
- Verify Dominion Energy SC's website is accessible

## Known Limitations

- **Data Delay**: Energy usage data is reported by Dominion Energy SC with a 24-48 hour delay. Real-time monitoring is not available. The API is polled every 12 hours for new energy data
- **Single Service Address**: Currently supports only one service address per Dominion account. Multiple service address support may be in a future release
- **No Time-of-Use Data**: Time-of-use pricing breakdowns or any other rate not discussed above are not available
- **No Solar/Grid Export**: Energy provided back to the grid from sources like solar panels is not yet supported
- **Historical Data**: Backfilling of data is limited to current billing cycle to allow for accurate cost calculation
- **Cost Estimates**: Calculations are estimates based on selected rate schedule; actual bills may vary
- **TFA Required**: The account must have TFA activated; flows are not supported for accounts without TFA

## Support

- [Open an issue](https://github.com/sctigercat1/ha-dominion-sc/issues)
- [dominion-sc-power library](https://github.com/sctigercat1/dominion-sc-power)

## Contributing

Contributions are welcome! Please submit a pull request with your proposed changes.

### Development Environment Setup
```bash
git clone https://github.com/sctigercat1/ha-dominion-sc.git
cd ha-dominion-sc
./scripts/setup
```

### Code Validation

After each change, please run the following scripts to format/check your code with `ruff` and run unit tests.

```bash
./scripts/lint
./scripts/test
```

## Credits

This project was inspired by [Opower](https://www.home-assistant.io/integrations/opower/), [ha-dominion-energy](https://github.com/YeomansIII/ha-dominion-energy), [ha-unraid](https://github.com/ruaan-deysel/ha-unraid), and [integration_blueprint](https://github.com/ludeeus/integration_blueprint). Much appreciated!

Special thanks also to the Home Assistant and HACS communities.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Disclaimer

This is an unofficial integration and is not affiliated with, endorsed by, or connected to Dominion Energy SC. Use at your own risk. The authors are not responsible for any issues that may arise from using this integration.

## Privacy

This integration communicates directly with Dominion Energy SC's servers using your credentials. No data is sent to third parties. Your credentials are stored securely in Home Assistant's configuration and are only used to authenticate with Dominion Energy SC.
