# ⚡ Silver Streak Analyzer (SSA)

**Version 4.8.0** - Multi-Spectrum RF Survey & GNSS Variance Analysis

A comprehensive RF signal analysis platform for wardriving, flight testing, and GNSS receiver comparison.

## Features

### 📡 RF Signal Analysis (Multi-Spectrum)
- **WiFi** (2.4/5/6 GHz) - Workers A-D
- **Bluetooth LE** - Worker M
- **Thread/Matter/Zigbee** - Worker E (ESP32-H2)
- **WiFi HaLow** (900 MHz) - Worker F
- DBSCAN clustering for static/mobile classification
- Real-time MQTT ingestion from field hardware
- Offline CSV analysis

### 📡 GNSS Link Module
- Multi-receiver GNSS variance analysis
- **Live GNSS MQTT** - Real-time data from Raspberry Pi recorder (NEW in v4.8.0)
- CSV import (Option B format with ReceiverID column)
- Truth receiver designation for comparison
- Variance metrics:
  - Standard deviation (horizontal/vertical)
  - CEP (50%) - Circular Error Probable
  - 2DRMS (95%) - 95% accuracy circle
  - Max deviation tracking
- Track visualization with coordinate overlay
- PDF report generation

### 📊 Visualization
- Dark mode Leaflet mapping with multiple basemap options
- Real-time data streaming via WebSocket
- Interactive table view with filtering
- PDF and HTML report generation

## Installation

### Quick Start (Ubuntu 24 LTS)

```bash
# Extract and enter directory
tar -xzf silver_streak_analyzer_v4.8.0.tar.gz
cd silver_streak_analyzer

# Run installer
chmod +x install.sh
./install.sh

# Start the application
./run.sh
```

### Manual Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r backend/requirements.txt

# Run
cd backend
python app.py
```

## Usage

### RF Analysis (Wardriving Mode)

1. Open browser to `http://localhost:5000`
2. Upload CSV files from wardriving workers
3. View on map or table
4. Generate reports

### GNSS Link Mode

1. Click "📡 GNSS Link" tab
2. Upload GNSS CSV files (Option B format recommended)
3. Mark truth receivers for comparison
4. View variance metrics
5. Generate PDF report

### Live GNSS MQTT (NEW in v4.8.0)

Connect to a Raspberry Pi running the GNSS recorder for live position data:

**Option A: Environment variables (at startup)**
```bash
export GNSS_MQTT_HOST=10.42.0.223
export GNSS_MQTT_PORT=1883
python app.py
```

**Option B: Connect via API (after startup)**
```bash
curl -X POST http://localhost:5000/api/gnss-mqtt/connect \
  -H "Content-Type: application/json" \
  -d '{"host": "10.42.0.223", "port": 1883}'
```

**Option C: From browser console**
```javascript
fetch('/api/gnss-mqtt/connect', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({host: '10.42.0.223', port: 1883})
}).then(r => r.json()).then(console.log);
```

## CSV Formats

### RF Survey CSV (Workers)
```csv
Timestamp,MAC,SSID,RSSI,Channel,Security,Lat,Lon,Worker
2025-01-19T14:30:00,AA:BB:CC:DD:EE:FF,MyNetwork,-65,6,WPA2,40.867,-124.083,A
```

### GNSS Link CSV (Option B)
```csv
Timestamp,ReceiverID,Latitude,Longitude,AltitudeMSL,FixType,SatCount,HDOP,VDOP,PDOP
2025-01-19T14:30:00,rcvr_01,40.867542,-124.083621,12.1,2,11,0.82,1.18,1.42
2025-01-19T14:30:00,rcvr_02,40.867540,-124.083619,12.2,3,12,0.78,1.12,1.36
```

## API Endpoints

### RF Analysis
- `POST /api/analysis/load-csv` - Load RF CSV files
- `GET /api/data/access-points` - Get detected APs
- `POST /api/report/pdf` - Generate PDF report

### GNSS Link
- `POST /api/gnss/load-csv` - Load GNSS CSV files
- `GET /api/gnss/metrics` - Get variance metrics
- `GET /api/gnss/receivers` - Get receiver list
- `PUT /api/gnss/receivers/<id>` - Update receiver config
- `POST /api/gnss/report` - Generate GNSS PDF report
- `GET /api/gnss/export/csv` - Export to Option B CSV

### GNSS MQTT (NEW in v4.8.0)
- `POST /api/gnss-mqtt/connect` - Connect to GNSS MQTT broker
- `POST /api/gnss-mqtt/disconnect` - Disconnect from broker
- `GET /api/gnss-mqtt/status` - Get connection status and stats
- `GET /api/gnss-mqtt/positions` - Get recent positions
- `POST /api/gnss-mqtt/positions/clear` - Clear stored positions
- `POST /api/gnss-mqtt/import-to-session` - Import positions to GNSS session

## Configuration

Settings are persisted in `backend/user_data/settings.json`:

```json
{
  "report": {
    "format": "pdf",
    "dark_mode": true,
    "route_map_enabled": true
  }
}
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `FLASK_PORT` | 5000 | Server port |
| `FLASK_DEBUG` | False | Debug mode |
| `MQTT_HOST` | (none) | Wardriving MQTT broker |
| `MQTT_PORT` | 1883 | Wardriving MQTT port |
| `GNSS_MQTT_HOST` | (none) | GNSS MQTT broker (Pi IP) |
| `GNSS_MQTT_PORT` | 1883 | GNSS MQTT port |

## System Requirements

- Python 3.10, 3.11, or 3.12
- 4GB RAM minimum
- Modern web browser (Chrome, Firefox, Edge)

## Changelog

### v4.8.0 (2026-01-25)
- **NEW**: GNSS receivers display on main Map View with track lines and colored markers
- **NEW**: GNSS statistics panel replaces SOI stats when MQTT connected (auto-switch)
- **NEW**: Lock View feature - keep map centered on a specific receiver at fixed zoom
- **NEW**: POI (Point of Interest) management with up to 3 configurable radius circles
- **NEW**: GNSS receiver color configuration via JSON upload in Settings > GNSS Link
- **NEW**: SQLite export for GNSS tracks (works from MQTT buffer or session)
- **NEW**: GeoJSON export for GNSS tracks (LineString + Point features)
- **NEW**: Settings > GNSS Link tab for POI, colors, and export options
- **FIX**: Settings > GNSS Link tab now responds to clicks properly
- **FIX**: SQLite/GeoJSON export now works directly from MQTT buffer (no import required)
- **FIX**: Version consistency across all files, installers, and UI header

### v4.7.1 (2026-01-25)
- **NEW**: Live GNSS MQTT integration - receive real-time position data from Raspberry Pi
- **NEW**: GNSS MQTT API endpoints for connect/disconnect/status/import
- **NEW**: WebSocket events for live GNSS position streaming
- **NEW**: Import MQTT positions directly into GNSS session for analysis
- **NEW**: GNSS Link source toggle - switch between CSV upload and Live MQTT modes
- **NEW**: GNSS mini map showing receiver positions and tracks
- **NEW**: MQTT connection panel with real-time statistics (positions, devices, sats)

### v4.7.0 (2026-01-19)
- **NEW**: Timestamps in list view (First Seen + Last Seen, Zulu time)
- **NEW**: Sort dropdown (Recent, Oldest, Most/Fewest Detections, Strongest/Weakest, A-Z, Classification)
- **NEW**: Map search bar - search by name/MAC directly on map view
- **NEW**: Selected AP moves to top of table list
- **NEW**: Duplicate SSID detection filter checkbox
- **NEW**: Notification toast system for user feedback
- **FIX**: Map zoom no longer resets when deselecting an SOI
- **FIX**: GNSS Link - improved error logging and status reporting
- Highlighted row for selected AP in table view
- Added console debug logging for GNSS uploads

### v4.6.1 (2026-01-19)
- **NEW**: GNSS Link module for multi-receiver variance analysis
- **NEW**: Option B CSV format support (multi-receiver with ReceiverID)
- **NEW**: Truth receiver designation and comparison metrics
- **NEW**: GNSS PDF report generation with track overlay
- Rebranded to "Silver Streak Analyzer" (SSA)
- Integrated GNSS Link tab in main UI
- Added CEP (50%), 2DRMS (95%), and max deviation metrics

### v4.4.11
- Added basemap tile options (OSM, Satellite, Terrain)
- Version tab in settings
- Route map overlay improvements
