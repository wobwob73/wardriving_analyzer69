# ⚡ Silver Streak Analyzer (SSA)

**Version 4.6.1** - Multi-Spectrum RF Survey & GNSS Variance Analysis

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

### 📡 GNSS Link Module (NEW in v4.6.1)
- Multi-receiver GNSS variance analysis
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
tar -xzf silver_streak_analyzer_v4.6.1.tar.gz
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

## System Requirements

- Python 3.10, 3.11, or 3.12
- 4GB RAM minimum
- Modern web browser (Chrome, Firefox, Edge)

## Changelog

### v4.6.2 (2026-01-19)
- **NEW**: Timestamps in list view (First Seen + Last Seen, Zulu time)
- **NEW**: Sort dropdown (Recent, Oldest, Most/Fewest Detections, Strongest/Weakest, A-Z, Classification)
- **NEW**: Map search bar - search by name/MAC directly on map view
- **NEW**: Selected AP auto-scrolls to visible in table view
- **FIX**: Map zoom no longer resets when deselecting an SOI
- Highlighted row for selected AP in table view

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
