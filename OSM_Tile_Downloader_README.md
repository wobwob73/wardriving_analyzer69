# OSM Tile Downloader for Silver Streak Analyzer

A simple tool to download map tiles for offline use in SSA (Silver Streak Analyzer).

This lets you pre-download map tiles for a specific area so you can use SSA without an internet connection.

---

## Table of Contents

1. [What This Tool Does](#what-this-tool-does)
2. [Requirements](#requirements)
3. [Installation (Step-by-Step)](#installation-step-by-step)
4. [How to Use](#how-to-use)
5. [Examples](#examples)
6. [Copying Tiles to SSA](#copying-tiles-to-ssa)
7. [Troubleshooting](#troubleshooting)
8. [Available Map Styles](#available-map-styles)

---

## What This Tool Does

- Downloads map tiles from OpenStreetMap or CartoDB (the same tiles SSA uses)
- Saves them in a single `.mbtiles` file
- You can then copy this file to your SSA system for offline use

**Important:** This tool requires an internet connection to download the tiles. Once downloaded, SSA can use them offline.

---

## Requirements

- **Python 3.6 or newer** (already installed on most Linux systems)
- **Internet connection** (to download the tiles)
- **Disk space** for the tile file (depends on area size and zoom levels)

### Checking if Python is Installed

Open a terminal and type:

```bash
python3 --version
```

You should see something like `Python 3.10.12`. If you get an error, see [Installing Python](#installing-python-if-needed) below.

---

## Installation (Step-by-Step)

### Step 1: Create a folder for the tool

Open a terminal and copy/paste these commands one at a time:

```bash
mkdir -p ~/osm_tile_downloader
```

```bash
cd ~/osm_tile_downloader
```

### Step 2: Download the script

If you received `tile_downloader.py` as a file, copy it to the folder you just created.

**OR** if you're copying from this document, create the file:

```bash
nano tile_downloader.py
```

Then paste the script contents and save (Ctrl+O, Enter, Ctrl+X).

### Step 3: Make it executable

```bash
chmod +x tile_downloader.py
```

### Step 4: Verify it works

```bash
python3 tile_downloader.py --help
```

You should see the help message with all available options.

---

## How to Use

The basic command format is:

```bash
python3 tile_downloader.py --center LAT,LON --radius DISTANCE --zoom MIN-MAX --output FILENAME.mbtiles
```

### Understanding the Options

| Option | What it means | Example |
|--------|---------------|---------|
| `--center` | The center point of your area (latitude,longitude) | `--center 35.12,-79.45` |
| `--radius` | How far from the center to download (km or mi) | `--radius 10km` |
| `--zoom` | Zoom levels to download (higher = more detail) | `--zoom 10-16` |
| `--output` | What to name the output file | `--output my_area.mbtiles` |
| `--source` | Which map style to use (optional) | `--source cartodb-dark` |

### Finding Your Coordinates

1. Go to [Google Maps](https://maps.google.com)
2. Right-click on the center of your desired area
3. Click the coordinates that appear (they'll be copied)
4. They'll look like: `35.1234, -79.5678`

### Choosing Zoom Levels

| Zoom Level | Shows | Tile Count |
|------------|-------|------------|
| 10-12 | Regional view (counties/large areas) | Few tiles |
| 12-14 | City/town level | Medium |
| 14-16 | Neighborhoods/streets | Many tiles |
| 16-18 | Individual buildings | Very many tiles |
| 18-20 | Maximum detail | Huge number of tiles |

**Recommendation:** Start with `10-16` for most use cases. You can always download more detail later.

### Choosing Radius

| Radius | Good For |
|--------|----------|
| 5km | Small area, single neighborhood |
| 10km | Town or small city |
| 25km | Metro area |
| 50km+ | Regional coverage (large file!) |

---

## Examples

### Example 1: Download 10km area around a point

```bash
python3 tile_downloader.py --center 35.12,-79.45 --radius 10km --zoom 10-16 --output my_area.mbtiles
```

### Example 2: Download with CartoDB Dark theme (matches SSA default)

```bash
python3 tile_downloader.py --source cartodb-dark --center 35.12,-79.45 --radius 10km --zoom 10-16 --output dark_tiles.mbtiles
```

### Example 3: Download a specific rectangular area

Use `--bbox` instead of `--center` and `--radius`:

```bash
python3 tile_downloader.py --bbox 35.0,-79.5,35.3,-79.0 --zoom 10-16 --output region.mbtiles
```

The bbox format is: `SOUTH,WEST,NORTH,EAST` (minimum lat, minimum lon, maximum lat, maximum lon)

### Example 4: High-detail download of a small area

```bash
python3 tile_downloader.py --center 35.12,-79.45 --radius 2km --zoom 14-18 --output detailed.mbtiles
```

### Example 5: Large regional download

```bash
python3 tile_downloader.py --center 35.12,-79.45 --radius 50km --zoom 8-14 --output regional.mbtiles
```

Note: Large downloads may take a while and create big files!

---

## Copying Tiles to SSA

After downloading, you need to copy the `.mbtiles` file to your SSA system.

### If SSA is on the same computer:

```bash
cp my_area.mbtiles ~/.ssa/tiles/
```

(The exact path may vary based on your SSA installation)

### If SSA is on a different computer:

**Option A: USB Drive**
1. Copy the `.mbtiles` file to a USB drive
2. Plug the USB drive into your SSA computer
3. Copy the file to the SSA tiles directory

**Option B: SCP (if you have SSH access)**
```bash
scp my_area.mbtiles user@ssa-computer:~/.ssa/tiles/
```

### In SSA (Future Feature):

Once the Map Tiles settings tab is implemented:
1. Open SSA
2. Go to **Settings** → **Map Tiles**
3. Click **Import Map Tile File**
4. Select your `.mbtiles` file

---

## Troubleshooting

### "python3: command not found"

Python isn't installed. Install it:

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install python3
```

**Fedora:**
```bash
sudo dnf install python3
```

### "Permission denied"

Make the script executable:
```bash
chmod +x tile_downloader.py
```

### Download is very slow

The tool adds a small delay between downloads to be respectful to tile servers. You can reduce this (not recommended):
```bash
python3 tile_downloader.py --delay 0.05 ...other options...
```

### "Failed to download tile" warnings

Some tiles may fail due to network issues. The tool will retry automatically. A few failures are normal and won't significantly impact your offline map.

### File is very large

You requested too many zoom levels or too large an area. Try:
- Reducing the radius
- Reducing the maximum zoom level
- Downloading only the zoom levels you actually need

### "No module named 'X'" error

The script only uses built-in Python modules, so this shouldn't happen. If it does, make sure you're using Python 3.6 or newer:
```bash
python3 --version
```

---

## Available Map Styles

| Style | Command | Description |
|-------|---------|-------------|
| CartoDB Dark | `--source cartodb-dark` | Dark theme, matches SSA default **(RECOMMENDED)** |
| CartoDB Light | `--source cartodb-light` | Light/white theme |
| OpenStreetMap | `--source osm` | Standard OSM style |
| OpenTopoMap | `--source opentopomap` | Topographic with elevation |
| Stamen Terrain | `--source stamen-terrain` | Artistic terrain style |

If you don't specify `--source`, it defaults to `cartodb-dark`.

---

## Tile Count Estimates

Here's roughly how many tiles different configurations will download:

| Area | Zoom 10-14 | Zoom 10-16 | Zoom 10-18 |
|------|------------|------------|------------|
| 5km radius | ~200 | ~1,500 | ~20,000 |
| 10km radius | ~800 | ~6,000 | ~80,000 |
| 25km radius | ~5,000 | ~40,000 | ~500,000+ |
| 50km radius | ~20,000 | ~160,000 | ~2,000,000+ |

Each tile is roughly 10-20 KB, so:
- 1,000 tiles ≈ 10-20 MB
- 10,000 tiles ≈ 100-200 MB
- 100,000 tiles ≈ 1-2 GB

---

## Quick Reference Card

```
DOWNLOAD TILES FOR OFFLINE USE:
===============================

Basic command:
  python3 tile_downloader.py --center LAT,LON --radius 10km --zoom 10-16 --output tiles.mbtiles

Find coordinates:
  Right-click on Google Maps → Click the coordinates

Map styles (pick one):
  --source cartodb-dark   ← Recommended (matches SSA)
  --source cartodb-light
  --source osm
  --source opentopomap

Radius options:
  --radius 5km    Small area
  --radius 10km   Town
  --radius 25km   City/metro
  --radius 50km   Regional

Zoom options:
  --zoom 10-14   Overview only
  --zoom 10-16   Good balance (recommended)
  --zoom 10-18   High detail (large file)
```

---

## License & Attribution

This tool downloads tiles from various providers. Please respect their terms of use:

- **OpenStreetMap**: © OpenStreetMap contributors, ODbL license
- **CartoDB/CARTO**: © CARTO, tiles free for non-commercial use
- **OpenTopoMap**: © OpenTopoMap contributors
- **Stamen/Stadia**: © Stamen Design, © Stadia Maps

Tiles are for personal/offline use with Silver Streak Analyzer.
