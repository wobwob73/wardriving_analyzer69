## v4.4.11
- Settings: Report tab now exposes **Route Overview Map** options (enable, use basemap tiles, padding%).
- Settings: adds an **About** tab showing the running version and build timestamp (via `/api/version`).

## v4.4.10
- Reports: adds a **Survey Route Overview** map image (PNG) to both PDF and HTML reports. The map auto-expands the route bounds by **+10%** for additional geographic context.
- Basemap tiles: when available, the route overview uses cached/online map tiles; when offline, it falls back to a plain route plot. Tiles are cached under `backend/user_data/tile_cache/`.

## v4.4.9
- Settings modal: fixes Charts/Whitelist tabs not rendering due to pane nesting/closing-tag issues.

# Changelog



## v4.4.9
- Settings modal: fixed DOM nesting so Charts and Whitelist panes render correctly (they were accidentally nested inside the Report pane and became invisible when switching tabs).

## v4.4.8
- Settings modal: fixed tab panes not rendering on some browsers by removing invalid HTML closing tags in embedded UI.

## v4.4.7
- Settings modal: added tabbed navigation (General/Report/Charts/Whitelist) so all settings are reachable without requiring scroll-wheel/trackpad scrolling.
- PDF reports: small annotation text now renders in black for readability on white pages.

## v4.4.6
- Settings modal: removed nested scroll container for the CSV profile list to avoid mouse-wheel “trap” behavior; report options and other sections are now reachable by scrolling the modal.
- PDF reports: overview pie chart titles/legends are rendered in black for better readability.

## v4.4.5
- Settings modal is now scrollable on smaller screens (fixes only the top portion being visible).
- PDF report overview charts layout tightened to prevent label/legend overlap inside the pie chart drawings.

## v4.4.4
- Adds optional **overview pie charts** near the top of PDF and HTML reports:
  - Signal type distribution (WiFi / BT-BLE / Thread / HaLow / etc)
  - Static vs Mobile vs Uncertain
  - Most-detected channels (Top-N + Other)
- Adds report settings toggles for enabling/disabling charts and setting Top-N.

## v4.4.3
- Playback UI is now **docked** in the left SOI Info panel (no popup).
- Playback REST loading fixed:
  - Client now accepts `/api/data/detections` payloads that return `data` (REST) or `detections` (SocketIO).
  - Uses the correct query arg `include_no_gps`.
  - Adds an auto-start path so pressing play immediately starts once detections finish loading.

## v4.4.2
- PDF export hotfix (ReportLab font option regression fix).

## v4.4.1
- Fix attempt for PDF export font option state.
- Playback popup introduced (later replaced by dock in v4.4.3).

## v4.4.0
- Map hover tooltips for selection rings.
- SOI info panel layout fixes and selection/visibility improvements.

