# Changelog

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
