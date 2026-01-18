"""Route map image generation utilities.

Generates a static PNG image of the survey route with optional basemap tiles.

Basemap behavior:
- If tiles exist in the local cache, they are used.
- If a tile is missing, we attempt to download it (best effort, short timeout).
- If downloading fails (offline), we fall back to a plain lat/lon plot.

This module is designed to work in offline environments while improving
output quality when network access (or cached tiles) is available.
"""

from __future__ import annotations

import io
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use('Agg')  # headless
import matplotlib.pyplot as plt

from PIL import Image

try:
    import requests
except Exception:
    requests = None


@dataclass
class RouteBounds:
    min_lat: float
    max_lat: float
    min_lon: float
    max_lon: float

    def padded(self, pct: float = 0.10) -> 'RouteBounds':
        pct = float(pct or 0.10)
        lat_span = self.max_lat - self.min_lat
        lon_span = self.max_lon - self.min_lon
        # Avoid zero-span bounding boxes
        if lat_span == 0:
            lat_span = 0.0005
        if lon_span == 0:
            lon_span = 0.0005
        pad_lat = lat_span * pct
        pad_lon = lon_span * pct
        return RouteBounds(
            self.min_lat - pad_lat,
            self.max_lat + pad_lat,
            self.min_lon - pad_lon,
            self.max_lon + pad_lon,
        )


def compute_bounds(points: Sequence[Tuple[float, float]]) -> Optional[RouteBounds]:
    if not points:
        return None
    lats = [p[0] for p in points]
    lons = [p[1] for p in points]
    return RouteBounds(min(lats), max(lats), min(lons), max(lons))


# ---- Slippy tile math ----

def _latlon_to_tile_frac(lat: float, lon: float, z: int) -> Tuple[float, float]:
    """Convert lat/lon to fractional XYZ tile coords at zoom z."""
    lat = max(min(lat, 85.05112878), -85.05112878)
    n = 2.0 ** z
    x = (lon + 180.0) / 360.0 * n
    lat_rad = math.radians(lat)
    y = (1.0 - math.log(math.tan(lat_rad) + (1.0 / math.cos(lat_rad))) / math.pi) / 2.0 * n
    return x, y


def _choose_zoom(bounds: RouteBounds, px_w: int = 1024, px_h: int = 640) -> int:
    """Choose a reasonable zoom so the bounds fit roughly within target pixels."""
    # Use lon span as a driver; choose zoom where world_px / span ~ px_w
    lon_span = max(1e-6, bounds.max_lon - bounds.min_lon)
    # px per degree at zoom z: 256*2^z / 360
    # We want lon_span * (256*2^z/360) ~ px_w
    z = math.log2((px_w * 360.0) / (256.0 * lon_span))
    z_int = int(max(1, min(19, round(z))))
    return z_int


def _fetch_tile(tile_url: str, cache_path: Path, timeout: float = 2.5) -> Optional[Image.Image]:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        try:
            return Image.open(cache_path).convert('RGBA')
        except Exception:
            pass

    if requests is None:
        return None

    try:
        headers = {
            'User-Agent': 'WardrivingAnalyzer/1.0 (route map report generator)'
        }
        r = requests.get(tile_url, headers=headers, timeout=timeout)
        if r.status_code == 200 and r.content:
            cache_path.write_bytes(r.content)
            return Image.open(io.BytesIO(r.content)).convert('RGBA')
    except Exception:
        return None
    return None


def _stitch_basemap(bounds: RouteBounds, cache_dir: Path, tile_template: str, z: int) -> Tuple[Optional[Image.Image], Optional[Tuple[int, int, int, int]]]:
    """Return (stitched_image, tile_range) where tile_range=(x0,y0,x1,y1) inclusive."""
    x0f, y1f = _latlon_to_tile_frac(bounds.min_lat, bounds.min_lon, z)
    x1f, y0f = _latlon_to_tile_frac(bounds.max_lat, bounds.max_lon, z)

    # Note y is inverted (north smaller), so y0f (for max_lat) is min y.
    x0 = int(math.floor(min(x0f, x1f)))
    x1 = int(math.floor(max(x0f, x1f)))
    y0 = int(math.floor(min(y0f, y1f)))
    y1 = int(math.floor(max(y0f, y1f)))

    # Limit tile count to prevent runaway image sizes
    max_tiles = 8  # 8x8=64 tiles max
    if (x1 - x0 + 1) > max_tiles or (y1 - y0 + 1) > max_tiles:
        # Zoom too high for the extent; caller should choose smaller zoom.
        return None, None

    w_tiles = (x1 - x0 + 1)
    h_tiles = (y1 - y0 + 1)
    out = Image.new('RGBA', (w_tiles * 256, h_tiles * 256), (255, 255, 255, 255))

    for yy in range(y0, y1 + 1):
        for xx in range(x0, x1 + 1):
            url = tile_template.format(z=z, x=xx, y=yy)
            cache_path = cache_dir / str(z) / str(xx) / f"{yy}.png"
            img = _fetch_tile(url, cache_path)
            if img is None:
                return None, None
            out.paste(img, ((xx - x0) * 256, (yy - y0) * 256))

    return out, (x0, y0, x1, y1)


def generate_route_png(
    points: Sequence[Tuple[float, float]],
    out_px: Tuple[int, int] = (1200, 800),
    padding_pct: float = 0.10,
    use_basemap: bool = True,
    tile_cache_dir: Optional[Path] = None,
    tile_template: str = 'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
) -> Optional[bytes]:
    """Generate a PNG route image.

    Args:
        points: sequence of (lat, lon)
        out_px: (width, height) output pixels
        padding_pct: bounds padding around route
        use_basemap: if True, attempt to use cached/downloaded tiles
        tile_cache_dir: cache directory for tiles
        tile_template: XYZ tile template

    Returns:
        PNG bytes or None if route not available.
    """
    if not points or len(points) < 2:
        return None

    bounds = compute_bounds(points)
    if not bounds:
        return None
    bounds = bounds.padded(padding_pct)

    w, h = int(out_px[0]), int(out_px[1])

    # Try basemap route plot in pixel space.
    if use_basemap:
        cache_dir = tile_cache_dir or Path(os.getcwd()) / 'tile_cache'
        # Choose zoom, then back off if too many tiles.
        z = _choose_zoom(bounds, px_w=w, px_h=h)
        stitched = None
        tile_range = None
        for zz in range(z, 0, -1):
            stitched, tile_range = _stitch_basemap(bounds, cache_dir, tile_template, zz)
            if stitched is not None:
                z = zz
                break
        if stitched is not None and tile_range is not None:
            x0, y0, x1, y1 = tile_range

            # Convert route points to pixel coords in stitched image
            xs = []
            ys = []
            for lat, lon in points:
                xf, yf = _latlon_to_tile_frac(lat, lon, z)
                px = (xf - x0) * 256.0
                py = (yf - y0) * 256.0
                xs.append(px)
                ys.append(py)

            # Plot using matplotlib for consistent styling
            fig = plt.figure(figsize=(w / 100.0, h / 100.0), dpi=100)
            ax = fig.add_subplot(111)
            ax.imshow(stitched)
            ax.plot(xs, ys, linewidth=2.0)
            ax.set_xlim(0, stitched.size[0])
            ax.set_ylim(stitched.size[1], 0)
            ax.axis('off')

            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
            plt.close(fig)
            return buf.getvalue()

    # Fallback: plain lat/lon plot (offline-safe)
    lats = [p[0] for p in points]
    lons = [p[1] for p in points]

    fig = plt.figure(figsize=(w / 100.0, h / 100.0), dpi=100)
    ax = fig.add_subplot(111)
    ax.plot(lons, lats, linewidth=2.0)
    ax.set_xlim(bounds.min_lon, bounds.max_lon)
    ax.set_ylim(bounds.min_lat, bounds.max_lat)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.grid(True, linewidth=0.4)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    return buf.getvalue()
