"""
RF Site Survey Analysis Engine
Handles clustering, classification, and statistical analysis of wardriving data
Supports both GPS-located and non-GPS signal data
"""

import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from datetime import datetime
import logging
from typing import Dict, List, Optional
import csv
import io
import math

logger = logging.getLogger(__name__)


class AccessPoint:
    """Represents a single Access Point with statistical analysis"""
    
    def __init__(self, mac: str, name: str = "", signal_type: str = "WIFI"):
        self.mac = mac
        self.name = name
        self.signal_type = signal_type
        self.detections = []
        self.locations = []  # Only GPS-valid detections
        self.rssi_values = []
        self.workers = set()
        self.channels = set()
        # WiFi security/encryption modes observed for this SOI (canonicalized).
        self.security_modes = set()
        self.run_ids = set()
        self.first_seen = None
        self.last_seen = None
        
    def add_detection(self, lat: float, lon: float, rssi: int, worker: str, 
                     channel: Optional[str] = None, timestamp: Optional[str] = None,
                     run_id: Optional[str] = None, has_gps: bool = True,
                     security: Optional[str] = None):
        """Add a single detection to this AP"""
        sec = self._normalize_security(security)
        detection = {
            'lat': lat,
            'lon': lon,
            'rssi': rssi,
            'worker': worker,
            'channel': channel,
            'timestamp': timestamp,
            'has_gps': has_gps,
            'security': sec,
            'run_id': run_id  # Store run_id for filtering
        }
        
        self.detections.append(detection)
        self.rssi_values.append(rssi)
        self.workers.add(worker)

        if sec:
            self.security_modes.add(sec)
        
        if channel:
            self.channels.add(str(channel))
        if run_id:
            self.run_ids.add(run_id)
            
        # Track timestamps
        if timestamp:
            if self.first_seen is None or timestamp < self.first_seen:
                self.first_seen = timestamp
            if self.last_seen is None or timestamp > self.last_seen:
                self.last_seen = timestamp
        
        # Only add to locations if valid GPS
        if has_gps and lat != 0.0 and lon != 0.0:
            self.locations.append((lat, lon))
    
    def get_statistics(self) -> Dict:
        """Calculate comprehensive statistics for this AP"""
        if not self.rssi_values:
            return {}
        
        rssi_array = np.array(self.rssi_values)
        
        stats = {
            'detection_count': len(self.detections),
            'rssi_mean': float(np.mean(rssi_array)),
            'rssi_std': float(np.std(rssi_array)),
            'rssi_min': int(np.min(rssi_array)),
            'rssi_max': int(np.max(rssi_array)),
            'rssi_range': int(np.ptp(rssi_array)),
            'unique_workers': len(self.workers),
            'unique_runs': len(self.run_ids),
            'channels': list(self.channels),
            'has_gps': len(self.locations) > 0,
            'gps_detections': len(self.locations)
        }

        # Security posture (WiFi only; may be UNKNOWN for other types)
        stats['security_modes'] = sorted([s for s in self.security_modes if s])
        stats['primary_security'] = self._primary_security()
        
        # Calculate geographic stats only if we have GPS data
        # NOTE: use detections (not just self.locations) so we can compute weighted centroids
        # and observed radius from the same set of points.
        gps_dets = [
            d for d in self.detections
            if d.get('has_gps') and d.get('lat', 0.0) != 0.0 and d.get('lon', 0.0) != 0.0
        ]

        if gps_dets:
            locs = np.array([(d['lat'], d['lon']) for d in gps_dets], dtype=float)
            
            # Centroid
            stats['centroid_lat'] = float(np.mean(locs[:, 0]))
            stats['centroid_lon'] = float(np.mean(locs[:, 1]))

            # Weighted centroid (simple RSSI-weighted estimate of likely AP location)
            # Weighting is intentionally conservative: stronger RSSI contributes more.
            weights = np.array([
                max(1.0, 100.0 + float(d.get('rssi', -100))) for d in gps_dets
            ], dtype=float)
            try:
                stats['weighted_centroid_lat'] = float(np.average(locs[:, 0], weights=weights))
                stats['weighted_centroid_lon'] = float(np.average(locs[:, 1], weights=weights))
            except Exception:
                # Fallback to unweighted centroid
                stats['weighted_centroid_lat'] = stats['centroid_lat']
                stats['weighted_centroid_lon'] = stats['centroid_lon']
            
            # Geographic spread
            if len(locs) > 1:
                lat_spread = np.ptp(locs[:, 0])
                lon_spread = np.ptp(locs[:, 1])
                stats['geographic_spread_km'] = float(np.sqrt((lat_spread * 111)**2 + (lon_spread * 111)**2))
            else:
                stats['geographic_spread_km'] = 0.0

            # Observed radius from (weighted) centroid to farthest detection point.
            # This is a useful "effective captured radius" metric for UI.
            wc_lat = stats.get('weighted_centroid_lat', stats['centroid_lat'])
            wc_lon = stats.get('weighted_centroid_lon', stats['centroid_lon'])
            if len(locs) > 1:
                stats['observed_radius_m'] = float(max(
                    self._haversine_m(wc_lat, wc_lon, float(lat), float(lon))
                    for lat, lon in locs
                ))
            else:
                stats['observed_radius_m'] = 0.0
            stats['strongest_rssi'] = 0.0
            stats['strongest_lat'] = 0.0
            stats['strongest_lon'] = 0.0
            stats['strongest_radius_m'] = 0.0
        
            # Strongest-signal point (best observed location)
            # Use highest RSSI (least negative) among GPS-valid detections.
            best = max(gps_dets, key=lambda d: float(d.get('rssi', -999)))
            stats['strongest_rssi'] = float(best.get('rssi', -999))
            stats['strongest_lat'] = float(best.get('lat', 0.0))
            stats['strongest_lon'] = float(best.get('lon', 0.0))

            # Strongest-signal region radius (smallest, used for focus overlay):
            # take all points within N dB of the best RSSI and compute max distance
            # from the strongest point to those points.
            # This provides a compact "best heard" footprint.
            try:
                thr_db = 6.0
                best_rssi = float(stats['strongest_rssi'])
                near_best = [d for d in gps_dets if float(d.get('rssi', -999)) >= (best_rssi - thr_db)]
                bs_lat = stats['strongest_lat']
                bs_lon = stats['strongest_lon']
                if near_best and bs_lat and bs_lon:
                    stats['strongest_region_radius_m'] = float(max(
                        self._haversine_m(bs_lat, bs_lon, float(d.get('lat', 0.0)), float(d.get('lon', 0.0)))
                        for d in near_best
                        if d.get('lat', 0.0) and d.get('lon', 0.0)
                    ))
                else:
                    stats['strongest_region_radius_m'] = 0.0
                stats['strongest_region_threshold_db'] = thr_db
            except Exception:
                stats['strongest_region_radius_m'] = 0.0
                stats['strongest_region_threshold_db'] = 6.0

            # Observed radius from strongest point to farthest detection point.
            bs_lat = stats['strongest_lat']
            bs_lon = stats['strongest_lon']
            if len(locs) > 1 and bs_lat and bs_lon:
                stats['strongest_radius_m'] = float(max(
                    self._haversine_m(bs_lat, bs_lon, float(lat), float(lon))
                    for lat, lon in locs
                ))
            else:
                stats['strongest_radius_m'] = 0.0

            # First/Last seen (GPS-valid) detection info
            def _parse_ts(ts: str):
                if not ts:
                    return None
                try:
                    t = str(ts).replace('Z', '+00:00')
                    return datetime.fromisoformat(t)
                except Exception:
                    return None

            try:
                gps_with_ts = [(d, _parse_ts(d.get('timestamp') or '')) for d in gps_dets]
                gps_with_ts_valid = [(d, dt) for d, dt in gps_with_ts if dt is not None]
                if gps_with_ts_valid:
                    first_d, first_dt = min(gps_with_ts_valid, key=lambda x: x[1])
                    last_d, last_dt = max(gps_with_ts_valid, key=lambda x: x[1])
                else:
                    # Fall back to insertion order
                    first_d = gps_dets[0]
                    last_d = gps_dets[-1]
                    first_dt = None
                    last_dt = None

                stats['first_seen_lat'] = float(first_d.get('lat', 0.0))
                stats['first_seen_lon'] = float(first_d.get('lon', 0.0))
                stats['first_seen_rssi'] = float(first_d.get('rssi', -999))
                stats['first_seen_ts'] = str(first_d.get('timestamp') or '')

                stats['last_seen_lat'] = float(last_d.get('lat', 0.0))
                stats['last_seen_lon'] = float(last_d.get('lon', 0.0))
                stats['last_seen_rssi'] = float(last_d.get('rssi', -999))
                stats['last_seen_ts'] = str(last_d.get('timestamp') or '')

                # A simple "first/last boundary" radius around the strongest point.
                if bs_lat and bs_lon:
                    r_first = self._haversine_m(bs_lat, bs_lon, stats['first_seen_lat'], stats['first_seen_lon']) if stats['first_seen_lat'] and stats['first_seen_lon'] else 0.0
                    r_last = self._haversine_m(bs_lat, bs_lon, stats['last_seen_lat'], stats['last_seen_lon']) if stats['last_seen_lat'] and stats['last_seen_lon'] else 0.0
                    stats['first_last_radius_m'] = float(max(r_first, r_last))
                else:
                    stats['first_last_radius_m'] = 0.0
            except Exception:
                stats['first_seen_lat'] = 0.0
                stats['first_seen_lon'] = 0.0
                stats['first_seen_rssi'] = 0.0
                stats['first_seen_ts'] = ''
                stats['last_seen_lat'] = 0.0
                stats['last_seen_lon'] = 0.0
                stats['last_seen_rssi'] = 0.0
                stats['last_seen_ts'] = ''
                stats['first_last_radius_m'] = 0.0

            # RSSI-weighted ellipse ("most likely" area) in meters.
            # We project lat/lon to a local tangent plane around the weighted centroid.
            try:
                wc_lat = float(stats.get('weighted_centroid_lat', stats['centroid_lat']))
                wc_lon = float(stats.get('weighted_centroid_lon', stats['centroid_lon']))
                if len(locs) >= 3 and wc_lat and wc_lon:
                    # Convert to local meters (equirectangular approximation)
                    lat0 = math.radians(wc_lat)
                    m_per_deg_lat = 111111.0
                    m_per_deg_lon = 111111.0 * math.cos(lat0)
                    xy = np.zeros((len(gps_dets), 2), dtype=float)
                    for i, d in enumerate(gps_dets):
                        dy = (float(d.get('lat', 0.0)) - wc_lat) * m_per_deg_lat
                        dx = (float(d.get('lon', 0.0)) - wc_lon) * m_per_deg_lon
                        xy[i, 0] = dx  # Easting
                        xy[i, 1] = dy  # Northing

                    w = np.array([max(1.0, 100.0 + float(d.get('rssi', -100))) for d in gps_dets], dtype=float)
                    w_sum = float(np.sum(w))
                    if w_sum <= 0:
                        raise ValueError('bad weights')
                    w_norm = w / w_sum
                    mu = np.sum(xy * w_norm[:, None], axis=0)
                    xc = xy - mu
                    cov = np.zeros((2, 2), dtype=float)
                    for i in range(xc.shape[0]):
                        cov += w_norm[i] * np.outer(xc[i], xc[i])

                    # Eigen decomposition
                    vals, vecs = np.linalg.eigh(cov)
                    order = np.argsort(vals)[::-1]
                    vals = vals[order]
                    vecs = vecs[:, order]

                    # 2-sigma ellipse radii (approx 95% for Gaussian)
                    a = float(2.0 * math.sqrt(max(vals[0], 1e-9)))
                    b = float(2.0 * math.sqrt(max(vals[1], 1e-9)))

                    # Clamp to reasonable minimum so it renders even for tight clusters.
                    stats['ellipse_a_m'] = float(max(a, 10.0))
                    stats['ellipse_b_m'] = float(max(b, 10.0))

                    # Angle in degrees (0 = East, 90 = North)
                    v0 = vecs[:, 0]
                    ang = math.degrees(math.atan2(float(v0[1]), float(v0[0])))
                    stats['ellipse_angle_deg'] = float(ang)
                    stats['ellipse_center_lat'] = wc_lat
                    stats['ellipse_center_lon'] = wc_lon
                    stats['ellipse_sigma'] = 2.0
                else:
                    stats['ellipse_a_m'] = 0.0
                    stats['ellipse_b_m'] = 0.0
                    stats['ellipse_angle_deg'] = 0.0
                    stats['ellipse_center_lat'] = wc_lat
                    stats['ellipse_center_lon'] = wc_lon
                    stats['ellipse_sigma'] = 2.0
            except Exception:
                stats['ellipse_a_m'] = 0.0
                stats['ellipse_b_m'] = 0.0
                stats['ellipse_angle_deg'] = 0.0
                stats['ellipse_center_lat'] = float(stats.get('weighted_centroid_lat', 0.0) or 0.0)
                stats['ellipse_center_lon'] = float(stats.get('weighted_centroid_lon', 0.0) or 0.0)
                stats['ellipse_sigma'] = 2.0
        else:
            stats['centroid_lat'] = 0.0
            stats['centroid_lon'] = 0.0
            stats['weighted_centroid_lat'] = 0.0
            stats['weighted_centroid_lon'] = 0.0
            stats['geographic_spread_km'] = 0.0
            stats['observed_radius_m'] = 0.0
            stats['strongest_rssi'] = 0.0
            stats['strongest_lat'] = 0.0
            stats['strongest_lon'] = 0.0
            stats['strongest_radius_m'] = 0.0
            stats['strongest_region_radius_m'] = 0.0
            stats['strongest_region_threshold_db'] = 6.0
            stats['first_seen_lat'] = 0.0
            stats['first_seen_lon'] = 0.0
            stats['first_seen_rssi'] = 0.0
            stats['first_seen_ts'] = ''
            stats['last_seen_lat'] = 0.0
            stats['last_seen_lon'] = 0.0
            stats['last_seen_rssi'] = 0.0
            stats['last_seen_ts'] = ''
            stats['first_last_radius_m'] = 0.0
            stats['ellipse_a_m'] = 0.0
            stats['ellipse_b_m'] = 0.0
            stats['ellipse_angle_deg'] = 0.0
            stats['ellipse_center_lat'] = 0.0
            stats['ellipse_center_lon'] = 0.0
            stats['ellipse_sigma'] = 2.0
        
        return stats

    @staticmethod
    def _normalize_security(security: Optional[str]) -> str:
        """Canonicalize security strings into a small set of categories.

        Categories:
          OPEN, OWE, WEP, WPA1, WPA2, WPA3, WPA2/WPA3, UNKNOWN
        """
        if not security:
            return ''
        s = str(security).strip()
        if not s or s.lower() == 'nan':
            return ''
        up = s.upper()

        # Common "open" markers
        if any(tok in up for tok in ['OPEN', 'NONE', 'UNSEC', 'NO ENC', 'NO_ENC', 'NOENCRYPT']):
            # But treat OWE (enhanced open) distinctly if mentioned.
            if 'OWE' in up:
                return 'OWE'
            return 'OPEN'

        if 'WEP' in up:
            return 'WEP'

        # WPA3 markers
        if any(tok in up for tok in ['WPA3', 'SAE']):
            # If it also includes WPA2, mark mixed mode.
            if 'WPA2' in up:
                return 'WPA2/WPA3'
            return 'WPA3'

        # WPA2 markers
        if 'WPA2' in up:
            return 'WPA2'

        # WPA1 / legacy WPA markers
        if 'WPA' in up:
            return 'WPA1'

        # OWE markers without explicit OPEN above
        if 'OWE' in up:
            return 'OWE'

        return 'UNKNOWN'

    def _primary_security(self) -> str:
        """Pick a single security label for filtering/summaries."""
        if not self.security_modes:
            return 'UNKNOWN'
        modes = set([m for m in self.security_modes if m])
        if not modes:
            return 'UNKNOWN'

        # Explicit mixed mode
        if 'WPA2/WPA3' in modes:
            return 'WPA2/WPA3'
        if 'WPA2' in modes and 'WPA3' in modes:
            return 'WPA2/WPA3'

        order = ['WPA3', 'WPA2', 'WPA1', 'WEP', 'OWE', 'OPEN', 'UNKNOWN']
        for k in order:
            if k in modes:
                return k
        return 'UNKNOWN'

    @staticmethod
    def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Great-circle distance in meters between two lat/lon points."""
        # Earth radius (mean) in meters
        r = 6371000.0
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)

        a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
        c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
        return r * c
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        stats = self.get_statistics()
        return {
            'mac': self.mac,
            'name': self.name,
            'signal_type': self.signal_type,
            'statistics': stats,
            'workers': list(self.workers),
            'first_seen': self.first_seen,
            'last_seen': self.last_seen
        }


class AnalysisEngine:
    """Main analysis engine for wardriving data"""
    
    def __init__(self, 
                 dbscan_eps_km: float = 0.1,
                 dbscan_min_samples: int = 3,
                 rssi_variance_threshold: float = 20.0,
                 geographic_spread_threshold: float = 0.5):
        
        self.dbscan_eps_km = dbscan_eps_km
        self.dbscan_min_samples = dbscan_min_samples
        self.rssi_variance_threshold = rssi_variance_threshold
        self.geographic_spread_threshold = geographic_spread_threshold
        
        self.access_points: Dict[str, AccessPoint] = {}
        self.classifications: Dict[str, Dict] = {}
        
    def clear(self):
        """Clear all data"""
        self.access_points = {}
        self.classifications = {}
        logger.info("All data cleared")
        

    def load_csv(self, filepath: str, run_id: str = None, profile: dict = None) -> Dict:
        """Load and ingest a wardriving CSV file.

        Args:
            filepath: Path to CSV file
            run_id: Optional run/session id
            profile: Optional profile dict that contains a 'mapping' dict, or a mapping dict directly.

        The mapping supports either a string column name or a list of candidate column names.
        Supported canonical keys:
            signal_type, mac, name, rssi, lat, lon, fix_ok, worker, channel, timestamp, security
        Optional:
            security_parts: list of columns to concatenate (e.g. [encryption, cipher, akm])
        """
        try:
            # Read file and normalize line endings (handles mixed CRLF/LF)
            with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read().replace('\r\n', '\n').replace('\r', '\n')

            # Parse CSV from normalized content
            from io import StringIO
            df = pd.read_csv(StringIO(content), on_bad_lines='skip', quotechar='"', index_col=False)
            logger.info(f"Loaded CSV: {filepath} ({len(df)} rows)")

            if run_id is None:
                run_id = datetime.now().isoformat()

            # Pull mapping from profile
            mapping = None
            security_parts = None
            if profile and isinstance(profile, dict):
                if isinstance(profile.get('mapping'), dict):
                    mapping = profile.get('mapping')
                    security_parts = profile.get('security_parts') or mapping.get('security_parts')
                else:
                    mapping = profile
                    security_parts = profile.get('security_parts')

            def _mapped_value(row, key, default=None):
                if not mapping:
                    return default
                col = mapping.get(key)
                if not col:
                    return default
                if isinstance(col, list):
                    for c in col:
                        try:
                            if c in row and pd.notna(row.get(c)):
                                return row.get(c)
                        except Exception:
                            continue
                    return default
                try:
                    if col in row and pd.notna(row.get(col)):
                        return row.get(col)
                except Exception:
                    return default
                return default

            def _truthy(v) -> int:
                try:
                    if v is None:
                        return 0
                    if isinstance(v, (int, float)):
                        return 1 if int(v) != 0 else 0
                    s = str(v).strip().lower()
                    return 1 if s in ('1', 'true', 't', 'yes', 'y', 'ok', 'valid') else 0
                except Exception:
                    return 0

            ingested = 0
            ingested_with_gps = 0
            ingested_no_gps = 0
            skipped_imu = 0

            for _, row in df.iterrows():
                # Extract key fields
                signal_type = str(_mapped_value(row, 'signal_type', row.get('signal_type', 'UNKNOWN'))).upper()

                # Skip IMU/ACCEL data
                if signal_type in ['IMU', 'ACCEL']:
                    skipped_imu += 1
                    continue

                mac_raw = _mapped_value(row, 'mac', row.get('identifier', ''))
                mac = str(mac_raw).lower().strip('"').strip()
                if not mac or mac == 'nan':
                    continue

                name_raw = _mapped_value(row, 'name', row.get('name', ''))
                name = str(name_raw).strip('"').strip() if name_raw is not None else ''
                if name == 'nan' or pd.isna(name_raw):
                    name = ''

                # Handle NaN values for numeric fields
                rssi_val = _mapped_value(row, 'rssi', row.get('rssi_dbm', -100))
                try:
                    rssi = int(float(rssi_val)) if pd.notna(rssi_val) else -100
                except Exception:
                    rssi = -100

                worker_raw = _mapped_value(row, 'worker', row.get('worker', 'unknown'))
                worker = str(worker_raw) if pd.notna(worker_raw) else 'unknown'
                if worker == 'nan' or worker.strip() == '':
                    worker = 'unknown'

                channel_val = _mapped_value(row, 'channel', row.get('channel', ''))
                channel = str(channel_val) if pd.notna(channel_val) else ''

                timestamp_val = _mapped_value(row, 'timestamp', row.get('iso8601_utc', ''))
                timestamp = str(timestamp_val) if pd.notna(timestamp_val) else ''

                # Security/encryption
                security = ''
                sec_val = _mapped_value(row, 'security', None)
                if sec_val is not None:
                    security = str(sec_val)
                else:
                    for key in ['security', 'encryption', 'auth', 'privacy', 'capabilities', 'wifi_security', 'sec']:
                        if key in row and pd.notna(row.get(key)):
                            security = str(row.get(key))
                            break

                # If profile specifies security_parts, optionally build a richer label
                if security_parts and isinstance(security_parts, list):
                    parts = []
                    for k in security_parts:
                        try:
                            v = row.get(k)
                            if pd.notna(v) and str(v).strip() != '':
                                parts.append(str(v).strip())
                        except Exception:
                            continue
                    if parts:
                        security = ' / '.join(parts)

                # Check GPS validity
                lat_val = _mapped_value(row, 'lat', row.get('lat', 0.0))
                lon_val = _mapped_value(row, 'lon', row.get('lon', 0.0))
                try:
                    lat = float(lat_val) if pd.notna(lat_val) else 0.0
                except Exception:
                    lat = 0.0
                try:
                    lon = float(lon_val) if pd.notna(lon_val) else 0.0
                except Exception:
                    lon = 0.0

                fix_ok_val = _mapped_value(row, 'fix_ok', row.get('fix_ok', 0))
                fix_ok = _truthy(fix_ok_val)

                has_gps = (fix_ok == 1 and lat != 0.0 and lon != 0.0)

                # Create or get AP
                if mac not in self.access_points:
                    self.access_points[mac] = AccessPoint(mac, name, signal_type)
                elif name and not self.access_points[mac].name:
                    self.access_points[mac].name = name

                # Add detection (with or without GPS)
                self.access_points[mac].add_detection(
                    lat, lon, rssi, worker, channel, timestamp, run_id, has_gps, security
                )

                ingested += 1
                if has_gps:
                    ingested_with_gps += 1
                else:
                    ingested_no_gps += 1

            result = {
                'total_rows': ingested,
                'with_gps': ingested_with_gps,
                'without_gps': ingested_no_gps,
                'unique_aps': len(self.access_points),
                'skipped_imu': skipped_imu
            }

            logger.info(
                f"Ingested {ingested} detections ({ingested_with_gps} with GPS, {ingested_no_gps} without) "
                f"into {len(self.access_points)} APs"
            )
            return result

        except Exception as e:
            logger.error(f"Error loading CSV {filepath}: {e}")
            raise

    def _lat_lon_to_km(self, locs_deg: np.ndarray) -> np.ndarray:
        """Convert lat/lon coordinates to kilometers for clustering"""
        if len(locs_deg) == 0:
            return np.array([])
        
        ref_lat = locs_deg[0, 0]
        locs_km = np.zeros_like(locs_deg)
        
        locs_km[:, 0] = (locs_deg[:, 0] - ref_lat) * 111.0
        lat_rad = np.radians(ref_lat)
        locs_km[:, 1] = (locs_deg[:, 1] - locs_deg[0, 1]) * 111.0 * np.cos(lat_rad)
        
        return locs_km
    
    def classify_access_points(self) -> Dict[str, Dict]:
        """
        Classify APs as static, mobile, or uncertain
        Works for both GPS and non-GPS data
        
        IMPORTANT: When an AP has GPS data, only GPS-valid detections are used
        for classification scoring. Non-GPS detections don't affect the score.
        """
        logger.info(f"Classifying {len(self.access_points)} access points...")
        self.classifications = {}
        
        for mac, ap in self.access_points.items():
            stats = ap.get_statistics()
            
            if not stats or len(ap.rssi_values) == 0:
                continue
            
            # Check if we have GPS data
            has_gps_data = len(ap.locations) > 0
            
            # For non-GPS data, we rely on RSSI patterns from all detections
            if not has_gps_data:
                # No GPS - classify based on RSSI patterns only
                rssi_std = stats['rssi_std']
                rssi_variance_score = min(rssi_std / self.rssi_variance_threshold, 1.0)
                
                if rssi_std < 5.0:
                    classification = 'static'
                    confidence = 0.7 - (rssi_std / 10.0)
                elif rssi_std > 15.0:
                    classification = 'mobile'
                    confidence = 0.5 + min(rssi_std / 40.0, 0.3)
                else:
                    classification = 'uncertain'
                    confidence = 0.5
                
                # Adjust confidence based on detection count
                detection_factor = min(len(ap.detections) / 50.0, 1.0)
                confidence = confidence * (0.5 + 0.5 * detection_factor)
                
                self.classifications[mac] = {
                    'classification': classification,
                    'confidence': float(confidence),
                    'mobile_score': float(rssi_variance_score),
                    'rssi_variance_score': float(rssi_variance_score),
                    'geographic_spread_score': 0.0,
                    'cluster_score': 0.5,
                    'consistency_score': 0.5,
                    'has_gps': False,
                    'statistics': stats,
                    'ap_data': {
                        'mac': mac,
                        'name': ap.name,
                        'signal_type': ap.signal_type,
                        'workers': list(ap.workers),
                        'channels': list(ap.channels),
                        'run_ids': list(ap.run_ids)
                    }
                }
                continue
            
            # GPS-based classification - ONLY use GPS-valid detections
            # Get RSSI values only from GPS-valid detections
            gps_detections = [d for d in ap.detections if d.get('has_gps', False)]
            gps_rssi_values = [d.get('rssi', -100) for d in gps_detections]
            
            if len(gps_rssi_values) >= 2:
                gps_rssi_array = np.array(gps_rssi_values)
                gps_rssi_std = float(np.std(gps_rssi_array))
            else:
                # Not enough GPS detections for meaningful RSSI variance
                gps_rssi_std = stats['rssi_std']  # Fall back to all detections
            
            # Feature 1: RSSI variance from GPS-valid detections only
            rssi_variance_score = min(gps_rssi_std / self.rssi_variance_threshold, 1.0)
            
            # Feature 2: Geographic spread (already GPS-only from stats)
            geo_spread = stats['geographic_spread_km']
            geo_spread_score = min(geo_spread / self.geographic_spread_threshold, 1.0)
            
            # Feature 3: Clustering analysis
            locs_deg = np.array(ap.locations)
            
            if len(locs_deg) >= self.dbscan_min_samples:
                locs_km = self._lat_lon_to_km(locs_deg)
                
                try:
                    clustering = DBSCAN(eps=self.dbscan_eps_km, 
                                       min_samples=self.dbscan_min_samples).fit(locs_km)
                    labels = clustering.labels_
                    
                    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                    n_noise = list(labels).count(-1)
                    
                    if n_clusters > 1 or n_noise > len(labels) * 0.2:
                        cluster_score = 0.8
                    else:
                        cluster_score = 0.1
                except Exception as e:
                    logger.warning(f"Clustering failed for {mac}: {e}")
                    cluster_score = 0.5
            else:
                cluster_score = 0.3
            
            # Feature 4: Detection frequency and consistency
            run_consistency = stats['unique_runs'] / max(1, len(ap.run_ids)) if ap.run_ids else 0.5
            static_score_from_consistency = run_consistency
            
            # Weighted combination
            mobile_score = (
                rssi_variance_score * 0.3 +
                geo_spread_score * 0.3 +
                cluster_score * 0.3 +
                (1.0 - static_score_from_consistency) * 0.1
            )
            
            # Classification
            if mobile_score < 0.35:
                classification = 'static'
                confidence = 1.0 - mobile_score
            elif mobile_score > 0.65:
                classification = 'mobile'
                confidence = mobile_score
            else:
                classification = 'uncertain'
                confidence = 0.5 + abs(mobile_score - 0.5)
            
            self.classifications[mac] = {
                'classification': classification,
                'confidence': float(confidence),
                'mobile_score': float(mobile_score),
                'rssi_variance_score': float(rssi_variance_score),
                'geographic_spread_score': float(geo_spread_score),
                'cluster_score': float(cluster_score),
                'consistency_score': float(static_score_from_consistency),
                'has_gps': True,
                'statistics': stats,
                'ap_data': {
                    'mac': mac,
                    'name': ap.name,
                    'signal_type': ap.signal_type,
                    'workers': list(ap.workers),
                    'channels': list(ap.channels),
                    'run_ids': list(ap.run_ids)
                }
            }
        
        logger.info(f"Classification complete. {len(self.classifications)} APs classified")
        return self.classifications
    
    def get_summary_stats(self) -> Dict:
        """Get high-level summary statistics"""
        if not self.classifications:
            # Return basic stats even without classification
            total_detections = sum(len(ap.detections) for ap in self.access_points.values())
            aps_with_gps = sum(1 for ap in self.access_points.values() if ap.locations)
            
            return {
                'total_aps': len(self.access_points),
                'total_detections': total_detections,
                'static': 0,
                'mobile': 0,
                'uncertain': 0,
                'average_confidence': 0.0,
                'with_gps': aps_with_gps
            }
        
        static_count = sum(1 for c in self.classifications.values() 
                          if c['classification'] == 'static')
        mobile_count = sum(1 for c in self.classifications.values() 
                          if c['classification'] == 'mobile')
        uncertain_count = sum(1 for c in self.classifications.values() 
                             if c['classification'] == 'uncertain')
        
        avg_confidence = np.mean([c['confidence'] for c in self.classifications.values()])
        
        total_detections = sum(len(ap.detections) for ap in self.access_points.values())
        aps_with_gps = sum(1 for c in self.classifications.values() if c.get('has_gps', False))
        
        return {
            'total_aps': len(self.classifications),
            'static': static_count,
            'mobile': mobile_count,
            'uncertain': uncertain_count,
            'average_confidence': float(avg_confidence),
            'total_detections': total_detections,
            'with_gps': aps_with_gps
        }
    
    def get_filtered_results(self, 
                            classification: Optional[str] = None,
                            worker: Optional[str] = None,
                            device_type: Optional[str] = None,
                            include_no_gps: bool = True) -> List[Dict]:
        """Get filtered results for API response"""
        results = []
        
        for mac, classification_data in self.classifications.items():
            ap = self.access_points[mac]
            
            # Apply filters
            if classification and classification_data['classification'] != classification:
                continue
            if worker and worker not in ap.workers:
                continue
            if device_type and self._categorize_device(ap.signal_type) != device_type.lower():
                continue
            if not include_no_gps and not classification_data.get('has_gps', False):
                continue
            
            device_category = self._categorize_device(ap.signal_type)
            stats = classification_data['statistics']
            
            result = {
                'id': mac,
                'mac': mac,
                'name': ap.name,
                'type': ap.signal_type,
                'device_category': device_category,
                'worker': list(ap.workers)[0] if ap.workers else 'unknown',
                'workers': list(ap.workers),
                'classification': classification_data['classification'],
                'confidence': classification_data['confidence'],
                'spectrum': self._get_spectrum_from_signal_type(ap.signal_type),
                # "lat/lon" remain the general centroid for map placement.
                'lat': stats.get('centroid_lat', 0),
                'lon': stats.get('centroid_lon', 0),
                # Estimated physical location (simple RSSI-weighted centroid).
                'est_lat': stats.get('weighted_centroid_lat', 0),
                'est_lon': stats.get('weighted_centroid_lon', 0),
                # Observed radius (meters) from estimated centroid to farthest GPS detection.
                'observed_radius_m': stats.get('observed_radius_m', 0),
                'strong_lat': stats.get('strongest_lat', 0),
                'strong_lon': stats.get('strongest_lon', 0),
                'strong_rssi': stats.get('strongest_rssi', 0),
                'strong_radius_m': stats.get('strongest_radius_m', 0),
                # Smallest/high-confidence region around strongest point
                'strong_region_radius_m': stats.get('strongest_region_radius_m', 0),
                'strong_region_threshold_db': stats.get('strongest_region_threshold_db', 6.0),
                # First/last boundary ring
                'first_seen_lat': stats.get('first_seen_lat', 0),
                'first_seen_lon': stats.get('first_seen_lon', 0),
                'first_seen_rssi': stats.get('first_seen_rssi', 0),
                'first_seen_ts': stats.get('first_seen_ts', ''),
                'last_seen_lat': stats.get('last_seen_lat', 0),
                'last_seen_lon': stats.get('last_seen_lon', 0),
                'last_seen_rssi': stats.get('last_seen_rssi', 0),
                'last_seen_ts': stats.get('last_seen_ts', ''),
                'first_last_radius_m': stats.get('first_last_radius_m', 0),
                # RSSI-weighted ellipse ("most likely" area)
                'ellipse_a_m': stats.get('ellipse_a_m', 0),
                'ellipse_b_m': stats.get('ellipse_b_m', 0),
                'ellipse_angle_deg': stats.get('ellipse_angle_deg', 0),
                'ellipse_center_lat': stats.get('ellipse_center_lat', 0),
                'ellipse_center_lon': stats.get('ellipse_center_lon', 0),
                'ellipse_sigma': stats.get('ellipse_sigma', 2.0),
                'has_gps': classification_data.get('has_gps', False),
                'detections': stats['detection_count'],
                'rssi_min': stats['rssi_min'],
                'rssi_max': stats['rssi_max'],
                'rssi_mean': stats['rssi_mean'],
                'rssi_std_dev': stats['rssi_std'],
                'coverage': int(abs(stats['rssi_range']) * 4),
                'runs': stats['unique_runs'],
                'channels': stats['channels'],
                'security': stats.get('primary_security', 'UNKNOWN'),
                'security_modes': stats.get('security_modes', []),
                'last_seen': ap.last_seen,
                'first_seen': ap.first_seen,
                'geographic_spread_km': stats.get('geographic_spread_km', 0)
            }
            results.append(result)
        
        return results
    
    def get_all_signals(self) -> List[Dict]:
        """Get all signals including those without classification"""
        results = []
        
        for mac, ap in self.access_points.items():
            stats = ap.get_statistics()
            
            result = {
                'mac': mac,
                'name': ap.name,
                'type': ap.signal_type,
                'device_category': self._categorize_device(ap.signal_type),
                'workers': list(ap.workers),
                'detections': len(ap.detections),
                'rssi_mean': stats.get('rssi_mean', -100),
                'rssi_min': stats.get('rssi_min', -100),
                'rssi_max': stats.get('rssi_max', -100),
                'has_gps': len(ap.locations) > 0,
                'channels': list(ap.channels),
                'security': stats.get('primary_security', 'UNKNOWN'),
                'security_modes': stats.get('security_modes', [])
            }
            results.append(result)
        
        return results

    def get_flat_detections(self, include_no_gps: bool = False) -> List[Dict]:
        """Return a flat list of detections suitable for UI playback/replay.

        Note: This can be large. The UI should request it only when needed.
        """
        out: List[Dict] = []

        def _parse_ts_ms(ts: str) -> Optional[int]:
            """Best-effort timestamp parser.

            Supports ISO8601, common human formats, and numeric epoch seconds/ms.
            """
            if not ts:
                return None
            s = str(ts).strip()
            if not s:
                return None

            # Numeric epoch?
            try:
                if s.replace('.', '', 1).isdigit():
                    v = float(s)
                    # Heuristic: >= 1e12 is already milliseconds
                    if v >= 1e12:
                        return int(v)
                    # Otherwise treat as seconds
                    if v >= 1e9:
                        return int(v * 1000)
                    # Small values are ambiguous; ignore
            except Exception:
                pass

            # ISO8601 variants
            try:
                t = s.replace('Z', '+00:00').replace(' UTC', '+00:00')
                # Allow space instead of 'T'
                if ' ' in t and 'T' not in t:
                    t = t.replace(' ', 'T', 1)
                dt = datetime.fromisoformat(t)
                return int(dt.timestamp() * 1000)
            except Exception:
                pass

            # Common strptime patterns
            patterns = [
                '%Y-%m-%d %H:%M:%S',
                '%Y/%m/%d %H:%M:%S',
                '%m/%d/%Y %H:%M:%S',
                '%Y-%m-%dT%H:%M:%S',
                '%Y-%m-%d %H:%M:%S.%f',
                '%Y/%m/%d %H:%M:%S.%f',
                '%m/%d/%Y %H:%M:%S.%f',
                '%Y-%m-%dT%H:%M:%S.%f',
            ]
            for p in patterns:
                try:
                    dt = datetime.strptime(s, p)
                    return int(dt.timestamp() * 1000)
                except Exception:
                    continue

            return None

        last_ms: Optional[int] = None

        for mac, ap in self.access_points.items():
            for d in ap.detections:
                has_gps = bool(d.get('has_gps')) and d.get('lat', 0.0) != 0.0 and d.get('lon', 0.0) != 0.0
                if not include_no_gps and not has_gps:
                    continue

                ts = str(d.get('timestamp') or '')
                ts_ms = _parse_ts_ms(ts)
                if ts_ms is None:
                    # Fall back to monotonically increasing timestamps so playback still works.
                    if last_ms is None:
                        last_ms = 0
                    else:
                        last_ms += 1000
                    ts_ms = last_ms
                else:
                    last_ms = ts_ms
                out.append({
                    'mac': mac,
                    'name': ap.name,
                    'type': ap.signal_type,
                    'worker': d.get('worker') or 'unknown',
                    'rssi': int(d.get('rssi', -100)),
                    'lat': float(d.get('lat', 0.0)),
                    'lon': float(d.get('lon', 0.0)),
                    'timestamp': ts,
                    'ts_ms': ts_ms,
                    'has_gps': has_gps,
                    'channel': d.get('channel'),
                    'security': d.get('security') or ''
                })

        # Sort with best-effort timestamp
        out.sort(key=lambda x: (x['ts_ms'] is None, x['ts_ms'] or 0))
        return out
    
    @staticmethod
    def _categorize_device(signal_type: str) -> str:
        """Categorize device based on signal type"""
        signal_type = signal_type.lower()
        if 'wifi' in signal_type or 'wlan' in signal_type:
            return 'wifi'
        elif 'ble' in signal_type or 'bluetooth' in signal_type:
            return 'ble'
        elif 'thread' in signal_type or 'zigbee' in signal_type or '802.15.4' in signal_type:
            return 'thread'
        elif 'halow' in signal_type or 'sub-1ghz' in signal_type or '802.11ah' in signal_type:
            return 'halow'
        else:
            return 'unknown'
    
    @staticmethod
    def _get_spectrum_from_signal_type(signal_type: str) -> str:
        """Get spectrum band from signal type"""
        signal_type = signal_type.lower()
        if 'halow' in signal_type or 'sub-1ghz' in signal_type:
            return 'Sub-1GHz'
        elif '5ghz' in signal_type or '5g' in signal_type:
            return '5GHz'
        elif '6ghz' in signal_type or '6g' in signal_type:
            return '6GHz'
        elif 'thread' in signal_type or 'zigbee' in signal_type or 'ble' in signal_type:
            return '2.4GHz'
        else:
            return '2.4GHz'
    
    def export_geojson(self) -> Dict:
        """Export as GeoJSON FeatureCollection"""
        features = []
        
        for mac, classification_data in self.classifications.items():
            if not classification_data.get('has_gps', False):
                continue
                
            ap = self.access_points[mac]
            stats = classification_data['statistics']
            
            color_map = {
                'static': '#10b981',
                'mobile': '#f59e0b',
                'uncertain': '#ef4444'
            }
            
            feature = {
                'type': 'Feature',
                'geometry': {
                    'type': 'Point',
                    'coordinates': [stats['centroid_lon'], stats['centroid_lat']]
                },
                'properties': {
                    'mac': mac,
                    'name': ap.name,
                    'signal_type': ap.signal_type,
                    'device_category': self._categorize_device(ap.signal_type),
                    'security': ap._primary_security(),
                    'security_modes': list(ap.security_modes),
                    'channels': list(stats.get('channels', [])),
                    'classification': classification_data['classification'],
                    'confidence': classification_data['confidence'],
                    'detections': stats['detection_count'],
                    'rssi_mean': stats['rssi_mean'],
                    'rssi_std': stats['rssi_std'],
                    'coverage_m': int(abs(stats['rssi_range']) * 4),
                    'workers': list(ap.workers),
                    'marker-color': color_map[classification_data['classification']]
                }
            }
            features.append(feature)
        
        return {
            'type': 'FeatureCollection',
            'features': features
        }
    
    def export_kml(self) -> str:
        """Export as KML string"""
        kml_header = '''<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>RF Site Survey Analysis</name>
    <Style id="static">
      <IconStyle>
        <Icon><href>http://maps.google.com/mapfiles/ms/icons/green-dot.png</href></Icon>
      </IconStyle>
    </Style>
    <Style id="mobile">
      <IconStyle>
        <Icon><href>http://maps.google.com/mapfiles/ms/icons/orange-dot.png</href></Icon>
      </IconStyle>
    </Style>
    <Style id="uncertain">
      <IconStyle>
        <Icon><href>http://maps.google.com/mapfiles/ms/icons/red-dot.png</href></Icon>
      </IconStyle>
    </Style>
'''
        
        placemarks = []
        for mac, classification_data in self.classifications.items():
            if not classification_data.get('has_gps', False):
                continue
                
            ap = self.access_points[mac]
            stats = classification_data['statistics']
            classification = classification_data['classification']
            
            # Escape XML special characters
            name_escaped = (ap.name or 'Unknown').replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            
            placemark = f'''    <Placemark>
      <name>{name_escaped}</name>
      <description><![CDATA[
        MAC: {mac}
        Classification: {classification.upper()}
        Confidence: {classification_data['confidence']:.1%}
        Detections: {stats['detection_count']}
        RSSI: {stats['rssi_mean']:.1f} ± {stats['rssi_std']:.1f} dBm
        Workers: {', '.join(ap.workers)}
        Channels: {', '.join(stats['channels']) if stats['channels'] else 'N/A'}
        Security: {ap._primary_security()}
      ]]></description>
      <styleUrl>#{classification}</styleUrl>
      <Point>
        <coordinates>{stats['centroid_lon']},{stats['centroid_lat']},0</coordinates>
      </Point>
    </Placemark>
'''
            placemarks.append(placemark)
        
        kml_footer = '''  </Document>
</kml>'''
        
        return kml_header + '\n'.join(placemarks) + '\n' + kml_footer
    
    def export_csv(self) -> str:
        """Export as CSV string"""
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Header
        writer.writerow([
            'mac', 'name', 'signal_type', 'classification', 'confidence',
            'lat', 'lon', 'has_gps', 'detections', 'rssi_mean', 'rssi_std',
            'rssi_min', 'rssi_max', 'workers', 'channels', 'runs'
            , 'security', 'security_modes'
        ])
        
        # Data
        for mac, classification_data in self.classifications.items():
            ap = self.access_points[mac]
            stats = classification_data['statistics']
            
            writer.writerow([
                mac,
                ap.name,
                ap.signal_type,
                classification_data['classification'],
                f"{classification_data['confidence']:.3f}",
                stats.get('centroid_lat', 0),
                stats.get('centroid_lon', 0),
                classification_data.get('has_gps', False),
                stats['detection_count'],
                f"{stats['rssi_mean']:.1f}",
                f"{stats['rssi_std']:.2f}",
                stats['rssi_min'],
                stats['rssi_max'],
                '|'.join(ap.workers),
                '|'.join(stats['channels']),
                stats['unique_runs'],
                ap._primary_security(),
                '|'.join(sorted(ap.security_modes))
            ])
        
        return output.getvalue()
