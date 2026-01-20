#!/usr/bin/env python3
"""
Silver Streak Analyzer (SSA) - Flask Backend with Embedded Frontend
Multi-spectrum RF analysis with GNSS Link integration
Single-file backend serving both API and UI
"""

from flask import Flask, request, jsonify, render_template_string, send_file
from flask_cors import CORS
from flask_socketio import SocketIO, emit, join_room
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
import json
import io

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analysis_engine import AnalysisEngine
from mqtt_handler import MQTTHandler, WardrivingDataProcessor, LiveDataServer

# Import polymorphic data sources
try:
    from data_sources import CSVDataSource, CSVFolderSource, MultiSource, Detection
    DATA_SOURCES_AVAILABLE = True
except ImportError:
    DATA_SOURCES_AVAILABLE = False

# Import PDF report generator
try:
    from pdf_report import PDFReportGenerator, generate_pdf_report
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# Import route map generator
try:
    from route_map import build_route_overview_png
except Exception:
    build_route_overview_png = None

# Import GNSS Link modules
try:
    from gnss_models import GNSSSession, GNSSPosition, ReceiverConfig, ProtocolType, FixType
    from gnss_csv_import import GNSSCSVImporter, create_option_b_csv
    from gnss_variance_analysis import GNSSVarianceAnalyzer, analyze_session
    from gnss_report import generate_gnss_report, ReportConfig as GNSSReportConfig
    GNSS_LINK_AVAILABLE = True
except ImportError as e:
    GNSS_LINK_AVAILABLE = False
    logger = logging.getLogger(__name__)
    # Will be logged after logger is configured

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ===================== Version =====================
APP_VERSION = "4.6.2"
APP_NAME = "Silver Streak Analyzer"
APP_SHORT_NAME = "SSA"
BUILD_TIMESTAMP_UTC = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

# ===================== Persistent Settings / Profiles =====================
BASE_DIR = Path(__file__).resolve().parent
USER_DATA_DIR = BASE_DIR / "user_data"
PROFILES_DIR = USER_DATA_DIR / "profiles"
SETTINGS_PATH = USER_DATA_DIR / "settings.json"
DEFAULT_PROFILES_DIR = BASE_DIR / "default_profiles"

DEFAULT_SETTINGS = {
    "active_profile_id": "v42_default",
    "report": {
        "format": "pdf",
        # Web report settings
        "html_font_family": "system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, sans-serif",
        "html_font_size": 14,
        "html_title_bold": True,
        # PDF report settings (limited to built-in reportlab fonts)
        "pdf_font_family": "Helvetica",
        "pdf_body_font_size": 10,
        "pdf_title_font_size": 24,
        "company_name": "",
        "company_font_family": "Helvetica",
        "company_font_size": 12,
        "company_bold": False,
        "watermark_enabled": False,
        "watermark_opacity": 0.08,
        "watermark_path": "",

        # Report charts (shown near top of reports)
        "charts_pdf_enabled": True,
        "charts_html_enabled": True,
        "chart_signal_types": True,
        "chart_classification": True,
        "chart_channels": True,
        "chart_top_n_channels": 6,

        # Route overview map image
        "route_map_enabled": True,
        "route_map_use_basemap": True,
        "route_map_padding_pct": 0.10

    }
}


def _ensure_user_dirs():
    USER_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)


def _load_settings() -> dict:
    _ensure_user_dirs()
    if SETTINGS_PATH.exists():
        try:
            data = json.loads(SETTINGS_PATH.read_text(encoding='utf-8'))
            # Merge defaults
            merged = json.loads(json.dumps(DEFAULT_SETTINGS))
            def _deep_update(a, b):
                for k, v in b.items():
                    if isinstance(v, dict) and isinstance(a.get(k), dict):
                        _deep_update(a[k], v)
                    else:
                        a[k] = v
            if isinstance(data, dict):
                _deep_update(merged, data)
            return merged
        except Exception:
            pass
    return json.loads(json.dumps(DEFAULT_SETTINGS))


def _save_settings(settings: dict) -> None:
    _ensure_user_dirs()
    try:
        SETTINGS_PATH.write_text(json.dumps(settings, indent=2), encoding='utf-8')
    except Exception as e:
        logger.warning(f"Failed to save settings: {e}")


def _bootstrap_default_profiles() -> None:
    """Copy built-in profile JSON files into the writable profiles directory if missing."""
    _ensure_user_dirs()
    try:
        existing = {p.name for p in PROFILES_DIR.glob('*.json')}
        if DEFAULT_PROFILES_DIR.exists():
            for src in DEFAULT_PROFILES_DIR.glob('*.json'):
                if src.name not in existing:
                    (PROFILES_DIR / src.name).write_text(src.read_text(encoding='utf-8'), encoding='utf-8')
    except Exception as e:
        logger.warning(f"Failed to bootstrap default profiles: {e}")


def _load_profiles() -> dict:
    """Return {profile_id: profile_dict}."""
    _bootstrap_default_profiles()
    profiles = {}
    for f in sorted(PROFILES_DIR.glob('*.json')):
        try:
            obj = json.loads(f.read_text(encoding='utf-8'))
            if not isinstance(obj, dict):
                continue
            pid = str(obj.get('id') or f.stem)
            name = str(obj.get('name') or pid)
            mapping = obj.get('mapping')
            if not isinstance(mapping, dict):
                continue
            profiles[pid] = {
                "id": pid,
                "name": name,
                "description": str(obj.get('description') or ''),
                "mapping": mapping,
                "filename": f.name
            }
        except Exception:
            continue
    return profiles


settings_store = _load_settings()
profiles_store = _load_profiles()


def _get_active_profile() -> dict:
    pid = settings_store.get('active_profile_id') or 'v42_default'
    return profiles_store.get(pid) or profiles_store.get('v42_default') or next(iter(profiles_store.values()), None)

# Initialize Flask app
app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Initialize analysis engine
analysis_engine = AnalysisEngine(
    dbscan_eps_km=0.1,
    dbscan_min_samples=3,
    rssi_variance_threshold=20.0,
    geographic_spread_threshold=0.5
)

mqtt_handler = None
data_processor = None
live_server = None
connected_clients = set()


def initialize_mqtt(config: dict):
    """Initialize MQTT connection"""
    global mqtt_handler, data_processor, live_server
    
    try:
        mqtt_handler = MQTTHandler(
            broker_host=config.get('host', 'localhost'),
            broker_port=config.get('port', 1883),
            username=config.get('username'),
            password=config.get('password')
        )
        
        data_processor = WardrivingDataProcessor(analysis_engine)
        mqtt_handler.set_data_callback(process_mqtt_data)
        mqtt_handler.subscribe("wardriving/+/detection")
        mqtt_handler.subscribe("wardriving/+/batch")
        
        if mqtt_handler.connect():
            logger.info("MQTT handler initialized")
            live_server = LiveDataServer(mqtt_handler, analysis_engine)
            live_server.start_broadcasting()
            return True
        else:
            logger.warning("Failed to connect MQTT - continuing without live data")
            return False
            
    except Exception as e:
        logger.warning(f"MQTT initialization failed (non-critical): {e}")
        return False


def process_mqtt_data(topic: str, data: dict):
    """Process MQTT data and broadcast to WebSocket clients"""
    try:
        if 'batch' in topic:
            data_processor.process_batch(topic, data)
        else:
            data_processor.process_detection(topic, data)
        
        if connected_clients:
            socketio.emit('data_update', {
                'timestamp': datetime.now().isoformat(),
                'topic': topic,
                'data': data
            }, room='wardriving_room')
            
    except Exception as e:
        logger.error(f"Error processing MQTT data: {e}")


# ============= REST API Endpoints =============

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'mqtt_connected': mqtt_handler.connected if mqtt_handler else False,
        'aps_loaded': len(analysis_engine.access_points),
        'aps_with_gps': len([ap for ap in analysis_engine.access_points.values() if ap.locations]),
        'timestamp': datetime.now().isoformat()
    })




@app.route('/api/version', methods=['GET'])
def api_version():
    return jsonify({"version": APP_VERSION, "build_utc": BUILD_TIMESTAMP_UTC})


@app.route('/api/settings', methods=['GET'])
def get_settings():
    """Return persisted UI + ingestion/report settings."""
    try:
        s = settings_store
        # Indicate watermark availability
        rep = s.get('report', {})
        rep = {**rep}
        rep['watermark_present'] = bool(rep.get('watermark_path') and Path(rep.get('watermark_path')).exists())
        return jsonify({
            'active_profile_id': s.get('active_profile_id'),
            'report': rep,
            'profiles': [
                {
                    'id': pr['id'],
                    'name': pr['name'],
                    'description': pr.get('description', '')
                }
                for pr in profiles_store.values()
            ]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/settings', methods=['POST'])
def update_settings():
    """Update persisted settings. Accepts partial JSON."""
    try:
        payload = request.get_json(silent=True) or {}
        if not isinstance(payload, dict):
            return jsonify({'error': 'Expected JSON object'}), 400

        def deep_update(dst, src):
            for k, v in src.items():
                if isinstance(v, dict) and isinstance(dst.get(k), dict):
                    deep_update(dst[k], v)
                else:
                    dst[k] = v

        deep_update(settings_store, payload)

        # Validate active profile
        pid = settings_store.get('active_profile_id')
        if pid and pid not in profiles_store:
            settings_store['active_profile_id'] = 'v42_default'

        _save_settings(settings_store)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/profiles', methods=['GET'])
def list_profiles():
    """List known CSV ingestion profiles."""
    try:
        return jsonify({
            'active_profile_id': settings_store.get('active_profile_id'),
            'profiles': [
                {
                    'id': pr['id'],
                    'name': pr['name'],
                    'description': pr.get('description', '')
                }
                for pr in profiles_store.values()
            ]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/profiles/upload', methods=['POST'])
def upload_profile():
    """Upload a profile JSON file and persist it on the server."""
    global profiles_store
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        f = request.files['file']
        raw = f.read().decode('utf-8', errors='replace')
        obj = json.loads(raw)
        if not isinstance(obj, dict):
            return jsonify({'error': 'Profile JSON must be an object'}), 400
        pid = str(obj.get('id') or '').strip()
        name = str(obj.get('name') or '').strip()
        mapping = obj.get('mapping')
        if not pid:
            return jsonify({'error': 'Profile missing required field: id'}), 400
        if not name:
            name = pid
        if not isinstance(mapping, dict) or not mapping:
            return jsonify({'error': 'Profile missing required field: mapping'}), 400

        safe_name = ''.join(c for c in pid if c.isalnum() or c in ('_', '-', '.'))
        if not safe_name:
            safe_name = f"profile_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        out_path = PROFILES_DIR / f"{safe_name}.json"
        out_path.write_text(json.dumps({
            'id': pid,
            'name': name,
            'description': str(obj.get('description') or ''),
            'mapping': mapping,
            'security_parts': obj.get('security_parts')
        }, indent=2), encoding='utf-8')

        profiles_store = _load_profiles()
        return jsonify({
            'success': True,
            'profiles': [
                {'id': pr['id'], 'name': pr['name'], 'description': pr.get('description', '')}
                for pr in profiles_store.values()
            ]
        })
    except json.JSONDecodeError as e:
        return jsonify({'error': f'Invalid JSON: {e}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/report/watermark', methods=['POST'])
def upload_watermark():
    """Upload a watermark logo for reports and persist it."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        f = request.files['file']
        if not f.filename:
            return jsonify({'error': 'No file provided'}), 400
        ext = os.path.splitext(f.filename)[1].lower()
        if ext not in ['.png', '.jpg', '.jpeg', '.webp']:
            return jsonify({'error': 'Unsupported image type. Use PNG/JPG/WEBP.'}), 400
        _ensure_user_dirs()
        out_path = USER_DATA_DIR / f"report_watermark{ext}"
        f.save(str(out_path))
        settings_store.setdefault('report', {})['watermark_path'] = str(out_path)
        settings_store.setdefault('report', {})['watermark_enabled'] = True
        _save_settings(settings_store)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
@app.route('/api/analysis/load-csv', methods=['POST'])
def load_csv():
    """Load CSV file and perform analysis"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        run_id = request.form.get('run_id', datetime.now().isoformat())
        profile_id = request.form.get('profile_id') or settings_store.get('active_profile_id')
        profile = profiles_store.get(profile_id) if profile_id else _get_active_profile()
        
        # Save file temporarily
        temp_path = f"/tmp/wardriving_{datetime.now().timestamp()}.csv"
        file.save(temp_path)
        
        # Ingest data
        result = analysis_engine.load_csv(temp_path, run_id, profile)
        
        # Classify
        classifications = analysis_engine.classify_access_points()
        
        # Get stats
        stats = analysis_engine.get_summary_stats()
        
        # Clean up
        os.remove(temp_path)
        
        # Broadcast update
        socketio.emit('analysis_complete', {
            'ingested': result['total_rows'],
            'classifications': len(classifications),
            'stats': stats,
            'timestamp': datetime.now().isoformat()
        }, room='wardriving_room')
        
        return jsonify({
            'success': True,
            'result': result,
            'classifications': len(classifications),
            'stats': stats
        })
        
    except Exception as e:
        logger.error(f"Error loading CSV: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/analysis/classify', methods=['POST'])
def classify():
    """Trigger classification analysis"""
    try:
        classifications = analysis_engine.classify_access_points()
        stats = analysis_engine.get_summary_stats()
        
        socketio.emit('classification_complete', {
            'classifications': len(classifications),
            'stats': stats,
            'timestamp': datetime.now().isoformat()
        }, room='wardriving_room')
        
        return jsonify({
            'success': True,
            'classifications': len(classifications),
            'stats': stats
        })
        
    except Exception as e:
        logger.error(f"Error classifying: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/data/summary', methods=['GET'])
def get_summary():
    """Get summary statistics"""
    try:
        stats = analysis_engine.get_summary_stats()
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/data/access-points', methods=['GET'])
def get_access_points():
    """Get filtered access points"""
    try:
        classification = request.args.get('classification', 'all')
        worker = request.args.get('worker', 'all')
        device_type = request.args.get('device_type', 'all')
        include_no_gps = request.args.get('include_no_gps', 'true').lower() == 'true'
        
        if classification == 'all':
            classification = None
        if worker == 'all':
            worker = None
        if device_type == 'all':
            device_type = None
        
        results = analysis_engine.get_filtered_results(
            classification=classification,
            worker=worker,
            device_type=device_type,
            include_no_gps=include_no_gps
        )
        
        return jsonify({
            'count': len(results),
            'data': results
        })
        
    except Exception as e:
        logger.error(f"Error getting APs: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/data/signals', methods=['GET'])
def get_signals():
    """Get all signals including those without GPS (for table view)"""
    try:
        signals = analysis_engine.get_all_signals()
        return jsonify({
            'count': len(signals),
            'data': signals
        })
    except Exception as e:
        logger.error(f"Error getting signals: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/data/detections', methods=['GET'])
def get_detections():
    """Get flat detections for playback/replay."""
    try:
        include_no_gps = request.args.get('include_no_gps', 'false').lower() == 'true'
        detections = analysis_engine.get_flat_detections(include_no_gps=include_no_gps)
        return jsonify({'count': len(detections), 'data': detections})
    except Exception as e:
        logger.error(f"Error getting detections: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/export/geojson', methods=['GET'])
def export_geojson():
    """Export analysis as GeoJSON"""
    try:
        geojson = analysis_engine.export_geojson()
        return jsonify(geojson)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/export/kml', methods=['GET'])
def export_kml():
    """Export analysis as KML"""
    try:
        kml = analysis_engine.export_kml()
        return kml, 200, {'Content-Type': 'application/vnd.google-earth.kml+xml'}
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/export/csv', methods=['GET'])
def export_csv():
    """Export analysis as CSV"""
    try:
        csv_data = analysis_engine.export_csv()
        return csv_data, 200, {
            'Content-Type': 'text/csv',
            'Content-Disposition': f'attachment; filename=wardriving_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        }
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/report/html', methods=['POST'])
def export_report_html():
    """Generate a human-readable HTML report for one or more selected SOIs."""
    try:
        payload = request.get_json(silent=True) or {}
        ids = payload.get('ids') or []
        if not isinstance(ids, list) or not ids:
            return jsonify({'error': 'Expected JSON body with non-empty "ids" list'}), 400

        # Ensure we have classifications/stats.
        if not analysis_engine.classifications:
            analysis_engine.classify_access_points()

        requested = [str(x).lower() for x in ids if str(x).strip()]
        now = datetime.now()

        def fmt_coord(lat, lon):
            try:
                return f"{float(lat):.6f}, {float(lon):.6f}"
            except Exception:
                return "-"

        def fmt_meters(m):
            try:
                return f"{float(m):.0f} m"
            except Exception:
                return "-"

        rows = []
        for mac in requested:
            ap = analysis_engine.access_points.get(mac)
            if not ap:
                continue
            c = analysis_engine.classifications.get(mac, {})
            stats = ap.get_statistics()
            rows.append({
                'mac': mac,
                'name': ap.name,
                'type': ap.signal_type,
                'device_category': analysis_engine._categorize_device(ap.signal_type),
                'classification': c.get('classification', 'unknown'),
                'confidence': float(c.get('confidence', 0.0) or 0.0),
                'first_seen': ap.first_seen,
                'last_seen': ap.last_seen,
                'workers': sorted(list(ap.workers)),
                'channels': sorted(list(ap.channels)),
                'detections': int(stats.get('detection_count', len(ap.detections))),
                'rssi_mean': stats.get('rssi_mean', None),
                'rssi_min': stats.get('rssi_min', None),
                'rssi_max': stats.get('rssi_max', None),
                'rssi_std': stats.get('rssi_std', None),
                'security': stats.get('primary_security', 'UNKNOWN'),
                'security_modes': stats.get('security_modes', []),
                'has_gps': bool(stats.get('has_gps')),
                'centroid': fmt_coord(stats.get('centroid_lat', 0), stats.get('centroid_lon', 0)),
                'est': fmt_coord(stats.get('weighted_centroid_lat', 0), stats.get('weighted_centroid_lon', 0)),
                'observed_radius_m': fmt_meters(stats.get('observed_radius_m', 0)),
                'strong': fmt_coord(stats.get('strongest_lat', 0), stats.get('strongest_lon', 0)),
                'strong_rssi': stats.get('strongest_rssi', None),
                'strong_radius_m': fmt_meters(stats.get('strongest_radius_m', 0)),
            })

        if not rows:
            return jsonify({'error': 'No matching SOIs found for requested ids'}), 404

        title = "Wardriving Analyzer Report"
        gen = now.strftime('%Y-%m-%d %H:%M:%S')

        # Apply persisted report options
        rep_cfg = settings_store.get('report', {}) or {}
        html_font_family = rep_cfg.get('html_font_family', 'system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, sans-serif')
        html_font_size = int(rep_cfg.get('html_font_size', 14) or 14)
        html_title_bold = bool(rep_cfg.get('html_title_bold', True))
        company_name = str(rep_cfg.get('company_name') or '').strip()
        company_font_family = str(rep_cfg.get('company_font_family') or 'Helvetica').strip()
        company_font_size = int(rep_cfg.get('company_font_size', 12) or 12)
        company_bold = bool(rep_cfg.get('company_bold', False))
        wm_enabled = bool(rep_cfg.get('watermark_enabled', False))
        try:
            wm_opacity = float(rep_cfg.get('watermark_opacity', 0.08) or 0.08)
        except Exception:
            wm_opacity = 0.08
        wm_data_uri = ''
        wm_path = rep_cfg.get('watermark_path')
        if wm_enabled and wm_path:
            try:
                pth = Path(wm_path)
                if pth.exists():
                    import base64
                    ext = pth.suffix.lower().lstrip('.') or 'png'
                    mime = 'image/png' if ext == 'png' else ('image/jpeg' if ext in ('jpg','jpeg') else ('image/webp' if ext == 'webp' else 'application/octet-stream'))
                    b64 = base64.b64encode(pth.read_bytes()).decode('ascii')
                    wm_data_uri = f"data:{mime};base64,{b64}"
            except Exception:
                wm_data_uri = ''
        html = [
            '<!DOCTYPE html>',
            '<html lang="en">',
            '<head>',
            '<meta charset="UTF-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1.0">',
            f'<title>{title}</title>',
            '<style>',
            f'body{{font-family:{html_font_family}; font-size:{html_font_size}px; margin:24px; color:#111; position:relative;}}',
            f'h1{{margin:0 0 8px 0; font-weight:{700 if html_title_bold else 400};}}',
            '.meta{color:#444; margin-bottom:18px;}',
            '.toc{background:#f5f7fb; border:1px solid #e3e7ef; padding:12px 14px; border-radius:10px; margin:18px 0;}',
            '.card{border:1px solid #e3e7ef; border-radius:12px; padding:16px; margin:16px 0;}',
            'table{border-collapse:collapse; width:100%; margin-top:10px;}',
            f'td,th{{border:1px solid #e3e7ef; padding:8px 10px; text-align:left; font-size:{max(11, html_font_size-1)}px;}}',
            'th{background:#f5f7fb;}',
            '.kv{display:grid; grid-template-columns: 180px 1fr; gap:6px 14px; font-size:14px;}',
            '.kv div{padding:2px 0;}',
            '.badge{display:inline-block; padding:2px 8px; border-radius:999px; font-size:12px; border:1px solid #e3e7ef; background:#f5f7fb;}',
            '.muted{color:#555;}',
            f'.company{{font-family:{company_font_family}; font-size:{company_font_size}px; font-weight:{700 if company_bold else 400}; color:#333; margin-bottom:8px;}}',
            (f"body::before{{content:''; position:fixed; inset:0; background-image:url({wm_data_uri}); background-repeat:no-repeat; background-position:center; background-size:70%; opacity:{wm_opacity}; pointer-events:none; z-index:-1;}}" if wm_data_uri else ''),
            '</style>',
            '</head>',
            '<body>',
            f'<h1>{title}</h1>',
            (f'<div class="company">{company_name}</div>' if company_name else ''),
            f'<div class="meta">Generated: <strong>{gen}</strong> • Items: <strong>{len(rows)}</strong></div>'
        ]

        # Optional route overview map
        if rep_cfg.get('route_map_enabled', True) is not False and build_route_overview_png:
            try:
                pts = analysis_engine.get_route_points(max_points=5000)
            except Exception:
                pts = []
            if pts and len(pts) >= 2:
                try:
                    import base64
                    padding = float(rep_cfg.get('route_map_padding_pct', 0.10) or 0.10)
                except Exception:
                    padding = 0.10
                use_basemap = rep_cfg.get('route_map_use_basemap', True) is not False
                try:
                    tile_cache = USER_DATA_DIR / 'tile_cache'
                    png = build_route_overview_png(
                        points=[(p['lat'], p['lon']) for p in pts],
                        padding_pct=padding,
                        use_basemap=use_basemap,
                        tile_cache_dir=str(tile_cache),
                        out_size=(1200, 700)
                    )
                    b64img = base64.b64encode(png).decode('ascii')
                    html.append('<div class="card" style="padding:12px;">'
                                '<div style="font-weight:600; margin-bottom:8px;">Survey Route Overview</div>'
                                '<div class="muted" style="font-size:12px; margin-bottom:10px;">Overall route extent with 10% padding for context.</div>'
                                + f'<img alt="Route overview" style="width:100%; border:1px solid #e3e7ef; border-radius:10px;" src="data:image/png;base64,{b64img}">'
                                + '</div>')
                except Exception:
                    pass

        # Optional overview charts
        if rep_cfg.get('charts_html_enabled', True) is not False:
            show_signal = rep_cfg.get('chart_signal_types', True) is not False
            show_class = rep_cfg.get('chart_classification', True) is not False
            show_channels = rep_cfg.get('chart_channels', True) is not False
            try:
                top_n = int(rep_cfg.get('chart_top_n_channels', 6) or 6)
            except Exception:
                top_n = 6
            top_n = max(3, min(12, top_n))

            # Aggregate counts for charts
            sig_counts = {}
            cls_counts = {'STATIC': 0, 'MOBILE': 0, 'UNCERTAIN': 0}
            chan_counts = {}
            for mac in requested:
                ap = analysis_engine.access_points.get(mac)
                if not ap:
                    continue
                if show_signal:
                    st = str(getattr(ap, 'signal_type', '') or 'UNKNOWN').strip().upper()
                    if st in ('BLE', 'BT', 'BLUETOOTH'):
                        st = 'BT/BLE'
                    sig_counts[st] = sig_counts.get(st, 0) + 1
                if show_class:
                    c = analysis_engine.classifications.get(mac, {}) or {}
                    cls = str(c.get('classification', 'uncertain') or 'uncertain').strip().upper()
                    if cls not in cls_counts:
                        cls = 'UNCERTAIN'
                    cls_counts[cls] = cls_counts.get(cls, 0) + 1
                if show_channels:
                    for d in getattr(ap, 'detections', []) or []:
                        ch = d.get('channel')
                        key = str(ch).strip() if ch is not None and str(ch).strip() else 'UNKNOWN'
                        chan_counts[key] = chan_counts.get(key, 0) + 1

            # Reduce channel list to top N + Other
            chan_items = sorted(chan_counts.items(), key=lambda kv: kv[1], reverse=True)
            top = chan_items[:top_n]
            other = sum(v for _, v in chan_items[top_n:])
            chan_labels = [k for k, _ in top] + (['OTHER'] if other > 0 else [])
            chan_values = [v for _, v in top] + ([other] if other > 0 else [])

            html.append('<div class="toc" style="margin-top:12px;">'
                        '<strong>Overview Charts</strong>'
                        '<div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap:14px; margin-top:12px;">')

            if show_signal:
                html.append('<div class="card" style="margin:0; padding:12px;">'
                            '<div style="font-weight:600; margin-bottom:8px;">Signal types</div>'
                            '<canvas id="chart-signal" width="220" height="160"></canvas>'
                            '</div>')
            if show_class:
                html.append('<div class="card" style="margin:0; padding:12px;">'
                            '<div style="font-weight:600; margin-bottom:8px;">Static vs Mobile</div>'
                            '<canvas id="chart-class" width="220" height="160"></canvas>'
                            '</div>')
            if show_channels:
                html.append('<div class="card" style="margin:0; padding:12px;">'
                            f'<div style="font-weight:600; margin-bottom:8px;">Most-detected channels (top {top_n})</div>'
                            '<canvas id="chart-channels" width="220" height="160"></canvas>'
                            '<div class="muted" style="margin-top:6px; font-size:12px;">Counts detections per channel; others are grouped under UNKNOWN.</div>'
                            '</div>')

            html.append('</div></div>')

            # Simple pie chart renderer (no external deps)
            def _js_array(lst):
                import json as _json
                return _json.dumps(list(lst))



        # TOC
        html.append('<div class="toc"><strong>Contents</strong><ul>')
        for r in rows:
            anchor = r['mac'].replace(':', '')
            disp = (r['name'] or '(no name)')
            html.append(f'<li><a href="#{anchor}">{disp}</a> <span class="muted">({r["mac"]})</span></li>')
        html.append('</ul></div>')

        for r in rows:
            anchor = r['mac'].replace(':', '')
            name = r['name'] or '(no name)'
            cls = str(r['classification'] or 'unknown').upper()
            conf = f"{r['confidence']*100:.1f}%" if r.get('confidence') is not None else '-'
            html.append(f'<div class="card" id="{anchor}">')
            html.append(f'<h2 style="margin:0 0 6px 0;">{name}</h2>')
            html.append(f'<div class="muted" style="font-family:monospace;">{r["mac"]}</div>')
            html.append(f'<div style="margin-top:10px; display:flex; gap:10px; flex-wrap:wrap;">'
                        f'<span class="badge">{cls}</span>'
                        f'<span class="badge">Confidence: {conf}</span>'
                        f'<span class="badge">Type: {r["device_category"]}</span>'
                        f'<span class="badge">Security: {r["security"]}</span>'
                        '</div>')

            html.append('<table>')
            html.append('<tr><th>Field</th><th>Value</th></tr>')
            html.append(f'<tr><td>Detections</td><td>{r["detections"]}</td></tr>')
            html.append(f'<tr><td>RSSI mean / min / max (dBm)</td><td>{r.get("rssi_mean")!s} / {r.get("rssi_min")!s} / {r.get("rssi_max")!s}</td></tr>')
            html.append(f'<tr><td>RSSI std dev</td><td>{r.get("rssi_std")!s}</td></tr>')
            html.append(f'<tr><td>Workers</td><td>{", ".join(r["workers"]) if r["workers"] else "-"}</td></tr>')
            html.append(f'<tr><td>Channels</td><td>{", ".join(r["channels"]) if r["channels"] else "-"}</td></tr>')
            html.append(f'<tr><td>Security modes observed</td><td>{", ".join(r["security_modes"]) if r["security_modes"] else "-"}</td></tr>')
            html.append(f'<tr><td>First seen</td><td>{r["first_seen"] or "-"}</td></tr>')
            html.append(f'<tr><td>Last seen</td><td>{r["last_seen"] or "-"}</td></tr>')
            if r['has_gps']:
                html.append(f'<tr><td>Strongest point heard</td><td>{r["strong"]} (RSSI {r.get("strong_rssi")!s} dBm)</td></tr>')
                html.append(f'<tr><td>Captured radius from strongest point</td><td>{r["strong_radius_m"]}</td></tr>')
                html.append(f'<tr><td>Likely physical location (RSSI-weighted)</td><td>{r["est"]}</td></tr>')
                html.append(f'<tr><td>Captured radius from estimated location</td><td>{r["observed_radius_m"]}</td></tr>')
                html.append(f'<tr><td>Map centroid</td><td>{r["centroid"]}</td></tr>')
            else:
                html.append('<tr><td>GPS</td><td>Not available (no valid fixes captured for this SOI)</td></tr>')
            html.append('</table>')
            html.append('</div>')

        html.append('</body></html>')
        out = "\n".join(html).encode('utf-8')

        fname = f"wardriving_report_{now.strftime('%Y%m%d_%H%M%S')}.html"
        return send_file(
            io.BytesIO(out),
            as_attachment=True,
            download_name=fname,
            mimetype='text/html'
        )

    except Exception as e:
        logger.error(f"Report generation error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/report/pdf', methods=['POST'])
def export_report_pdf():
    """Generate a comprehensive PDF report for selected SOIs or all data."""
    if not PDF_AVAILABLE:
        return jsonify({'error': 'PDF generation not available. Install reportlab: pip install reportlab'}), 501
    
    try:
        payload = request.get_json(silent=True) or {}
        title = payload.get('title', 'RF Site Survey Analysis Report')
        include_details = payload.get('include_details', True)
        ids = payload.get('ids')  # Optional list of SOI IDs to include
        
        # Ensure we have classifications
        if not analysis_engine.classifications:
            analysis_engine.classify_access_points()
        
        # Generate PDF with optional ID filtering
        generator = PDFReportGenerator(analysis_engine, settings_store.get('report', {}) or {})
        pdf_bytes = generator.generate(title=title, include_details=include_details, filter_ids=ids)
        
        fname = f"wardriving_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        return send_file(
            io.BytesIO(pdf_bytes),
            as_attachment=True,
            download_name=fname,
            mimetype='application/pdf'
        )
        
    except Exception as e:
        logger.error(f"PDF report generation error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/analysis/load-multi', methods=['POST'])
def load_multi_csv():
    """Load multiple CSV files at once for multi-day/multi-run analysis."""
    try:
        if 'files' not in request.files and 'file' not in request.files:
            return jsonify({'error': 'No files provided'}), 400
        
        # Handle both single and multiple file uploads
        files = request.files.getlist('files') or request.files.getlist('file')
        if not files:
            return jsonify({'error': 'No files provided'}), 400
        
        results = []
        total_rows = 0
        total_with_gps = 0
        total_without_gps = 0

        profile_id = request.form.get('profile_id') or settings_store.get('active_profile_id')
        profile = profiles_store.get(profile_id) if profile_id else _get_active_profile()

        for file in files:
            if not file.filename:
                continue
                
            # Generate run_id from filename
            run_id = f"run_{Path(file.filename).stem}_{datetime.now().strftime('%H%M%S')}"
            
            # Save file temporarily
            temp_path = f"/tmp/wardriving_{datetime.now().timestamp()}_{file.filename}"
            file.save(temp_path)
            
            try:
                # Ingest data
                result = analysis_engine.load_csv(temp_path, run_id, profile)
                results.append({
                    'filename': file.filename,
                    'run_id': run_id,
                    'result': result
                })
                
                total_rows += result.get('total_rows', 0)
                total_with_gps += result.get('with_gps', 0)
                total_without_gps += result.get('without_gps', 0)
                
            finally:
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        
        # Classify after all files loaded
        classifications = analysis_engine.classify_access_points()
        stats = analysis_engine.get_summary_stats()
        
        # Broadcast update
        socketio.emit('analysis_complete', {
            'files_loaded': len(results),
            'total_ingested': total_rows,
            'classifications': len(classifications),
            'stats': stats,
            'timestamp': datetime.now().isoformat()
        }, room='wardriving_room')
        
        return jsonify({
            'success': True,
            'files_processed': len(results),
            'results': results,
            'totals': {
                'rows': total_rows,
                'with_gps': total_with_gps,
                'without_gps': total_without_gps,
                'unique_aps': stats.get('total_aps', 0)
            },
            'stats': stats
        })
        
    except Exception as e:
        logger.error(f"Error loading multiple CSVs: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/data/runs', methods=['GET'])
def get_loaded_runs():
    """Get information about loaded runs/sessions."""
    try:
        # Collect run information from access points
        runs = {}
        for mac, ap in analysis_engine.access_points.items():
            for run_id in ap.run_ids:
                if run_id not in runs:
                    runs[run_id] = {
                        'run_id': run_id,
                        'aps': set(),
                        'detections': 0,
                        'with_gps': 0
                    }
                runs[run_id]['aps'].add(mac)
                runs[run_id]['detections'] += len([d for d in ap.detections if d.get('timestamp', '').startswith(run_id[:10]) or True])
        
        # Convert sets to counts
        run_list = []
        for run_id, data in runs.items():
            run_list.append({
                'run_id': run_id,
                'unique_aps': len(data['aps']),
                'detections': data['detections']
            })
        
        return jsonify({
            'runs': sorted(run_list, key=lambda x: x['run_id']),
            'total_runs': len(run_list),
            'total_aps': len(analysis_engine.access_points)
        })
        
    except Exception as e:
        logger.error(f"Error getting runs: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/mqtt/status', methods=['GET'])
def mqtt_status():
    """Get MQTT connection status"""
    if mqtt_handler:
        return jsonify({
            'connected': mqtt_handler.connected,
            'broker': f"{mqtt_handler.broker_host}:{mqtt_handler.broker_port}",
            'subscribed_topics': mqtt_handler.topic_subscriptions,
            'data_points': len(analysis_engine.access_points)
        })
    else:
        return jsonify({
            'connected': False,
            'error': 'MQTT not initialized',
            'data_points': len(analysis_engine.access_points)
        })


@app.route('/api/clear', methods=['POST'])
def clear_data():
    """Clear all loaded data"""
    try:
        analysis_engine.clear()
        socketio.emit('data_cleared', {
            'timestamp': datetime.now().isoformat()
        }, room='wardriving_room')
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/analysis/remove-runs', methods=['POST'])
def remove_runs():
    """Remove specific runs from loaded data"""
    try:
        payload = request.get_json(silent=True) or {}
        run_ids_to_remove = payload.get('run_ids', [])
        
        if not run_ids_to_remove:
            return jsonify({'error': 'No run_ids provided'}), 400
        
        run_ids_set = set(run_ids_to_remove)
        removed_count = 0
        aps_affected = 0
        aps_removed = 0
        
        # Iterate through all APs and remove detections from specified runs
        macs_to_remove = []
        
        for mac, ap in analysis_engine.access_points.items():
            # Filter out detections from runs being removed
            original_count = len(ap.detections)
            ap.detections = [d for d in ap.detections if d.get('run_id') not in run_ids_set]
            
            # Also filter locations (only keep those with detections that remain)
            # Rebuild locations from remaining detections
            ap.locations = []
            ap.rssi_values = []
            for d in ap.detections:
                ap.rssi_values.append(d.get('rssi', -100))
                if d.get('has_gps') and d.get('lat', 0) != 0 and d.get('lon', 0) != 0:
                    ap.locations.append((d.get('lat'), d.get('lon')))
            
            # Remove run_ids that were deleted
            ap.run_ids = ap.run_ids - run_ids_set
            
            if len(ap.detections) < original_count:
                aps_affected += 1
                removed_count += (original_count - len(ap.detections))
            
            # Mark AP for removal if no detections left
            if len(ap.detections) == 0:
                macs_to_remove.append(mac)
        
        # Remove empty APs
        for mac in macs_to_remove:
            del analysis_engine.access_points[mac]
            aps_removed += 1
        
        # Re-classify remaining data
        if analysis_engine.access_points:
            analysis_engine.classify_access_points()
        else:
            analysis_engine.classifications = {}
        
        # Get new stats
        stats = analysis_engine.get_summary_stats() if analysis_engine.access_points else {}
        
        # Broadcast update
        socketio.emit('runs_removed', {
            'removed_runs': run_ids_to_remove,
            'detections_removed': removed_count,
            'aps_affected': aps_affected,
            'aps_removed': aps_removed,
            'stats': stats,
            'timestamp': datetime.now().isoformat()
        }, room='wardriving_room')
        
        return jsonify({
            'success': True,
            'removed_runs': run_ids_to_remove,
            'detections_removed': removed_count,
            'aps_affected': aps_affected,
            'aps_removed': aps_removed,
            'remaining_aps': len(analysis_engine.access_points),
            'stats': stats
        })
        
    except Exception as e:
        logger.error(f"Error removing runs: {e}")
        return jsonify({'error': str(e)}), 500


# ============= GNSS Link State =============
gnss_session = None
gnss_metrics = None


# ============= GNSS Link API Endpoints =============

@app.route('/api/gnss/status', methods=['GET'])
def gnss_status():
    """Get GNSS Link module status"""
    global gnss_session, gnss_metrics
    
    return jsonify({
        'available': GNSS_LINK_AVAILABLE,
        'session_active': gnss_session is not None,
        'session_id': gnss_session.session_id if gnss_session else None,
        'receivers': len(gnss_session.receivers) if gnss_session else 0,
        'positions': len(gnss_session.positions) if gnss_session else 0,
        'has_metrics': gnss_metrics is not None
    })


@app.route('/api/gnss/load-csv', methods=['POST'])
def gnss_load_csv():
    """Load GNSS CSV files into a session"""
    global gnss_session, gnss_metrics
    
    if not GNSS_LINK_AVAILABLE:
        return jsonify({'error': 'GNSS Link module not available'}), 500
    
    try:
        if 'files' not in request.files:
            return jsonify({'error': 'No files provided'}), 400
        
        files = request.files.getlist('files')
        if not files:
            return jsonify({'error': 'No files provided'}), 400
        
        # Get session parameters
        session_id = request.form.get('session_id', f'gnss_session_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        username = request.form.get('username', 'analyst')
        
        # Create new session
        gnss_session = GNSSSession(
            session_id=session_id,
            username=username,
            role='analyst'
        )
        
        importer = GNSSCSVImporter()
        total_positions = 0
        
        for file in files:
            if file.filename:
                # Save temp file
                temp_path = Path(f'/tmp/gnss_upload_{file.filename}')
                file.save(str(temp_path))
                
                # Import
                try:
                    for pos in importer.import_file(temp_path):
                        gnss_session.positions.append(pos)
                        total_positions += 1
                finally:
                    temp_path.unlink(missing_ok=True)
        
        # Copy detected receivers
        gnss_session.receivers = importer.detected_receivers.copy()
        
        # Set time bounds
        if gnss_session.positions:
            gnss_session.positions.sort(key=lambda p: p.timestamp)
            gnss_session.start_time = gnss_session.positions[0].timestamp
            gnss_session.end_time = gnss_session.positions[-1].timestamp
        
        # Run analysis
        analyzer = GNSSVarianceAnalyzer()
        gnss_metrics = analyzer.analyze(gnss_session)
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'files_loaded': len(files),
            'positions': total_positions,
            'receivers': list(gnss_session.receivers.keys()),
            'time_range': {
                'start': gnss_session.start_time.isoformat() if gnss_session.start_time else None,
                'end': gnss_session.end_time.isoformat() if gnss_session.end_time else None
            },
            'metrics': {
                'std_h': gnss_metrics.all_receivers_std_h,
                'std_v': gnss_metrics.all_receivers_std_v,
                'cep_50': gnss_metrics.all_receivers_cep_50,
                'drms_2': gnss_metrics.all_receivers_2drms,
                'max_deviation': gnss_metrics.all_receivers_max_deviation
            }
        })
        
    except Exception as e:
        logger.error(f"GNSS CSV load error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/gnss/receivers', methods=['GET'])
def gnss_get_receivers():
    """Get receiver list and statistics"""
    global gnss_session, gnss_metrics
    
    if not gnss_session:
        return jsonify({'error': 'No GNSS session active'}), 400
    
    receivers = []
    for rcvr_id, config in gnss_session.receivers.items():
        stats = gnss_metrics.receiver_stats.get(rcvr_id) if gnss_metrics else None
        receivers.append({
            'id': rcvr_id,
            'nickname': config.nickname,
            'protocol': config.protocol.value,
            'is_truth': config.is_truth_receiver,
            'icon_color': config.icon_color,
            'positions': stats.position_count if stats else 0,
            'stats': {
                'sat_count_mean': stats.sat_count_mean if stats else 0,
                'hdop_mean': stats.hdop_mean if stats else 0,
                'deviation_mean': stats.deviation_from_centroid_mean if stats else 0,
                'cep_50': stats.cep_50 if stats else 0,
                'drms_2': stats.drms_2 if stats else 0
            } if stats else None
        })
    
    return jsonify({'receivers': receivers})


@app.route('/api/gnss/receivers/<receiver_id>', methods=['PUT'])
def gnss_update_receiver(receiver_id):
    """Update receiver configuration"""
    global gnss_session
    
    if not gnss_session:
        return jsonify({'error': 'No GNSS session active'}), 400
    
    if receiver_id not in gnss_session.receivers:
        return jsonify({'error': 'Receiver not found'}), 404
    
    data = request.get_json()
    config = gnss_session.receivers[receiver_id]
    
    if 'nickname' in data:
        config.nickname = data['nickname']
    if 'is_truth' in data:
        config.is_truth_receiver = data['is_truth']
    if 'icon_color' in data:
        config.icon_color = data['icon_color']
    if 'enabled' in data:
        config.enabled = data['enabled']
    
    # Re-analyze if truth receiver changed
    if 'is_truth' in data:
        analyzer = GNSSVarianceAnalyzer()
        global gnss_metrics
        gnss_metrics = analyzer.analyze(gnss_session)
    
    return jsonify({'success': True, 'receiver': config.to_dict()})


@app.route('/api/gnss/positions', methods=['GET'])
def gnss_get_positions():
    """Get position data for mapping/charting"""
    global gnss_session
    
    if not gnss_session:
        return jsonify({'error': 'No GNSS session active'}), 400
    
    receiver_id = request.args.get('receiver_id')
    limit = int(request.args.get('limit', 10000))
    
    positions = gnss_session.positions
    if receiver_id:
        positions = [p for p in positions if p.receiver_id == receiver_id]
    
    # Limit for performance
    if len(positions) > limit:
        step = len(positions) // limit
        positions = positions[::step]
    
    return jsonify({
        'positions': [p.to_dict() for p in positions],
        'count': len(positions)
    })


@app.route('/api/gnss/deviations', methods=['GET'])
def gnss_get_deviations():
    """Get deviation time series data"""
    global gnss_session, gnss_metrics
    
    if not gnss_session or not gnss_metrics:
        return jsonify({'error': 'No GNSS session or metrics available'}), 400
    
    receiver_id = request.args.get('receiver_id')
    
    records = gnss_metrics.deviation_records
    if receiver_id:
        records = [r for r in records if r.receiver_id == receiver_id]
    
    return jsonify({
        'deviations': [
            {
                'timestamp': r.timestamp.isoformat(),
                'receiver_id': r.receiver_id,
                'deviation_h_m': r.deviation_h_m,
                'deviation_v_m': r.deviation_v_m,
                'lat': r.lat,
                'lon': r.lon
            }
            for r in records
        ],
        'count': len(records)
    })


@app.route('/api/gnss/metrics', methods=['GET'])
def gnss_get_metrics():
    """Get computed variance metrics"""
    global gnss_metrics
    
    if not gnss_metrics:
        return jsonify({'error': 'No GNSS metrics available'}), 400
    
    return jsonify({
        'all_receivers': {
            'std_h': gnss_metrics.all_receivers_std_h,
            'std_v': gnss_metrics.all_receivers_std_v,
            'cep_50': gnss_metrics.all_receivers_cep_50,
            'drms_2': gnss_metrics.all_receivers_2drms,
            'max_deviation': gnss_metrics.all_receivers_max_deviation
        },
        'truth_comparison': {
            'std_h': gnss_metrics.truth_comparison_std_h,
            'std_v': gnss_metrics.truth_comparison_std_v,
            'cep_50': gnss_metrics.truth_comparison_cep_50,
            'drms_2': gnss_metrics.truth_comparison_2drms,
            'max_deviation': gnss_metrics.truth_comparison_max_deviation,
            'truth_receivers': gnss_metrics.truth_receiver_ids
        } if gnss_metrics.truth_comparison_std_h is not None else None,
        'per_receiver': {
            rcvr_id: {
                'position_count': stats.position_count,
                'lat_std': stats.lat_std,
                'lon_std': stats.lon_std,
                'alt_std': stats.alt_std,
                'sat_count_mean': stats.sat_count_mean,
                'hdop_mean': stats.hdop_mean,
                'deviation_mean': stats.deviation_from_centroid_mean,
                'deviation_max': stats.deviation_from_centroid_max,
                'deviation_std': stats.deviation_from_centroid_std,
                'cep_50': stats.cep_50,
                'drms_2': stats.drms_2,
                'deviation_from_truth_mean': stats.deviation_from_truth_mean,
                'deviation_from_truth_max': stats.deviation_from_truth_max
            }
            for rcvr_id, stats in gnss_metrics.receiver_stats.items()
        }
    })


@app.route('/api/gnss/report', methods=['POST'])
def gnss_generate_report():
    """Generate GNSS Link PDF report"""
    global gnss_session, gnss_metrics
    
    if not gnss_session or not gnss_metrics:
        return jsonify({'error': 'No GNSS session or metrics available'}), 400
    
    if not GNSS_LINK_AVAILABLE:
        return jsonify({'error': 'GNSS Link module not available'}), 500
    
    try:
        data = request.get_json() or {}
        
        config = GNSSReportConfig(
            title=data.get('title', f'GNSS Link Report - {gnss_session.session_id}'),
            dark_mode=data.get('dark_mode', True),
            include_map_overview=data.get('include_map', True),
            include_track_overlay=data.get('include_tracks', True)
        )
        
        pdf_bytes = generate_gnss_report(gnss_session, gnss_metrics, config=config)
        
        return send_file(
            io.BytesIO(pdf_bytes),
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f'gnss_report_{gnss_session.session_id}.pdf'
        )
        
    except Exception as e:
        logger.error(f"GNSS report generation error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/gnss/export/csv', methods=['GET'])
def gnss_export_csv():
    """Export GNSS session to Option B CSV"""
    global gnss_session
    
    if not gnss_session:
        return jsonify({'error': 'No GNSS session active'}), 400
    
    try:
        output = io.StringIO()
        
        # Write header
        fieldnames = [
            'Timestamp', 'ReceiverID', 'Latitude', 'Longitude',
            'AltitudeMSL', 'AltitudeHAE', 'FixType', 'SatCount',
            'HDOP', 'VDOP', 'PDOP', 'AccuracyH_m', 'AccuracyV_m',
            'Speed_mps', 'Heading_deg', 'CarrierSolution'
        ]
        
        import csv
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        
        for pos in sorted(gnss_session.positions, key=lambda p: (p.timestamp, p.receiver_id)):
            writer.writerow({
                'Timestamp': pos.timestamp.isoformat(),
                'ReceiverID': pos.receiver_id,
                'Latitude': f"{pos.lat:.9f}",
                'Longitude': f"{pos.lon:.9f}",
                'AltitudeMSL': f"{pos.alt_msl:.3f}" if pos.alt_msl else '',
                'AltitudeHAE': f"{pos.alt_hae:.3f}" if pos.alt_hae else '',
                'FixType': pos.fix_type.value,
                'SatCount': pos.sat_count,
                'HDOP': f"{pos.hdop:.2f}" if pos.hdop else '',
                'VDOP': f"{pos.vdop:.2f}" if pos.vdop else '',
                'PDOP': f"{pos.pdop:.2f}" if pos.pdop else '',
                'AccuracyH_m': f"{pos.accuracy_h_m:.3f}" if pos.accuracy_h_m else '',
                'AccuracyV_m': f"{pos.accuracy_v_m:.3f}" if pos.accuracy_v_m else '',
                'Speed_mps': f"{pos.speed_mps:.3f}" if pos.speed_mps else '',
                'Heading_deg': f"{pos.heading_deg:.2f}" if pos.heading_deg else '',
                'CarrierSolution': pos.carrier_solution
            })
        
        output.seek(0)
        return send_file(
            io.BytesIO(output.getvalue().encode('utf-8')),
            mimetype='text/csv',
            as_attachment=True,
            download_name=f'gnss_export_{gnss_session.session_id}.csv'
        )
        
    except Exception as e:
        logger.error(f"GNSS CSV export error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/gnss/clear', methods=['POST'])
def gnss_clear():
    """Clear GNSS session"""
    global gnss_session, gnss_metrics
    
    gnss_session = None
    gnss_metrics = None
    
    return jsonify({'success': True})


# ============= WebSocket Events =============

@socketio.on('connect')
def handle_connect():
    """Handle WebSocket connection"""
    client_id = request.sid
    connected_clients.add(client_id)
    join_room('wardriving_room')
    logger.info(f"Client {client_id} connected. Total: {len(connected_clients)}")
    
    emit('connection_response', {
        'data': 'Connected to wardriving analyzer',
        'timestamp': datetime.now().isoformat(),
        'aps_loaded': len(analysis_engine.access_points)
    })


@socketio.on('disconnect')
def handle_disconnect():
    """Handle WebSocket disconnection"""
    client_id = request.sid
    connected_clients.discard(client_id)
    logger.info(f"Client {client_id} disconnected. Total: {len(connected_clients)}")


@socketio.on('request_stats')
def handle_stats_request():
    """Handle stats request from client"""
    try:
        stats = analysis_engine.get_summary_stats()
        emit('stats_update', {
            'stats': stats,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error sending stats: {e}")
        emit('error', {'message': str(e)})


@socketio.on('request_aps')
def handle_aps_request(data):
    """Handle AP data request from client"""
    try:
        classification = data.get('classification', 'all') if data else 'all'
        worker = data.get('worker', 'all') if data else 'all'
        device_type = data.get('deviceType', 'all') if data else 'all'
        include_no_gps = data.get('includeNoGps', True) if data else True
        
        if classification == 'all':
            classification = None
        if worker == 'all':
            worker = None
        if device_type == 'all':
            device_type = None
        
        aps = analysis_engine.get_filtered_results(
            classification=classification,
            worker=worker,
            device_type=device_type,
            include_no_gps=include_no_gps
        )
        
        emit('aps_data', {
            'aps': aps,
            'count': len(aps),
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error sending APs: {e}")
        emit('error', {'message': str(e)})


@socketio.on('request_detections')
def handle_detections_request(data):
    """Handle flat detections request from client (for playback)."""
    try:
        include_no_gps = bool(data.get('includeNoGps', False)) if data else False
        detections = analysis_engine.get_flat_detections(include_no_gps=include_no_gps)
        emit('detections_data', {
            'detections': detections,
            'count': len(detections),
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error sending detections: {e}")
        emit('error', {'message': str(e)})


# ============= Main UI =============

@app.route('/')
def index():
    """Serve embedded frontend"""
    return render_template_string(FRONTEND_HTML)


# Embedded frontend HTML
FRONTEND_HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Silver Streak Analyzer</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.6.1/socket.io.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        :root {
            --bg-primary: #0a0e14;
            --bg-secondary: #131920;
            --bg-tertiary: #1a2029;
            --text-primary: #e6edf3;
            --text-secondary: #8b949e;
            --accent-green: #10b981;
            --accent-orange: #f59e0b;
            --accent-red: #ef4444;
            --accent-blue: #3b82f6;
            --border-color: #30363d;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
        }
        
        .app-container {
            display: flex;
            flex-direction: column;
            height: 100vh;
        }
        
        /* Header */
        .header {
            background: var(--bg-secondary);
            border-bottom: 1px solid var(--border-color);
            padding: 12px 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .header-left {
            display: flex;
            align-items: center;
            gap: 12px;
        }
        
        .header h1 {
            font-size: 18px;
            font-weight: 600;
            color: var(--text-primary);
        }
        
        .header-badge {
            background: linear-gradient(135deg, var(--accent-green), var(--accent-blue));
            padding: 4px 10px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 600;
            letter-spacing: 0.5px;
        }
        
        .header-right {
            display: flex;
            align-items: center;
            gap: 16px;
            font-size: 13px;
            color: var(--text-secondary);
        }
        
        .status-indicator {
            display: flex;
            align-items: center;
            gap: 6px;
        }
        
        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--accent-red);
        }
        
        .status-dot.connected {
            background: var(--accent-green);
            box-shadow: 0 0 8px var(--accent-green);
        }
        
        /* Main Content */
        .main-content {
            display: flex;
            flex: 1;
            overflow: hidden;
        }

        /* Side Panels */
        .side-panel {
            background: var(--bg-secondary);
            display: flex;
            flex-direction: column;
            overflow: hidden;
            transition: width 0.15s ease;
        }

        .left-panel {
            width: 340px;
            border-right: 1px solid var(--border-color);
        }

        .right-panel {
            width: 360px;
            border-left: 1px solid var(--border-color);
        }

        .side-panel.collapsed {
            width: 46px;
        }

        .side-panel-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 8px;
            padding: 10px 12px;
            border-bottom: 1px solid var(--border-color);
            background: rgba(255,255,255,0.02);
        }

        .side-panel-title {
            font-size: 11px;
            font-weight: 700;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.6px;
            display: flex;
            align-items: center;
            gap: 8px;
            white-space: nowrap;
        }

        .side-panel-toggle {
            background: var(--bg-tertiary);
            border: 1px solid var(--border-color);
            color: var(--text-secondary);
            border-radius: 6px;
            padding: 4px 8px;
            cursor: pointer;
            font-size: 12px;
        }

        .side-panel-toggle:hover {
            border-color: var(--accent-blue);
            color: var(--text-primary);
        }

        .side-panel-body {
            flex: 1;
            overflow-y: auto;
        }

        .side-panel.collapsed .side-panel-body {
            display: none;
        }

        .side-panel.collapsed .side-panel-title span.label {
            display: none;
        }

        /* Center stack */
        .center-content {
            flex: 1;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        
        /* Map Container */
        .map-section {
            flex: 1;
            position: relative;
        }
        
        #map {
            width: 100%;
            height: 100%;
            background: var(--bg-primary);
        }
        
        .leaflet-container {
            background: var(--bg-primary);
        }
        
        /* Legend */
        .legend {
            position: absolute;
            bottom: 20px;
            left: 20px;
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 12px 16px;
            z-index: 1000;
        }
        
        .legend-title {
            font-size: 11px;
            font-weight: 600;
            color: var(--text-secondary);
            margin-bottom: 8px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .legend-item {
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 6px;
            font-size: 12px;
        }
        
        .legend-color {
            width: 12px;
            height: 12px;
            border-radius: 50%;
        }
        
        /* Map Search Bar */
        .map-search-bar {
            position: absolute;
            top: 12px;
            left: 50%;
            transform: translateX(-50%);
            z-index: 1000;
            width: 320px;
        }
        
        .map-search-bar input {
            width: 100%;
            padding: 10px 14px;
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            color: var(--text-primary);
            font-size: 13px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        }
        
        .map-search-bar input:focus {
            outline: none;
            border-color: var(--accent-blue);
        }
        
        .map-search-results {
            position: absolute;
            top: 100%;
            left: 0;
            right: 0;
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-top: none;
            border-radius: 0 0 8px 8px;
            max-height: 300px;
            overflow-y: auto;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        }
        
        .map-search-item {
            padding: 10px 14px;
            cursor: pointer;
            border-bottom: 1px solid var(--border-color);
            font-size: 12px;
        }
        
        .map-search-item:last-child {
            border-bottom: none;
        }
        
        .map-search-item:hover {
            background: var(--bg-tertiary);
        }
        
        .map-search-item .name {
            font-weight: 500;
            color: var(--text-primary);
        }
        
        .map-search-item .mac {
            font-family: monospace;
            font-size: 10px;
            color: var(--text-secondary);
        }
        
        .map-search-item .meta {
            font-size: 10px;
            color: var(--text-muted);
            margin-top: 2px;
        }
        
        /* Sidebar (right panel content styling) */
        .sidebar {
            width: 100%;
            background: transparent;
            border-left: none;
            display: flex;
            flex-direction: column;
        }
        
        .sidebar-section {
            padding: 16px;
            border-bottom: 1px solid var(--border-color);
        }
        
        .section-title {
            font-size: 11px;
            font-weight: 600;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 12px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        /* Stats Grid */
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 10px;
        }
        
        .stat-card {
            background: var(--bg-tertiary);
            border-radius: 8px;
            padding: 12px;
            text-align: center;
        }
        
        .stat-value {
            font-size: 24px;
            font-weight: 700;
            color: var(--text-primary);
        }
        
        .stat-label {
            font-size: 11px;
            color: var(--text-secondary);
            margin-top: 4px;
        }
        
        .stat-card.small {
            padding: 8px;
        }
        
        .stat-card.small .stat-value {
            font-size: 18px;
        }
        
        /* Upload Section */
        .upload-zone {
            border: 2px dashed var(--border-color);
            border-radius: 8px;
            padding: 24px;
            text-align: center;
            cursor: pointer;
            transition: all 0.2s;
        }
        
        .upload-zone:hover {
            border-color: var(--accent-blue);
            background: rgba(59, 130, 246, 0.05);
        }
        
        .upload-zone.dragover {
            border-color: var(--accent-green);
            background: rgba(16, 185, 129, 0.1);
        }
        
        .upload-icon {
            font-size: 32px;
            margin-bottom: 8px;
        }
        
        .upload-text {
            font-size: 13px;
            color: var(--text-secondary);
        }
        
        .upload-text strong {
            color: var(--accent-blue);
        }
        
        #file-input {
            display: none;
        }
        
        /* Filters */
        .filter-group {
            margin-bottom: 12px;
        }
        
        .filter-label {
            font-size: 11px;
            color: var(--text-secondary);
            margin-bottom: 6px;
        }
        
        .filter-buttons {
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
        }
        
        .filter-btn {
            background: var(--bg-tertiary);
            border: 1px solid var(--border-color);
            color: var(--text-secondary);
            padding: 6px 12px;
            border-radius: 4px;
            font-size: 12px;
            cursor: pointer;
            transition: all 0.2s;
        }
        
        .filter-btn:hover {
            border-color: var(--accent-blue);
            color: var(--text-primary);
        }
        
        .filter-btn.active {
            background: var(--accent-blue);
            border-color: var(--accent-blue);
            color: white;
        }
        
        /* AP Detail Panel */
        .detail-panel {
            flex: 1;
            padding: 16px;
            overflow-y: auto;
        }

        /* Leaflet hover tooltip */
        .leaflet-tooltip.wa-tooltip {
            background: rgba(15, 23, 42, 0.95);
            color: #e5e7eb;
            border: 1px solid rgba(148, 163, 184, 0.35);
            border-radius: 8px;
            padding: 8px 10px;
            box-shadow: 0 12px 28px rgba(0,0,0,0.35);
            font-size: 12px;
            line-height: 1.35;
        }
        
        .detail-empty {
            text-align: center;
            padding: 40px 20px;
            color: var(--text-secondary);
        }
        
        .detail-empty-icon {
            font-size: 48px;
            margin-bottom: 12px;
            opacity: 0.5;
        }
        
        .ap-header {
            margin-bottom: 16px;
        }
        
        .ap-name {
            font-size: 16px;
            font-weight: 600;
            color: var(--text-primary);
            word-break: break-all;
        }
        
        .ap-mac {
            font-size: 12px;
            color: var(--text-secondary);
            font-family: monospace;
            margin-top: 4px;
        }
        
        .classification-badge {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 6px 12px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: 600;
            margin-top: 8px;
        }
        
        .classification-badge.static {
            background: rgba(16, 185, 129, 0.2);
            color: var(--accent-green);
        }
        
        .classification-badge.mobile {
            background: rgba(245, 158, 11, 0.2);
            color: var(--accent-orange);
        }
        
        .classification-badge.uncertain {
            background: rgba(239, 68, 68, 0.2);
            color: var(--accent-red);
        }
        
        .ap-details-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 12px;
            margin-top: 16px;
        }
        
        .detail-item {
            background: var(--bg-tertiary);
            padding: 10px;
            border-radius: 6px;
        }
        
        .detail-item-label {
            font-size: 10px;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .detail-item-value {
            font-size: 14px;
            font-weight: 600;
            color: var(--text-primary);
            margin-top: 2px;
        }
        
        /* Confidence Bar */
        .confidence-section {
            margin-top: 16px;
        }
        
        .confidence-bar {
            height: 6px;
            background: var(--bg-tertiary);
            border-radius: 3px;
            overflow: hidden;
            margin-top: 6px;
        }
        
        .confidence-fill {
            height: 100%;
            background: var(--accent-blue);
            transition: width 0.3s;
        }
        
        /* Export Buttons */
        .export-buttons {
            display: flex;
            gap: 8px;
            padding: 16px;
            border-top: 1px solid var(--border-color);
        }
        
        .export-btn {
            flex: 1;
            background: var(--bg-tertiary);
            border: 1px solid var(--border-color);
            color: var(--text-primary);
            padding: 10px;
            border-radius: 6px;
            font-size: 12px;
            cursor: pointer;
            transition: all 0.2s;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 6px;
        }
        
        .export-btn:hover {
            background: var(--accent-blue);
            border-color: var(--accent-blue);
        }
        
        /* View Toggle */
        .view-toggle {
            display: flex;
            background: var(--bg-tertiary);
            border-radius: 6px;
            padding: 4px;
            margin-bottom: 12px;
        }
        
        .view-toggle-btn {
            flex: 1;
            padding: 8px;
            border: none;
            background: transparent;
            color: var(--text-secondary);
            font-size: 12px;
            cursor: pointer;
            border-radius: 4px;
            transition: all 0.2s;
        }
        
        .view-toggle-btn.active {
            background: var(--accent-blue);
            color: white;
        }
        
        /* Table View */

        /* Table Toolbar */
        .table-toolbar {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            align-items: center;
            margin-bottom: 12px;
            padding: 10px;
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 10px;
        }

        .table-toolbar .tb-group {
            display: flex;
            flex-direction: column;
            gap: 4px;
        }

        .table-toolbar label {
            font-size: 10px;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .table-toolbar input[type='text'],
        .table-toolbar input[type='number'],
        .table-toolbar select {
            background: var(--bg-tertiary);
            border: 1px solid var(--border-color);
            color: var(--text-primary);
            padding: 7px 8px;
            border-radius: 8px;
            font-size: 12px;
            min-width: 160px;
        }

        .table-toolbar input[type='number'] {
            min-width: 100px;
            width: 100px;
        }

        .table-toolbar .tb-actions {
            margin-left: auto;
            display: flex;
            gap: 10px;
            align-items: center;
        }

        .table-toolbar .tb-btn {
            background: rgba(30, 41, 59, 0.6);
            border: 1px solid var(--border-color);
            color: var(--text-primary);
            padding: 8px 10px;
            border-radius: 10px;
            font-size: 12px;
            cursor: pointer;
            white-space: nowrap;
        }

        .table-toolbar .tb-btn:hover {
            border-color: var(--accent-blue);
        }

        .sel-col {
            width: 46px;
        }

        .table-container {
            flex: 1;
            overflow: auto;
            padding: 16px;
        }
        
        .data-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 12px;
        }
        
        .data-table th,
        .data-table td {
            padding: 10px 12px;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
        }
        
        .data-table th {
            background: var(--bg-tertiary);
            color: var(--text-secondary);
            font-weight: 600;
            text-transform: uppercase;
            font-size: 10px;
            letter-spacing: 0.5px;
            position: sticky;
            top: 0;
        }
        
        .data-table tr:hover {
            background: var(--bg-tertiary);
        }
        
        .data-table td {
            color: var(--text-primary);
        }
        
        .rssi-bar {
            width: 60px;
            height: 6px;
            background: var(--bg-primary);
            border-radius: 3px;
            overflow: hidden;
        }
        
        .rssi-fill {
            height: 100%;
            background: linear-gradient(90deg, var(--accent-red), var(--accent-orange), var(--accent-green));
        }
        
        /* Loading Spinner */
        .loading {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 40px;
        }
        
        .spinner {
            width: 32px;
            height: 32px;
            border: 3px solid var(--border-color);
            border-top-color: var(--accent-blue);
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        
        /* No GPS Warning */
        .no-gps-warning {
            background: rgba(245, 158, 11, 0.1);
            border: 1px solid var(--accent-orange);
            border-radius: 8px;
            padding: 12px 16px;
            margin-bottom: 12px;
            font-size: 12px;
            color: var(--accent-orange);
        }
        
        /* Clear Button */
        .clear-btn {
            background: var(--accent-red);
            border: none;
            color: white;
            padding: 8px 16px;
            border-radius: 4px;
            font-size: 12px;
            cursor: pointer;
            margin-top: 12px;
        }
        
        .clear-btn:hover {
            background: #dc2626;
        }
        
        /* Loaded runs indicator */
        .loaded-runs {
            background: var(--surface-light);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 10px;
            margin-top: 12px;
            font-size: 12px;
        }
        
        .runs-label {
            color: var(--text-secondary);
            margin-bottom: 8px;
            font-weight: 500;
        }
        
        .run-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 4px 0;
            border-bottom: 1px solid var(--border-color);
        }
        
        .run-item:last-child {
            border-bottom: none;
        }
        
        .run-name {
            color: var(--text-primary);
            font-size: 11px;
            max-width: 150px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        
        .run-count {
            color: var(--text-muted);
            font-size: 10px;
        }

        /* Header buttons */
        .header-btn {
            background: rgba(30, 41, 59, 0.6);
            border: 1px solid var(--border-color);
            color: var(--text-primary);
            padding: 6px 10px;
            border-radius: 8px;
            font-size: 12px;
            cursor: pointer;
            margin-left: 8px;
        }

        .header-btn:hover {
            border-color: var(--accent-blue);
        }

        /* Worker opacity controls */
        .worker-opacity-list {
            display: flex;
            flex-direction: column;
            gap: 8px;
            margin-top: 6px;
        }

        .worker-row {
            display: grid;
            grid-template-columns: 16px 18px 1fr 42px;
            gap: 8px;
            align-items: center;
            font-size: 12px;
            color: var(--text-primary);
        }

        .worker-row input[type="range"] {
            width: 100%;
        }

        .worker-tag {
            font-family: monospace;
            font-weight: 700;
            color: var(--accent-blue);
        }

        .worker-alpha {
            font-family: monospace;
            font-size: 11px;
            color: var(--text-secondary);
            text-align: right;
        }

        /* Modal */
        .modal-overlay {
            position: fixed;
            inset: 0;
            background: rgba(0,0,0,0.6);
            display: none;
            /* Allow tall settings content to be scrollable on smaller screens */
            align-items: flex-start;
            justify-content: center;
            z-index: 9999;
            overflow-y: auto;
            padding: 30px 0;
        }

        .modal {
            width: min(560px, 92vw);
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.5);
            padding: 16px;
            max-height: 85vh;
            overflow-y: auto;
        }

        .modal-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 12px;
        }

        .modal-title {
            font-size: 14px;
            font-weight: 700;
            color: var(--text-primary);
        }

        .modal-close {
            background: transparent;
            border: 1px solid var(--border-color);
            color: var(--text-primary);
            border-radius: 8px;
            padding: 6px 10px;
            cursor: pointer;
        }

        .modal-close:hover {
            border-color: var(--accent-blue);
        }

        .settings-tabs {
            display: flex;
            gap: 8px;
            margin: 8px 0 12px 0;
            padding-bottom: 10px;
            border-bottom: 1px solid var(--border-color);
        }
        .settings-tab {
            padding: 6px 10px;
            border: 1px solid var(--border-color);
            border-radius: 999px;
            background: rgba(255,255,255,0.04);
            color: var(--text-primary);
            cursor: pointer;
            font-size: 12px;
            user-select: none;
        }
        .settings-tab.active {
            border-color: var(--accent-blue);
            box-shadow: 0 0 0 2px rgba(59,130,246,0.15);
        }
        .settings-pane { display: none; }
        .settings-pane.active { display: block; }

        .settings-row {
            display: grid;
            grid-template-columns: 1fr 120px;
            align-items: center;
            gap: 12px;
            margin: 10px 0;
        }

        .settings-row input[type="range"] {
            width: 100%;
        }

        /* Playback dock (affixed in left panel) */
        .side-panel-body.left-split {
            display: flex;
            flex-direction: column;
            height: 100%;
            overflow: hidden;
        }
        #detail-panel {
            flex: 1 1 auto;
            overflow: auto;
            min-height: 0;
        }
        .playback-dock {
            flex: 0 0 30%;
            min-height: 190px;
            background: rgba(15, 23, 42, 0.6);
            border-top: 1px solid var(--border-color);
            padding: 10px;
            display: flex;
            flex-direction: column;
            gap: 8px;
        }
        .playback-dock.collapsed {
            flex: 0 0 auto;
            min-height: unset;
        }
        .playback-dock-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        .playback-dock-title {
            font-size: 12px;
            font-weight: 700;
            color: var(--text-primary);
        }
        .playback-dock-toggle {
            background: rgba(30, 41, 59, 0.6);
            border: 1px solid var(--border-color);
            color: var(--text-primary);
            border-radius: 8px;
            padding: 4px 8px;
            cursor: pointer;
            font-size: 12px;
        }
        .playback-dock-toggle:hover {
            border-color: var(--accent-blue);
        }
        .playback-dock-body {
            display: flex;
            flex-direction: column;
            gap: 8px;
        }
        .playback-dock.collapsed .playback-dock-body {
            display: none;
        }
        .pb-row {
            display: flex;
            gap: 8px;
            align-items: center;
            flex-wrap: wrap;
        }

        .pb-btn {
            background: rgba(30, 41, 59, 0.6);
            border: 1px solid var(--border-color);
            color: var(--text-primary);
            padding: 6px 8px;
            border-radius: 8px;
            font-size: 12px;
            cursor: pointer;
        }

        .pb-btn:hover {
            border-color: var(--accent-blue);
        }

        .pb-time {
            font-family: monospace;
            font-size: 12px;
            color: var(--text-secondary);
            text-align: center;
        }

        .pb-row input[type="range"] {
            flex: 1 1 auto;
            min-width: 120px;
        }

        /* Detail actions */
        .detail-actions {
            margin-top: 12px;
            display: flex;
            gap: 10px;
        }

        .action-btn {
            flex: 1;
            background: rgba(30, 41, 59, 0.6);
            border: 1px solid var(--border-color);
            color: var(--text-primary);
            padding: 8px 10px;
            border-radius: 8px;
            font-size: 12px;
            cursor: pointer;
        }

        .action-btn:hover {
            border-color: var(--accent-blue);
        }

        /* Leaflet hover tooltips for rings */
        .wa-tooltip {
            background: rgba(15, 23, 42, 0.92) !important;
            border: 1px solid var(--border-color) !important;
            color: var(--text-primary) !important;
            box-shadow: 0 10px 25px rgba(0,0,0,0.35) !important;
            border-radius: 10px !important;
            padding: 8px 10px !important;
        }
        
        /* GNSS Link Styles */
        .gnss-container {
            flex: 1;
            padding: 20px;
            overflow-y: auto;
            background: var(--bg-primary);
        }
        
        .gnss-header {
            margin-bottom: 24px;
        }
        
        .gnss-header h2 {
            font-size: 20px;
            color: var(--text-primary);
            margin-bottom: 4px;
        }
        
        .gnss-subtitle {
            font-size: 13px;
            color: var(--text-secondary);
        }
        
        .gnss-upload-section {
            margin-bottom: 24px;
        }
        
        .gnss-upload-zone {
            border: 2px dashed var(--border-color);
            border-radius: 12px;
            padding: 32px;
            text-align: center;
            cursor: pointer;
            transition: all 0.2s;
            background: var(--bg-secondary);
        }
        
        .gnss-upload-zone:hover {
            border-color: var(--accent-blue);
            background: rgba(59, 130, 246, 0.05);
        }
        
        .gnss-session-info {
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 16px;
            margin-bottom: 24px;
        }
        
        .gnss-info-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 12px;
        }
        
        .gnss-session-title {
            font-weight: 600;
            color: var(--text-primary);
        }
        
        .gnss-clear-btn {
            background: rgba(239, 68, 68, 0.2);
            border: 1px solid var(--accent-red);
            color: var(--accent-red);
            padding: 4px 10px;
            border-radius: 6px;
            font-size: 11px;
            cursor: pointer;
        }
        
        .gnss-info-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 12px;
        }
        
        .gnss-info-card {
            background: var(--bg-tertiary);
            border-radius: 8px;
            padding: 12px;
            text-align: center;
        }
        
        .gnss-info-label {
            font-size: 11px;
            color: var(--text-secondary);
            margin-bottom: 4px;
        }
        
        .gnss-info-value {
            font-size: 18px;
            font-weight: 600;
            color: var(--accent-blue);
        }
        
        .gnss-info-small {
            font-size: 12px;
        }
        
        .gnss-metrics {
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 16px;
            margin-bottom: 24px;
        }
        
        .gnss-metrics h3 {
            font-size: 14px;
            color: var(--text-primary);
            margin-bottom: 12px;
        }
        
        .gnss-metrics-grid {
            display: grid;
            grid-template-columns: repeat(5, 1fr);
            gap: 12px;
        }
        
        .gnss-metric-card {
            background: var(--bg-tertiary);
            border-radius: 8px;
            padding: 16px;
            text-align: center;
        }
        
        .gnss-metric-label {
            font-size: 11px;
            color: var(--text-secondary);
            margin-bottom: 6px;
        }
        
        .gnss-metric-value {
            font-size: 20px;
            font-weight: 600;
            color: var(--accent-green);
        }
        
        .gnss-metric-unit {
            font-size: 10px;
            color: var(--text-muted);
            margin-top: 2px;
        }
        
        .gnss-receivers {
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 16px;
            margin-bottom: 24px;
        }
        
        .gnss-receivers h3 {
            font-size: 14px;
            color: var(--text-primary);
            margin-bottom: 12px;
        }
        
        .gnss-receiver-table th,
        .gnss-receiver-table td {
            text-align: center;
        }
        
        .gnss-actions {
            display: flex;
            gap: 12px;
            justify-content: center;
        }
        
        .gnss-action-btn {
            background: linear-gradient(135deg, var(--accent-blue), #6366f1);
            border: none;
            color: white;
            padding: 12px 24px;
            border-radius: 8px;
            font-size: 14px;
            font-weight: 500;
            cursor: pointer;
            transition: transform 0.2s;
        }
        
        .gnss-action-btn:hover {
            transform: translateY(-2px);
        }
        
        @media (max-width: 1200px) {
            .gnss-metrics-grid {
                grid-template-columns: repeat(3, 1fr);
            }
        }
        
        @media (max-width: 768px) {
            .gnss-metrics-grid {
                grid-template-columns: repeat(2, 1fr);
            }
            .gnss-info-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="app-container">
        <!-- Header -->
        <header class="header">
            <div class="header-left">
                <span style="font-size: 24px;">⚡</span>
                <h1>Silver Streak Analyzer</h1>
                <span class="header-badge">SSA v4.6.2</span>
            </div>
            <div class="header-right">
                <div class="status-indicator">
                    <div class="status-dot" id="status-dot"></div>
                    <span id="status-text">Connecting...</span>
                </div>
                <span id="ap-count">0 APs loaded</span>
                <button class="header-btn" id="btn-playback" onclick="togglePlaybackPopup()">⏯️ Playback</button>
                <button class="header-btn" onclick="openSettings()">⚙️ Settings</button>
            </div>
        </header>
        
        <!-- Main Content -->
        <div class="main-content">
            <!-- Left: SOI Info Panel -->
            <div class="side-panel left-panel" id="left-panel">
                <div class="side-panel-header">
                    <div class="side-panel-title"><span>ℹ️</span> <span class="label">SOI Info</span></div>
                    <button class="side-panel-toggle" onclick="togglePanel('left-panel')" title="Collapse/expand">⇔</button>
                </div>
                <div class="side-panel-body left-split">
                    <div class="detail-panel" id="detail-panel">
                        <div class="detail-empty">
                            <div class="detail-empty-icon">📡</div>
                            <div>Select a signal to view details</div>
                            <div style="font-size: 11px; margin-top: 8px; color: var(--text-muted);">
                                Click any marker on the map or any row in the table.
                            </div>
                        </div>
                    </div>
                    <div class="playback-dock" id="playback-dock">
                        <div class="playback-dock-header">
                            <div class="playback-dock-title">Playback</div>
                            <button class="playback-dock-toggle" onclick="togglePlaybackPopup()" title="Collapse/expand playback">▾</button>
                        </div>
                        <div class="playback-dock-body" id="playback-dock-body">
                            <div class="pb-row">
                                <button class="pb-btn" title="Start" onclick="pbJumpStart()">⏮</button>
                                <button class="pb-btn" title="Back" onclick="pbPlayBackward()">⏪</button>
                                <button class="pb-btn" id="pb-playpause" title="Play/Pause" onclick="pbTogglePlay()">▶</button>
                                <button class="pb-btn" title="Forward" onclick="pbPlayForward()">⏩</button>
                                <button class="pb-btn" title="End" onclick="pbJumpEnd()">⏭</button>
                                <button class="pb-btn" title="Stop" onclick="pbStop()">⏹</button>
                            </div>
                            <div class="pb-row" style="justify-content:center;">
                                <div class="pb-time" id="pb-time">Playback: --</div>
                            </div>
                            <div class="pb-row" style="justify-content:space-between; gap:10px; align-items:center;">
                                <span style="font-size: 12px; color: var(--text-secondary);">Speed</span>
                                <input type="range" id="pb-speed" min="0.25" max="10" value="1" step="0.25" oninput="pbSetSpeed(this.value)">
                                <span style="font-family: monospace; font-size: 12px; color: var(--text-secondary);" id="pb-speed-label">1.00x</span>
                            </div>
                        </div>
                    </div>

                </div>
            </div>

            <!-- Center: Map/Table -->
            <div class="center-content">
            <!-- Map/Table Section -->
            <div class="map-section" id="map-section">
                <div id="map"></div>
                
                <!-- Floating Search Bar on Map -->
                <div class="map-search-bar" id="map-search-bar">
                    <input type="text" id="map-search" placeholder="🔍 Search name, MAC, or SSID..." oninput="handleMapSearch(this.value)">
                    <div class="map-search-results" id="map-search-results" style="display: none;"></div>
                </div>
                
                <!-- Legend -->
                <div class="legend">
                    <div class="legend-title">Classification</div>
                    <div class="legend-item">
                        <div class="legend-color" style="background: var(--accent-green);"></div>
                        <span>Static Networks</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background: var(--accent-orange);"></div>
                        <span>Mobile/Vehicle APs</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background: var(--accent-red);"></div>
                        <span>Uncertain</span>
                    </div>
                </div>
            </div>
            
            <!-- Table View (hidden by default) -->
            <div class="table-container" id="table-section" style="display: none;">
                <div class="table-toolbar">
                    <div class="tb-group">
                        <label>Search</label>
                        <input type="text" id="tb-search" placeholder="Name or MAC" oninput="setTableFilter('q', this.value)">
                    </div>
                    <div class="tb-group">
                        <label>Sort By</label>
                        <select id="tb-sort" onchange="setTableSort(this.value)">
                            <option value="recent">Most Recent</option>
                            <option value="oldest">Oldest First</option>
                            <option value="most-detections">Most Detections</option>
                            <option value="fewest-detections">Fewest Detections</option>
                            <option value="strongest">Strongest Signal</option>
                            <option value="weakest">Weakest Signal</option>
                            <option value="alpha-asc">A → Z</option>
                            <option value="alpha-desc">Z → A</option>
                            <option value="class">By Classification</option>
                        </select>
                    </div>
                    <div class="tb-group">
                        <label>Type</label>
                        <select id="tb-type" onchange="setTableFilter('type', this.value)">
                            <option value="all">All</option>
                            <option value="wifi">WiFi</option>
                            <option value="ble">BLE</option>
                            <option value="thread">Thread</option>
                            <option value="halow">HaLow</option>
                        </select>
                    </div>
                    <div class="tb-group">
                        <label>Classification</label>
                        <select id="tb-class" onchange="setTableFilter('class', this.value)">
                            <option value="all">All</option>
                            <option value="static">Static</option>
                            <option value="mobile">Mobile</option>
                            <option value="uncertain">Uncertain</option>
                        </select>
                    </div>
                    <div class="tb-group">
                        <label>Channel</label>
                        <input type="text" id="tb-channel" placeholder="e.g., 1, 6, 11" oninput="setTableFilter('channel', this.value)">
                    </div>
                    <div class="tb-group">
                        <label>WiFi Security</label>
                        <select id="tb-security" onchange="setTableFilter('security', this.value)">
                            <option value="all">All</option>
                            <option value="open">Open</option>
                            <option value="owe">OWE</option>
                            <option value="wep">WEP</option>
                            <option value="wpa1">WPA1</option>
                            <option value="wpa2">WPA2</option>
                            <option value="wpa3">WPA3</option>
                            <option value="wpa2wpa3">WPA2/WPA3</option>
                            <option value="unknown">Unknown</option>
                        </select>
                    </div>
                    <div class="tb-group">
                        <label>RSSI min</label>
                        <input type="number" id="tb-rssi-min" placeholder="-90" min="-120" max="0" step="1" oninput="setTableFilter('rssiMin', this.value)">
                    </div>
                    <div class="tb-group">
                        <label>RSSI max</label>
                        <input type="number" id="tb-rssi-max" placeholder="-30" min="-120" max="0" step="1" oninput="setTableFilter('rssiMax', this.value)">
                    </div>

                    <div class="tb-actions">
                        <button class="tb-btn" onclick="viewSelectedFromTable()">🗺️ View Selected</button>
                        <button class="tb-btn" onclick="generateReportFromTable()">📑 Generate Report</button>
                        <button class="tb-btn" onclick="clearTableSelection()">Clear Selection</button>
                    </div>
                </div>

                <table class="data-table" id="data-table">
                    <thead>
                        <tr>
                            <th class="sel-col"><input type="checkbox" id="tb-select-all" onchange="toggleSelectAllTable(this.checked)" title="Select all visible"></th>
                            <th>Name/SSID</th>
                            <th>MAC</th>
                            <th>Type</th>
                            <th>First Seen</th>
                            <th>Last Seen</th>
                            <th>Security</th>
                            <th>Worker</th>
                            <th>RSSI</th>
                            <th>Detections</th>
                            <th>Classification</th>
                        </tr>
                    </thead>
                    <tbody id="table-body"></tbody>
                </table>
            </div>
            
            <!-- GNSS Link View (hidden by default) -->
            <div class="gnss-container" id="gnss-section" style="display: none;">
                <div class="gnss-header">
                    <h2>📡 GNSS Link - Multi-Receiver Variance Analysis</h2>
                    <p class="gnss-subtitle">Compare position accuracy across multiple GNSS receivers</p>
                </div>
                
                <div class="gnss-content">
                    <!-- Upload Section -->
                    <div class="gnss-upload-section">
                        <div class="gnss-upload-zone" id="gnss-upload-zone" onclick="document.getElementById('gnss-file-input').click()">
                            <div class="upload-icon">📤</div>
                            <div class="upload-text">
                                <strong>Click to upload GNSS CSV files</strong>
                                <div style="font-size: 11px; color: var(--text-muted); margin-top: 4px;">
                                    Option B format (multi-receiver with ReceiverID column) or individual receiver CSVs
                                </div>
                            </div>
                        </div>
                        <input type="file" id="gnss-file-input" accept=".csv" multiple onchange="handleGNSSFileUpload(event)" style="display:none;">
                    </div>
                    
                    <!-- Session Info -->
                    <div class="gnss-session-info" id="gnss-session-info" style="display: none;">
                        <div class="gnss-info-header">
                            <span class="gnss-session-title">Session: <span id="gnss-session-id">-</span></span>
                            <button class="gnss-clear-btn" onclick="clearGNSSSession()">🗑️ Clear Session</button>
                        </div>
                        <div class="gnss-info-grid">
                            <div class="gnss-info-card">
                                <div class="gnss-info-label">Receivers</div>
                                <div class="gnss-info-value" id="gnss-receiver-count">0</div>
                            </div>
                            <div class="gnss-info-card">
                                <div class="gnss-info-label">Positions</div>
                                <div class="gnss-info-value" id="gnss-position-count">0</div>
                            </div>
                            <div class="gnss-info-card">
                                <div class="gnss-info-label">Time Range</div>
                                <div class="gnss-info-value gnss-info-small" id="gnss-time-range">-</div>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Metrics Cards -->
                    <div class="gnss-metrics" id="gnss-metrics" style="display: none;">
                        <h3>📊 Variance Metrics (All Receivers)</h3>
                        <div class="gnss-metrics-grid">
                            <div class="gnss-metric-card">
                                <div class="gnss-metric-label">Std Dev (H)</div>
                                <div class="gnss-metric-value" id="gnss-std-h">-</div>
                                <div class="gnss-metric-unit">meters</div>
                            </div>
                            <div class="gnss-metric-card">
                                <div class="gnss-metric-label">Std Dev (V)</div>
                                <div class="gnss-metric-value" id="gnss-std-v">-</div>
                                <div class="gnss-metric-unit">meters</div>
                            </div>
                            <div class="gnss-metric-card">
                                <div class="gnss-metric-label">CEP (50%)</div>
                                <div class="gnss-metric-value" id="gnss-cep-50">-</div>
                                <div class="gnss-metric-unit">meters</div>
                            </div>
                            <div class="gnss-metric-card">
                                <div class="gnss-metric-label">2DRMS (95%)</div>
                                <div class="gnss-metric-value" id="gnss-2drms">-</div>
                                <div class="gnss-metric-unit">meters</div>
                            </div>
                            <div class="gnss-metric-card">
                                <div class="gnss-metric-label">Max Deviation</div>
                                <div class="gnss-metric-value" id="gnss-max-dev">-</div>
                                <div class="gnss-metric-unit">meters</div>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Receiver Table -->
                    <div class="gnss-receivers" id="gnss-receivers" style="display: none;">
                        <h3>📡 Receiver Statistics</h3>
                        <table class="data-table gnss-receiver-table" id="gnss-receiver-table">
                            <thead>
                                <tr>
                                    <th>Receiver</th>
                                    <th>Truth</th>
                                    <th>Positions</th>
                                    <th>Avg Sats</th>
                                    <th>Avg HDOP</th>
                                    <th>Avg Dev</th>
                                    <th>CEP50</th>
                                    <th>2DRMS</th>
                                </tr>
                            </thead>
                            <tbody id="gnss-receiver-tbody"></tbody>
                        </table>
                    </div>
                    
                    <!-- Actions -->
                    <div class="gnss-actions" id="gnss-actions" style="display: none;">
                        <button class="gnss-action-btn" onclick="generateGNSSReport()">📑 Generate Report</button>
                        <button class="gnss-action-btn" onclick="exportGNSSCSV()">📥 Export CSV</button>
                    </div>
                </div>
            </div>
            
            </div>

            <!-- Right: Controls -->
            <div class="side-panel right-panel" id="right-panel">
                <div class="side-panel-header">
                    <div class="side-panel-title"><span>🧰</span> <span class="label">Controls</span></div>
                    <button class="side-panel-toggle" onclick="togglePanel('right-panel')" title="Collapse/expand">⇔</button>
                </div>
                <div class="side-panel-body sidebar">
                <!-- View Toggle -->
                <div class="sidebar-section">
                    <div class="view-toggle">
                        <button class="view-toggle-btn active" id="btn-map-view" onclick="setView('map')">🗺️ Map View</button>
                        <button class="view-toggle-btn" id="btn-table-view" onclick="setView('table')">📊 Table View</button>
                        <button class="view-toggle-btn" id="btn-gnss-view" onclick="setView('gnss')">📡 GNSS Link</button>
                    </div>
                </div>
                
                <!-- Upload -->
                <div class="sidebar-section">
                    <div class="section-title">📁 Load Data</div>
                    <div class="upload-zone" id="upload-zone" onclick="document.getElementById('file-input').click()">
                        <div class="upload-icon">📤</div>
                        <div class="upload-text">
                            <strong>Click to upload</strong> or drag CSV files
                            <div style="font-size: 11px; color: var(--text-muted); margin-top: 4px;">
                                Multiple files supported for multi-day analysis
                            </div>
                        </div>
                    </div>
                    <input type="file" id="file-input" accept=".csv" multiple onchange="handleFileUpload(event)">
                    <div id="loaded-runs" class="loaded-runs" style="display: none;">
                        <div class="runs-header" style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                            <div class="runs-label">📋 Loaded Runs:</div>
                            <button class="runs-clear-btn" onclick="clearSelectedRuns()" style="font-size: 10px; padding: 2px 6px; background: var(--surface-light); border: 1px solid var(--border-color); border-radius: 4px; color: var(--text-secondary); cursor: pointer;">Clear Selected</button>
                        </div>
                        <div id="runs-list"></div>
                    </div>
                    <button class="clear-btn" onclick="clearData()">🗑️ Clear All Data</button>
                </div>
                
                <!-- Stats -->
                <div class="sidebar-section">
                    <div class="section-title">📊 Statistics</div>
                    <div id="no-gps-warning" class="no-gps-warning" style="display: none;">
                        ⚠️ Data loaded without GPS coordinates. Using table view for signal analysis.
                    </div>
                    <div class="stats-grid">
                        <div class="stat-card">
                            <div class="stat-value" id="stat-total">0</div>
                            <div class="stat-label">Total APs</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value" id="stat-detections">0</div>
                            <div class="stat-label">Detections</div>
                        </div>
                        <div class="stat-card small">
                            <div class="stat-value" id="stat-static" style="color: var(--accent-green);">0</div>
                            <div class="stat-label">Static</div>
                        </div>
                        <div class="stat-card small">
                            <div class="stat-value" id="stat-mobile" style="color: var(--accent-orange);">0</div>
                            <div class="stat-label">Mobile</div>
                        </div>
                        <div class="stat-card small">
                            <div class="stat-value" id="stat-uncertain" style="color: var(--accent-red);">0</div>
                            <div class="stat-label">Uncertain</div>
                        </div>
                        <div class="stat-card small">
                            <div class="stat-value" id="stat-with-gps">0</div>
                            <div class="stat-label">With GPS</div>
                        </div>
                    </div>
                    
                    <!-- Filtered/Selection Stats -->
                    <div class="filter-stats-section" style="margin-top: 12px; padding-top: 12px; border-top: 1px solid var(--border-color);">
                        <div class="filter-label" style="margin-bottom: 8px;">📋 Displayed / Selected</div>
                        <div class="stats-grid">
                            <div class="stat-card small">
                                <div class="stat-value" id="stat-displayed" style="color: var(--accent-blue);">0</div>
                                <div class="stat-label">Displayed</div>
                            </div>
                            <div class="stat-card small">
                                <div class="stat-value" id="stat-selected" style="color: var(--text-primary);">0</div>
                                <div class="stat-label">Selected</div>
                            </div>
                            <div class="stat-card small">
                                <div class="stat-value" id="stat-filtered-static" style="color: var(--accent-green);">0</div>
                                <div class="stat-label">Flt Static</div>
                            </div>
                            <div class="stat-card small">
                                <div class="stat-value" id="stat-filtered-mobile" style="color: var(--accent-orange);">0</div>
                                <div class="stat-label">Flt Mobile</div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Filters -->
                <div class="sidebar-section">
                    <div class="section-title">🎚️ Filters</div>
                    
                    <div class="filter-group">
                        <div class="filter-label">Classification</div>
                        <div class="filter-buttons" id="classification-filters">
                            <button class="filter-btn active" data-filter="all">All</button>
                            <button class="filter-btn" data-filter="static">Static</button>
                            <button class="filter-btn" data-filter="mobile">Mobile</button>
                            <button class="filter-btn" data-filter="uncertain">Uncertain</button>
                        </div>
                    </div>
                    
                    <div class="filter-group">
                        <div class="filter-label">Worker/Antenna</div>
                        <div class="worker-opacity-list" id="worker-opacity-list"></div>
                        <div style="font-size: 11px; color: var(--text-secondary); margin-top: 6px;">
                            ✅ check = show, 🎚️ slider = opacity
                        </div>
                    </div>
                    
                    <div class="filter-group">
                        <div class="filter-label">Device Type</div>
                        <div class="filter-buttons" id="device-filters">
                            <button class="filter-btn active" data-filter="all">All</button>
                            <button class="filter-btn" data-filter="wifi">WiFi</button>
                            <button class="filter-btn" data-filter="ble">BLE</button>
                            <button class="filter-btn" data-filter="thread">Thread/Zigbee</button>
                            <button class="filter-btn" data-filter="matter">Matter</button>
                            <button class="filter-btn" data-filter="halow">HaLow</button>
                        </div>
                    </div>

                    <div class="filter-group">
                        <div class="filter-label">WiFi Security</div>
                        <div style="font-size: 11px; color: var(--text-secondary); margin-bottom: 6px;">Auto-filters to WiFi devices only.</div>
                        <div class="filter-buttons" id="security-filters">
                            <button class="filter-btn active" data-filter="all">All</button>
                            <button class="filter-btn" data-filter="open">Open</button>
                            <button class="filter-btn" data-filter="owe">OWE</button>
                            <button class="filter-btn" data-filter="wep">WEP</button>
                            <button class="filter-btn" data-filter="wpa1">WPA1</button>
                            <button class="filter-btn" data-filter="wpa2">WPA2</button>
                            <button class="filter-btn" data-filter="wpa3">WPA3</button>
                            <button class="filter-btn" data-filter="wpa2wpa3">WPA2/3</button>
                            <button class="filter-btn" data-filter="unknown">Unknown</button>
                        </div>
                    </div>
                </div>
                <!-- Export -->
                <div class="export-buttons">
                    <button class="export-btn" onclick="generateSelectedReport()">📑 Report</button>
                    <button class="export-btn" onclick="exportGeoJSON()">📍 GeoJSON</button>
                    <button class="export-btn" onclick="exportKML()">🌍 KML</button>
                    <button class="export-btn" onclick="exportCSV()">📄 CSV</button>
                </div>
            </div>
            </div>
        </div>
    </div>

    <!-- Settings Modal -->
    <div class="modal-overlay" id="settings-modal" onclick="closeSettingsFromBackdrop(event)">
        <div class="modal" onclick="event.stopPropagation()">
            <div class="modal-header">
                <div class="modal-title">⚙️ Settings</div>
                <button class="modal-close" onclick="closeSettings()">Close</button>
            </div>

            <div class="settings-tabs">
                <div class="settings-tab active" id="settings-tab-general" onclick="showSettingsPane('general')">General</div>
                <div class="settings-tab" id="settings-tab-report" onclick="showSettingsPane('report')">Report</div>
                <div class="settings-tab" id="settings-tab-charts" onclick="showSettingsPane('charts')">Charts</div>
                <div class="settings-tab" id="settings-tab-whitelist" onclick="showSettingsPane('whitelist')">Whitelist</div>
                <div class="settings-tab" id="settings-tab-about" onclick="showSettingsPane('about')">About</div>
            </div>

            <div class="settings-pane active" id="settings-pane-general">
            <div class="settings-row">
                <div>
                    <div class="filter-label">Report Format</div>
                    <div style="font-size: 11px; color: var(--text-secondary);">Select output format for generated reports.</div>
                </div>
                <select id="report-format" style="padding: 6px 10px; border-radius: 6px; background: var(--surface-light); border: 1px solid var(--border-color); color: var(--text-primary);">
                    <option value="pdf">PDF Report</option>
                    <option value="html">HTML Report</option>
                </select>
            </div>

            <div class="settings-row">
                <div>
                    <div class="filter-label">Focus mode: other SOI opacity</div>
                    <div style="font-size: 11px; color: var(--text-secondary);">0% = hide others, 100% = no dimming. Applies when a single SOI is selected.</div>
                </div>
                <div style="display: flex; flex-direction: column; gap: 6px;">
                    <input type="range" id="dim-others-slider" min="0" max="100" value="25" step="1" oninput="setDimOthersOpacity(this.value)">
                    <div style="font-size: 11px; color: var(--text-secondary); text-align: right;"><span id="dim-others-value">25</span>%</div>
                </div>
            </div>

            <div style="margin-top: 14px;">
                <div class="filter-label">CSV Profile</div>
                <div style="font-size: 11px; color: var(--text-secondary); margin-bottom: 8px;">Select the column mapping used when importing CSV files.</div>
                <!--
                  IMPORTANT UX NOTE:
                  Avoid nested scroll containers inside the Settings modal.
                  A scrollable profile list can "trap" the mouse wheel, making it feel like
                  the Settings modal is missing content. Let the modal itself scroll instead.
                -->
                <div id="profile-list" style="border: 1px solid var(--border-color); border-radius: 10px; padding: 10px;"></div>
                <div style="display: flex; gap: 10px; margin-top: 10px; align-items:center;">
                    <button class="action-btn" onclick="triggerProfileUpload()">+ Upload Profile JSON</button>
                    <input type="file" id="profile-upload" accept="application/json" style="display:none" onchange="uploadProfileJson(this.files[0])">
                    <div id="profile-status" style="font-size: 11px; color: var(--text-secondary);"></div>
                </div>
                <div style="margin-top: 8px; font-size: 11px; color: var(--text-secondary);">More settings below — scroll ↓</div>
            </div>

            </div>

            <div class="settings-pane" id="settings-pane-report">
            <div style="margin-top: 14px;">
                <div class="filter-label">Report Options</div>
                <div style="font-size: 11px; color: var(--text-secondary); margin-bottom: 8px;">Applies to generated PDF/HTML reports.</div>

                <div class="settings-row">
                    <div>
                        <div class="filter-label">PDF Font</div>
                        <div style="font-size: 11px; color: var(--text-secondary);">Built-in ReportLab fonts.</div>
                    </div>
                    <select id="pdf-font-family" style="padding: 6px 10px; border-radius: 6px; background: var(--surface-light); border: 1px solid var(--border-color); color: var(--text-primary);" onchange="saveReportOptions()">
                        <option value="Helvetica">Helvetica</option>
                        <option value="Times-Roman">Times</option>
                        <option value="Courier">Courier</option>
                    </select>
                </div>

                <div class="settings-row">
                    <div>
                        <div class="filter-label">PDF Body Font Size</div>
                        <div style="font-size: 11px; color: var(--text-secondary);">Default: 10</div>
                    </div>
                    <input id="pdf-body-size" type="number" min="8" max="18" value="10" style="width: 90px; padding: 6px 10px; border-radius: 6px; background: var(--surface-light); border: 1px solid var(--border-color); color: var(--text-primary);" onchange="saveReportOptions()">
                </div>

                <div class="settings-row">
                    <div>
                        <div class="filter-label">PDF Title Font Size</div>
                        <div style="font-size: 11px; color: var(--text-secondary);">Default: 24</div>
                    </div>
                    <input id="pdf-title-size" type="number" min="16" max="48" value="24" style="width: 90px; padding: 6px 10px; border-radius: 6px; background: var(--surface-light); border: 1px solid var(--border-color); color: var(--text-primary);" onchange="saveReportOptions()">
                </div>

                <div class="settings-row">
                    <div>
                        <div class="filter-label">HTML Font</div>
                        <div style="font-size: 11px; color: var(--text-secondary);">Applies to HTML report export.</div>
                    </div>
                    <select id="html-font-family" style="padding: 6px 10px; border-radius: 6px; background: var(--surface-light); border: 1px solid var(--border-color); color: var(--text-primary);" onchange="saveReportOptions()">
                        <option value="system-ui">System UI</option>
                        <option value="serif">Serif</option>
                        <option value="monospace">Monospace</option>
                    </select>
                </div>

                <div class="settings-row">
                    <div>
                        <div class="filter-label">HTML Font Size</div>
                        <div style="font-size: 11px; color: var(--text-secondary);">Default: 14</div>
                    </div>
                    <input id="html-font-size" type="number" min="10" max="22" value="14" style="width: 90px; padding: 6px 10px; border-radius: 6px; background: var(--surface-light); border: 1px solid var(--border-color); color: var(--text-primary);" onchange="saveReportOptions()">
                </div>

                <div class="settings-row">
                    <div>
                        <div class="filter-label">HTML Title Bold</div>
                        <div style="font-size: 11px; color: var(--text-secondary);">Bold report title in HTML.</div>
                    </div>
                    <input id="html-title-bold" type="checkbox" checked onchange="saveReportOptions()">
                </div>

                <div class="settings-row">
                    <div>
                        <div class="filter-label">Company Name</div>
                        <div style="font-size: 11px; color: var(--text-secondary);">Shown near the top of the report.</div>
                    </div>
                    <input id="company-name" type="text" placeholder="(optional)" style="width: 260px; padding: 6px 10px; border-radius: 6px; background: var(--surface-light); border: 1px solid var(--border-color); color: var(--text-primary);" oninput="saveReportOptions(true)">
                </div>

                <div class="settings-row">
                    <div>
                        <div class="filter-label">Company Font</div>
                        <div style="font-size: 11px; color: var(--text-secondary);">PDF company line uses these settings.</div>
                    </div>
                    <div style="display:flex; gap:10px; align-items:center;">
                        <select id="company-font-family" style="padding: 6px 10px; border-radius: 6px; background: var(--surface-light); border: 1px solid var(--border-color); color: var(--text-primary);" onchange="saveReportOptions()">
                            <option value="Helvetica">Helvetica</option>
                            <option value="Times-Roman">Times</option>
                            <option value="Courier">Courier</option>
                        </select>
                        <input id="company-font-size" type="number" min="8" max="24" value="12" style="width: 90px; padding: 6px 10px; border-radius: 6px; background: var(--surface-light); border: 1px solid var(--border-color); color: var(--text-primary);" onchange="saveReportOptions()">
                        <label style="display:flex; align-items:center; gap:6px; font-size: 11px; color: var(--text-secondary);"><input id="company-bold" type="checkbox" onchange="saveReportOptions()">Bold</label>
                    </div>
                </div>

                <div class="settings-row">
                    <div>
                        <div class="filter-label">Watermark Logo</div>
                        <div style="font-size: 11px; color: var(--text-secondary);">Upload a logo used as a watermark in reports.</div>
                    </div>
                    <div style="display:flex; flex-direction:column; gap:8px;">
                        <label style="display:flex; align-items:center; gap:8px; font-size: 11px; color: var(--text-secondary);"><input id="watermark-enabled" type="checkbox" onchange="saveReportOptions()">Enable watermark</label>
                        <div style="display:flex; gap:10px; align-items:center;">
                            <button class="action-btn" onclick="triggerWatermarkUpload()">Upload Logo</button>
                            <input type="file" id="watermark-upload" accept="image/*" style="display:none" onchange="uploadWatermarkFile(this.files[0])">
                            <div id="watermark-status" style="font-size: 11px; color: var(--text-secondary);"></div>
                        </div>
                        <div style="display:flex; gap:10px; align-items:center;">
                            <span style="font-size: 11px; color: var(--text-secondary);">Opacity</span>
                            <input type="range" id="watermark-opacity" min="0" max="20" value="8" step="1" oninput="updateWatermarkOpacityLabel(this.value); saveReportOptions(true)">
                            <span style="font-family: monospace; font-size: 11px; color: var(--text-secondary);" id="watermark-opacity-label">8%</span>
                        </div>
                    </div>
                </div>

                
                </div>
            
                <div style="margin-top: 18px; padding-top: 12px; border-top: 1px solid var(--border-color);">
                    <div class="filter-label">Route Overview Map</div>
                    <div style="font-size: 11px; color: var(--text-secondary); margin-bottom: 8px;">Adds an overview of the full survey route to the report.</div>

                    <div class="settings-row">
                        <div>
                            <div class="filter-label">Include route overview map</div>
                            <div style="font-size: 11px; color: var(--text-secondary);">Adds a map image of the full GPS route (if available).</div>
                        </div>
                        <input id="route-map-enabled" type="checkbox" checked onchange="saveReportOptions()">
                    </div>

                    <div class="settings-row">
                        <div>
                            <div class="filter-label">Use basemap tiles (if available)</div>
                            <div style="font-size: 11px; color: var(--text-secondary);">Uses cached map tiles when available; falls back to a plain route plot if offline.</div>
                        </div>
                        <input id="route-map-use-basemap" type="checkbox" checked onchange="saveReportOptions()">
                    </div>

                    <div class="settings-row">
                        <div>
                            <div class="filter-label">Route padding</div>
                            <div style="font-size: 11px; color: var(--text-secondary);">Adds extra space around the route for context (default 10%).</div>
                        </div>
                        <div style="display:flex; gap:8px; align-items:center;">
                            <input id="route-map-padding" type="number" min="0" max="50" value="10" style="width: 90px; padding: 6px 10px; border-radius: 6px; background: var(--surface-light); border: 1px solid var(--border-color); color: var(--text-primary);" onchange="saveReportOptions()">
                            <div style="font-size: 11px; color: var(--text-secondary);">%</div>
                        </div>
                    </div>
                </div>
</div>

            <div class="settings-pane" id="settings-pane-charts">
            <div style="margin-top: 14px;">
                    <div class="filter-label">Report Charts</div>
                    <div style="font-size: 11px; color: var(--text-secondary); margin-bottom: 8px;">Optional pie charts shown near the top of reports.</div>

                    <div class="settings-row">
                        <div>
                            <div class="filter-label">Include charts in PDF</div>
                            <div style="font-size: 11px; color: var(--text-secondary);">Adds summary charts after the executive summary.</div>
                        </div>
                        <input id="charts-pdf-enabled" type="checkbox" checked onchange="saveReportOptions()">
                    </div>

                    <div class="settings-row">
                        <div>
                            <div class="filter-label">Include charts in HTML</div>
                            <div style="font-size: 11px; color: var(--text-secondary);">Adds summary charts near the top of HTML reports.</div>
                        </div>
                        <input id="charts-html-enabled" type="checkbox" checked onchange="saveReportOptions()">
                    </div>

                    <div class="settings-row">
                        <div>
                            <div class="filter-label">Charts to include</div>
                            <div style="font-size: 11px; color: var(--text-secondary);">Signal types, static/mobile, and most-detected channels.</div>
                        </div>
                        <div style="display:flex; flex-direction:column; gap:8px;">
                            <label style="display:flex; align-items:center; gap:8px; font-size: 11px; color: var(--text-secondary);"><input id="chart-signal-types" type="checkbox" checked onchange="saveReportOptions()">Signal types (WiFi/BLE/Thread/HaLow/etc)</label>
                            <label style="display:flex; align-items:center; gap:8px; font-size: 11px; color: var(--text-secondary);"><input id="chart-classification" type="checkbox" checked onchange="saveReportOptions()">Static vs Mobile vs Uncertain</label>
                            <label style="display:flex; align-items:center; gap:8px; font-size: 11px; color: var(--text-secondary);"><input id="chart-channels" type="checkbox" checked onchange="saveReportOptions()">Most detected channels (top N)</label>
                        </div>
                    </div>

                    <div class="settings-row">
                        <div>
                            <div class="filter-label">Channel chart: top N</div>
                            <div style="font-size: 11px; color: var(--text-secondary);">Groups remaining channels into “Other”. Default: 6</div>
                        </div>
                        <input id="chart-top-n-channels" type="number" min="3" max="12" value="6" style="width: 90px; padding: 6px 10px; border-radius: 6px; background: var(--surface-light); border: 1px solid var(--border-color); color: var(--text-primary);" onchange="saveReportOptions()">
                    </div>
                </div>
            </div>

            <div class="settings-pane" id="settings-pane-whitelist">
            <div style="margin-top: 14px;">
                <div class="filter-label">Hidden/Whitelisted SOIs</div>
                <div style="font-size: 11px; color: var(--text-secondary); margin-bottom: 8px;">These are ignored from static view and playback.</div>
                <div id="whitelist-list" style="max-height: 220px; overflow: auto; border: 1px solid var(--border-color); border-radius: 10px; padding: 10px;"></div>
                <div style="display: flex; gap: 10px; margin-top: 10px;">
                    <button class="action-btn" onclick="clearWhitelist()">Clear Whitelist</button>
                </div>
            </div>
        </div>

            <div class="settings-pane" id="settings-pane-about">
            <div style="margin-top: 14px;">
                <div class="filter-label">Program Version</div>
                <div style="font-size: 11px; color: var(--text-secondary); margin-bottom: 8px;">Use this to confirm the running build.</div>
                <div style="display:flex; flex-direction:column; gap:6px; padding:10px; border:1px solid var(--border-color); border-radius:10px; background: var(--surface-light);">
                    <div style="display:flex; gap:10px; align-items:center;">
                        <div style="min-width:90px; color: var(--text-secondary); font-size: 12px;">Version</div>
                        <div id="about-version" style="font-family: monospace; color: var(--text-primary); font-size: 13px;">—</div>
                    </div>
                    <div style="display:flex; gap:10px; align-items:center;">
                        <div style="min-width:90px; color: var(--text-secondary); font-size: 12px;">Build UTC</div>
                        <div id="about-build" style="font-family: monospace; color: var(--text-primary); font-size: 13px;">—</div>
                    </div>
                </div>
            </div>
            </div>

    </div>

    <script>
        // State
        let map = null;
        let markers = {};
        let circles = {};
        let socket = null;
        let currentView = 'map';
        let accessPoints = [];
        let selectedAP = null;

        // Table selection + search filters
        let tableSelection = new Set();
        let showOnlySelected = false;  // When true, map shows only selected SOIs
        let loadedRuns = [];  // Track loaded run metadata
        let currentSort = 'recent';  // Default sort order
        let skipNextFitBounds = false;  // Prevent zoom reset when deselecting
        let tableFilters = {
            q: '',
            type: 'all',
            class: 'all',
            channel: '',
            security: 'all',
            rssiMin: '',
            rssiMax: ''
        };

        // Worker visibility (checkbox + opacity per worker)
        let workerEnabled = {};
        let workerOpacity = {};

        // Settings
        let dimOthersOpacity = 0.25; // 25%
        let activeProfileId = null;
        let serverReportOptions = {};
        let serverProfiles = [];

        // Hidden/whitelist (ignored from static view and playback)
        // Stored as [{mac, name}] in localStorage (backward compatible with string arrays).
        let whitelistMap = new Map();
        (function loadWhitelist() {
            try {
                const raw = JSON.parse(localStorage.getItem('wa_whitelist') || '[]');
                if (Array.isArray(raw)) {
                    raw.forEach(item => {
                        if (!item) return;
                        if (typeof item === 'string') {
                            const k = String(item).toLowerCase();
                            if (k) whitelistMap.set(k, { mac: k, name: '' });
                        } else if (typeof item === 'object' && item.mac) {
                            const k = String(item.mac).toLowerCase();
                            whitelistMap.set(k, { mac: k, name: String(item.name || '') });
                        }
                    });
                }
            } catch (e) {
                whitelistMap = new Map();
            }
        })();

        // Selection overlays
        let selectionOverlay = {
            estMarker: null,
            radiusCircle: null,
            rings: []
        };

        // Playback state
        let playback = {
            open: false,
            active: false,
            playing: false,
            direction: 1,
            speed: 1.0,
            timer: null,
            detections: [],
            frames: [],
            frameMap: new Map(),
            frameIndex: 0,
            markers: {}
        };
        
        let filters = {
            classification: 'all',
            deviceType: 'all',
            security: 'all'
        };
        
        // Initialize
        document.addEventListener('DOMContentLoaded', () => {
            initMap();
            initSocket();
            initFilters();
            initDragDrop();
            restorePanelCollapseState();
            // Safety binding in case inline handlers are blocked or shadowed
            try {
                const pb = document.getElementById('btn-playback');
                if (pb) pb.addEventListener('click', (e) => { e.preventDefault(); togglePlaybackPopup(); });
            } catch (e) { /* no-op */ }
        });
        
        // Map initialization
        function initMap() {
            map = L.map('map', {
                center: [35.1225, -79.4567],
                zoom: 14,
                zoomControl: true
            });
            
            L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
                attribution: '&copy; CartoDB',
                maxZoom: 19
            }).addTo(map);
            
            // Click on map (not on marker) deselects all SOIs and exits selection mode
            map.on('click', (e) => {
                // Only deselect if click was on the map itself, not on a marker
                if (e.originalEvent && !e.originalEvent.target.closest('.leaflet-marker-icon')) {
                    selectedAP = null;
                    tableSelection.clear();
                    showOnlySelected = false;  // Exit selection-only mode
                    skipNextFitBounds = true;  // Prevent zoom reset when deselecting
                    
                    // Reset detail panel
                    const panel = document.getElementById('detail-panel');
                    if (panel) {
                        panel.innerHTML = `
                            <div class="detail-empty">
                                <div class="detail-empty-icon">👆</div>
                                <div>Click on a marker or table row</div>
                                <div style="font-size: 11px; margin-top: 4px;">to view AP details</div>
                            </div>
                        `;
                    }
                    
                    // Clear any overlay circles
                    if (window.focusCircle) {
                        map.removeLayer(window.focusCircle);
                        window.focusCircle = null;
                    }
                    if (window.strongCircle) {
                        map.removeLayer(window.strongCircle);
                        window.strongCircle = null;
                    }
                    
                    updateMarkers();
                    updateTable();
                    updateSelectionStats();
                    syncSelectAllCheckbox();
                }
            });
        }
        
        // Socket initialization
        function initSocket() {
            socket = io();
            
            socket.on('connect', () => {
                document.getElementById('status-dot').classList.add('connected');
                document.getElementById('status-text').textContent = 'Connected';
                socket.emit('request_stats');
                requestAPs();
            });
            
            socket.on('disconnect', () => {
                document.getElementById('status-dot').classList.remove('connected');
                document.getElementById('status-text').textContent = 'Disconnected';
            });
            
            socket.on('stats_update', (data) => {
                updateStats(data.stats);
            });
            
            socket.on('aps_data', (data) => {
                accessPoints = data.aps || [];
                // Refresh whitelist display names when new data arrives
                try {
                    let changed = false;
                    whitelistMap.forEach((v, k) => {
                        if (v && (!v.name || v.name === '')) {
                            const ap = accessPoints.find(a => String(a.id).toLowerCase() === k);
                            if (ap && ap.name) {
                                v.name = String(ap.name);
                                whitelistMap.set(k, v);
                                changed = true;
                            }
                        }
                    });
                    if (changed) persistWhitelist();
                } catch (e) {}
                rebuildWorkerOpacityUIFromData(accessPoints);
                updateMarkers();
                updateTable();
                document.getElementById('ap-count').textContent = `${data.count || accessPoints.length} APs loaded`;
            });

            socket.on('detections_data', (data) => {
                try {
                    const dets = (data && (data.detections || data.data || data.items)) || [];
                    playback.detections = Array.isArray(dets) ? dets : [];
                    buildPlaybackFrames();
                    updatePlaybackTimeLabel();
                    if (playback._autoplay && playback.frames && playback.frames.length > 0) {
                        const dir = playback._autoplay_dir || 1;
                        playback._autoplay = false;
                        playback._autoplay_dir = 1;
                        playback.direction = dir;
                        pbTogglePlay(true);
                    }
                } finally {
                    playback._loading = false;
                }
            });
            
            socket.on('analysis_complete', (data) => {
                updateStats(data.stats);
                requestAPs();
            });
            
            socket.on('data_cleared', () => {
                accessPoints = [];
                tableSelection.clear();
                showOnlySelected = false;
                loadedRuns = [];  // Clear loaded runs tracking
                updateMarkers();
                updateTable();
                updateStats({});
                
                // Clear loaded runs display
                const runsContainer = document.getElementById('loaded-runs');
                if (runsContainer) {
                    runsContainer.style.display = 'none';
                    const runsList = document.getElementById('runs-list');
                    if (runsList) runsList.innerHTML = '';
                }
            });
            
            socket.on('runs_removed', (data) => {
                console.log('Runs removed:', data);
                // Update stats
                if (data.stats) {
                    updateStats(data.stats);
                }
                // Request fresh AP data
                requestAPs();
            });
        }
        
        // Request APs with current filters
        function requestAPs() {
            socket.emit('request_aps', {
                classification: filters.classification,
                deviceType: filters.deviceType,
                includeNoGps: true
            });
        }
        
        // Update stats display
        function updateStats(stats) {
            if (!stats) stats = {};
            document.getElementById('stat-total').textContent = stats.total_aps || 0;
            document.getElementById('stat-detections').textContent = formatNumber(stats.total_detections || 0);
            document.getElementById('stat-static').textContent = stats.static || 0;
            document.getElementById('stat-mobile').textContent = stats.mobile || 0;
            document.getElementById('stat-uncertain').textContent = stats.uncertain || 0;
            document.getElementById('stat-with-gps').textContent = stats.with_gps || 0;
            
            // Show no-GPS warning if needed
            const noGpsWarning = document.getElementById('no-gps-warning');
            if (stats.total_aps > 0 && stats.with_gps === 0) {
                noGpsWarning.style.display = 'block';
                setView('table');
            } else {
                noGpsWarning.style.display = 'none';
            }
        }
        
        // Format large numbers
        function formatNumber(num) {
            if (num >= 1000000) return (num / 1000000).toFixed(1) + 'M';
            if (num >= 1000) return (num / 1000).toFixed(1) + 'k';
            return num.toString();
        }

        function formatMeters(m) {
            const val = Number(m || 0);
            if (val >= 1000) return (val / 1000).toFixed(2) + ' km';
            return Math.round(val) + ' m';
        }

        // ------------------- Worker visibility + whitelist -------------------
        function isWhitelisted(mac) {
            return whitelistMap.has(String(mac || '').toLowerCase());
        }

        function persistWhitelist() {
            localStorage.setItem('wa_whitelist', JSON.stringify(Array.from(whitelistMap.values())));
            renderWhitelistList();
            refreshVersionInfo();
        }

        function toggleWhitelist(mac, name) {
            const key = String(mac || '').toLowerCase();
            if (!key) return;

            if (whitelistMap.has(key)) {
                whitelistMap.delete(key);
            } else {
                // Try to capture a friendly name if not provided
                let friendly = String(name || '');
                if (!friendly) {
                    const ap = accessPoints.find(a => String(a.id).toLowerCase() === key);
                    if (ap && ap.name) friendly = String(ap.name);
                }
                whitelistMap.set(key, { mac: key, name: friendly });

                // If the selected AP was hidden, clear selection to avoid confusion
                if (selectedAP && String(selectedAP.id).toLowerCase() === key) {
                    selectedAP = null;
                    document.getElementById('detail-panel').innerHTML = `
                        <div class="detail-empty">
                            <div class="detail-empty-icon">👆</div>
                            <div>Click on a marker or table row</div>
                            <div style="font-size: 11px; margin-top: 4px;">to view AP details</div>
                        </div>
                    `;
                }
            }

            persistWhitelist();
            updateMarkers();
            updateTable();
        }

        function toggleWhitelistSelected() {
            if (!selectedAP) return;
            toggleWhitelist(selectedAP.id, selectedAP.name || '');
        }

        function clearWhitelist() {
            whitelistMap = new Map();
            persistWhitelist();
            updateMarkers();
            updateTable();
        }

        function getKnownWorkersFromData(aps) {
            const s = new Set();
            (aps || []).forEach(ap => {
                (ap.workers || []).forEach(w => s.add(String(w)));
                if (ap.worker) s.add(String(ap.worker));
            });
            return Array.from(s).filter(Boolean).sort();
        }

        function initWorkerOpacityUI(workers) {
            const list = document.getElementById('worker-opacity-list');
            if (!list) return;

            // Seed defaults (enabled + full opacity)
            (workers || []).forEach(w => {
                if (workerEnabled[w] === undefined) workerEnabled[w] = true;
                if (workerOpacity[w] === undefined) workerOpacity[w] = 1.0;
            });

            list.innerHTML = '';
            (workers || []).forEach(w => {
                const row = document.createElement('div');
                row.className = 'worker-row';

                const cb = document.createElement('input');
                cb.type = 'checkbox';
                cb.checked = !!workerEnabled[w];
                cb.addEventListener('change', () => {
                    workerEnabled[w] = cb.checked;
                    updateMarkers();
                    updateTable();
                });

                const tag = document.createElement('div');
                tag.className = 'worker-tag';
                tag.textContent = w;

                const slider = document.createElement('input');
                slider.type = 'range';
                slider.min = '0';
                slider.max = '100';
                slider.step = '1';
                slider.value = Math.round((workerOpacity[w] || 1.0) * 100);

                const label = document.createElement('div');
                label.className = 'worker-alpha';
                label.textContent = `${slider.value}%`;

                slider.addEventListener('input', () => {
                    workerOpacity[w] = Math.max(0, Math.min(1, Number(slider.value) / 100));
                    label.textContent = `${slider.value}%`;
                    updateMarkers();
                    updateTable();
                });

                row.appendChild(cb);
                row.appendChild(tag);
                row.appendChild(slider);
                row.appendChild(label);
                list.appendChild(row);
            });
        }

        function rebuildWorkerOpacityUIFromData(aps) {
            const workers = getKnownWorkersFromData(aps);
            if (workers.length === 0) return;
            initWorkerOpacityUI(workers);
        }

        function getEffectiveOpacityForAP(ap) {
            // Use max opacity of any contributing worker
            const ws = (ap.workers && ap.workers.length > 0) ? ap.workers : [ap.worker || 'unknown'];
            let a = 0.0;
            ws.forEach(w => {
                const key = String(w || 'unknown');
                if (workerEnabled[key] === false) return;
                a = Math.max(a, workerOpacity[key] !== undefined ? workerOpacity[key] : 1.0);
            });

            // Apply "dim others" when one AP is selected
            if (selectedAP && ap.id !== selectedAP.id) {
                a = a * dimOthersOpacity;
            }
            return a;
        }

        function _canonSecurity(v) {
            const s = String(v || '').toUpperCase().trim();
            if (!s) return 'UNKNOWN';
            if (s === 'WPA2/WPA3') return 'WPA2WPA3';
            return s.replace(/[\s\/]+/g, '');
        }

        function passesSecurityFilter(ap) {
            const f = String(filters.security || 'all').toLowerCase();
            if (f === 'all') return true;
            const dev = String(ap.device_category || ap.type || '').toLowerCase();
            if (dev !== 'wifi') return true; // security filter applies to WiFi only
            const apSec = _canonSecurity(ap.security);
            // Map UI values to canonical
            const wantMap = {
                open: 'OPEN',
                owe: 'OWE',
                wep: 'WEP',
                wpa1: 'WPA1',
                wpa2: 'WPA2',
                wpa3: 'WPA3',
                wpa2wpa3: 'WPA2WPA3',
                unknown: 'UNKNOWN'
            };
            const want = wantMap[f] || String(f).toUpperCase();
            return apSec === want;
        }

        // ------------------- Settings modal -------------------
        async function refreshServerSettings(populateUI = false) {
            try {
                const [settingsRes, profilesRes] = await Promise.all([
                    fetch('/api/settings'),
                    fetch('/api/profiles')
                ]);
                const settings = await settingsRes.json();
                const profilesPayload = await profilesRes.json();
                activeProfileId = settings.active_profile_id || activeProfileId;
                serverReportOptions = settings.report || {};
                serverProfiles = (profilesPayload.profiles || []).slice();
                if (populateUI) {
                    renderProfileList();
                    applyReportOptionsToUI();
                }
            } catch (e) {
                console.warn('Failed to load server settings/profiles', e);
            }
        }

        function renderProfileList() {
            const c = document.getElementById('profile-list');
            if (!c) return;
            if (!Array.isArray(serverProfiles) || serverProfiles.length === 0) {
                c.innerHTML = '<div style="color: var(--text-secondary); font-size: 12px;">No profiles available.</div>';
                return;
            }
            c.innerHTML = '';
            serverProfiles.forEach(p => {
                const id = String(p.id || '');
                const name = String(p.name || id || 'Profile');
                const desc = String(p.description || '');
                const row = document.createElement('label');
                row.style.display = 'flex';
                row.style.flexDirection = 'column';
                row.style.gap = '4px';
                row.style.padding = '8px 10px';
                row.style.border = '1px solid var(--border-color)';
                row.style.borderRadius = '10px';
                row.style.marginBottom = '8px';
                row.style.cursor = 'pointer';
                const checked = (activeProfileId && id && String(activeProfileId) === id) ? 'checked' : '';
                row.innerHTML = `
                    <div style="display:flex; align-items:center; gap:10px;">
                        <input type="radio" name="csv-profile" value="${id}" ${checked} onchange="setActiveProfile(this.value)">
                        <div style="color: var(--text-primary); font-weight: 600; font-size: 13px;">${name}</div>
                        <div style="margin-left:auto; font-family: monospace; font-size: 11px; color: var(--text-secondary);">${id}</div>
                    </div>
                    <div style="color: var(--text-secondary); font-size: 11px;">${desc}</div>
                `;
                c.appendChild(row);
            });
        }

        async function setActiveProfile(profileId) {
            activeProfileId = String(profileId || '') || null;
            try {
                await fetch('/api/settings', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ active_profile_id: activeProfileId })
                });
            } catch (e) {
                console.warn('Failed to persist active profile', e);
            }
        }

        function triggerProfileUpload() {
            const inp = document.getElementById('profile-upload');
            if (inp) inp.click();
        }

        async function uploadProfileJson(file) {
            if (!file) return;
            try {
                const fd = new FormData();
                fd.append('file', file);
                const resp = await fetch('/api/profiles/upload', { method: 'POST', body: fd });
                const out = await resp.json();
                if (!resp.ok) throw new Error(out.error || 'Upload failed');
                await refreshServerSettings(true);
            refreshVersionInfo();
            } catch (e) {
                alert('Profile upload failed: ' + (e && e.message ? e.message : String(e)));
            }
        }

        function applyReportOptionsToUI() {
            const r = serverReportOptions || {};
            const setVal = (id, v) => { const el = document.getElementById(id); if (el) el.value = v; };
            const setChk = (id, v) => { const el = document.getElementById(id); if (el) el.checked = !!v; };
            setVal('pdf-font-family', r.pdf_font_family || 'Helvetica');
            setVal('pdf-body-size', r.pdf_body_font_size || 10);
            setVal('pdf-title-size', r.pdf_title_font_size || 24);
            setVal('html-font-family', r.html_font_family || 'system-ui');
            setVal('html-font-size', r.html_font_size || 14);
            setChk('html-title-bold', r.html_title_bold !== false);
            setVal('company-name', r.company_name || '');
            setVal('company-font-family', r.company_font_family || (r.pdf_font_family || 'Helvetica'));
            setVal('company-font-size', r.company_font_size || 12);
            setChk('company-bold', !!r.company_bold);
            setChk('watermark-enabled', !!r.watermark_enabled);
            const wmOp = Math.round(Number(r.watermark_opacity || 0.08) * 100);
            setVal('watermark-opacity', wmOp);
            const wmv = document.getElementById('watermark-opacity-label');
            if (wmv) wmv.textContent = String(wmOp) + '%';
            const fn = document.getElementById('watermark-status');
            if (fn) fn.textContent = r.watermark_filename ? ('Logo: ' + String(r.watermark_filename)) : 'No file';

            // Charts
            setChk('charts-pdf-enabled', r.charts_pdf_enabled !== false);
            setChk('charts-html-enabled', r.charts_html_enabled !== false);
            setChk('chart-signal-types', r.chart_signal_types !== false);
            setChk('chart-classification', r.chart_classification !== false);
            setChk('chart-channels', r.chart_channels !== false);
            setVal('chart-top-n-channels', r.chart_top_n_channels || 6);
            // Route map
            setChk('route-map-enabled', r.route_map_enabled !== false);
            setChk('route-map-use-basemap', r.route_map_use_basemap !== false);
            setVal('route-map-padding', Math.round(Number((r.route_map_padding_pct || 0.10) * 100)));

        }

                function updateWatermarkOpacityLabel(val) {
            const v = Math.max(0, Math.min(100, Number(val)));
            const el = document.getElementById('watermark-opacity-label');
            if (el) el.textContent = String(v) + '%';
        }

async function saveReportOptions() {
            const getVal = (id) => { const el = document.getElementById(id); return el ? el.value : ''; };
            const getChk = (id) => { const el = document.getElementById(id); return el ? !!el.checked : false; };
            const wmOpPct = Math.max(0, Math.min(100, Number(getVal('watermark-opacity') || 8)));
            const payload = {
                report: {
                    pdf_font_family: String(getVal('pdf-font-family') || 'Helvetica'),
                    pdf_body_font_size: Number(getVal('pdf-body-size') || 10),
                    pdf_title_font_size: Number(getVal('pdf-title-size') || 24),
                    html_font_family: String(getVal('html-font-family') || 'system-ui'),
                    html_font_size: Number(getVal('html-font-size') || 14),
                    html_title_bold: getChk('html-title-bold'),
                    company_name: String(getVal('company-name') || ''),
                    company_font_family: String(getVal('company-font-family') || getVal('pdf-font-family') || 'Helvetica'),
                    company_font_size: Number(getVal('company-font-size') || 12),
                    company_bold: getChk('company-bold'),
                    watermark_enabled: getChk('watermark-enabled'),
                    watermark_opacity: wmOpPct / 100.0,

                    charts_pdf_enabled: getChk('charts-pdf-enabled'),
                    charts_html_enabled: getChk('charts-html-enabled'),
                    chart_signal_types: getChk('chart-signal-types'),
                    chart_classification: getChk('chart-classification'),
                    chart_channels: getChk('chart-channels'),
                    chart_top_n_channels: Math.max(3, Math.min(12, Number(getVal('chart-top-n-channels') || 6))),
                    route_map_enabled: getChk('route-map-enabled'),
                    route_map_use_basemap: getChk('route-map-use-basemap'),
                    route_map_padding_pct: Math.max(0, Math.min(0.50, Number(getVal('route-map-padding') || 10) / 100.0))

                }
            };
            try {
                await fetch('/api/settings', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                // Keep local copy in sync
                serverReportOptions = Object.assign({}, serverReportOptions || {}, payload.report);
            } catch (e) {
                console.warn('Failed to persist report options', e);
            }
        }

        function triggerWatermarkUpload() {
            const inp = document.getElementById('watermark-upload');
            if (inp) inp.click();
        }

        async function uploadWatermarkFile(file) {
            if (!file) return;
            try {
                const fd = new FormData();
                fd.append('file', file);
                const resp = await fetch('/api/report/watermark', { method: 'POST', body: fd });
                const out = await resp.json();
                if (!resp.ok) throw new Error(out.error || 'Upload failed');
                await refreshServerSettings(true);
            refreshVersionInfo();
            } catch (e) {
                alert('Watermark upload failed: ' + (e && e.message ? e.message : String(e)));
            }
        }
        function loadSettings() {
            const raw = localStorage.getItem('wa_dim_others');
            if (raw !== null) {
                const v = Math.max(0, Math.min(100, Number(raw)));
                dimOthersOpacity = v / 100;
                const el = document.getElementById('dim-others-slider');
                if (el) el.value = String(v);
                const valEl = document.getElementById('dim-others-value');
                if (valEl) valEl.textContent = String(v);
            }
            // Load persisted server-side settings (profiles/report options)
            try { refreshServerSettings(false); } catch (e) {}
        }

        function setDimOthersOpacity(val) {
            const v = Math.max(0, Math.min(100, Number(val)));
            dimOthersOpacity = v / 100;
            localStorage.setItem('wa_dim_others', String(v));
            const valEl = document.getElementById('dim-others-value');
            if (valEl) valEl.textContent = String(v);
            updateMarkers();
            updateTable();
        }

        function showSettingsPane(name) {
            const panes = ['general','report','charts','whitelist','about'];
            const n = (name && panes.includes(String(name))) ? String(name) : 'general';
            panes.forEach(p => {
                const paneEl = document.getElementById('settings-pane-' + p);
                if (paneEl) paneEl.classList.toggle('active', p === n);
                const tabEl = document.getElementById('settings-tab-' + p);
                if (tabEl) tabEl.classList.toggle('active', p === n);
            });
            try { localStorage.setItem('wa_settings_pane', n); } catch (e) {}
        }

        async function openSettings() {
            // Show modal first so tabs/panes exist in DOM
            document.getElementById('settings-modal').style.display = 'flex';
            // Restore last-opened pane (avoids requiring scrolling on trackpads)
            let last = 'general';
            try { last = localStorage.getItem('wa_settings_pane') || 'general'; } catch (e) {}
            showSettingsPane(last);
            renderWhitelistList();
            await refreshServerSettings(true);
            refreshVersionInfo();
        }

        function closeSettings() {
            document.getElementById('settings-modal').style.display = 'none';
        }

        function closeSettingsFromBackdrop(e) {
            if (e && e.target && e.target.id === 'settings-modal') closeSettings();
        }

        
        async function refreshVersionInfo() {
            try {
                const resp = await fetch('/api/version');
                const out = await resp.json();
                const vEl = document.getElementById('about-version');
                const bEl = document.getElementById('about-build');
                if (vEl) vEl.textContent = out.version || '—';
                if (bEl) bEl.textContent = out.build_utc || '—';
            } catch (e) {
                // ignore
            }
        }

function renderWhitelistList() {
            const c = document.getElementById('whitelist-list');
            if (!c) return;
            const items = Array.from(whitelistMap.values());
            if (items.length === 0) {
                c.innerHTML = `<div style="color: var(--text-secondary); font-size: 12px;">No hidden items.</div>`;
                return;
            }
            c.innerHTML = '';
            items.sort((a, b) => String(a.mac || '').localeCompare(String(b.mac || '')));
            items.forEach(item => {
                const mac = String(item.mac || '');
                const name = String(item.name || '');
                const row = document.createElement('div');
                row.style.display = 'flex';
                row.style.justifyContent = 'space-between';
                row.style.alignItems = 'center';
                row.style.gap = '10px';
                row.style.padding = '6px 0';
                row.innerHTML = `
                    <div style="display:flex; flex-direction:column; gap:2px;">
                        <div style="color: var(--text-primary); font-size: 12px;">${name || '(no name)'}</div>
                        <div style="font-family: monospace; color: var(--text-secondary); font-size: 11px;">${mac}</div>
                    </div>
                    <button class="pb-btn" onclick="toggleWhitelist('${mac}')">Remove</button>
                `;
                c.appendChild(row);
            });
        }

// ------------------- Layer helpers -------------------
        function clearStaticLayers() {
            Object.values(markers).forEach(m => m.remove());
            Object.values(circles).forEach(c => c.remove());
            markers = {};
            circles = {};
            clearSelectionOverlay();
        }

        function clearPlaybackLayers() {
            Object.values(playback.markers || {}).forEach(m => m.remove());
            playback.markers = {};
        }

        // ------------------- Selection overlay -------------------
        function clearSelectionOverlay() {
            if (selectionOverlay.estMarker) {
                selectionOverlay.estMarker.remove();
                selectionOverlay.estMarker = null;
            }
            if (selectionOverlay.radiusCircle) {
                selectionOverlay.radiusCircle.remove();
                selectionOverlay.radiusCircle = null;
            }
            // Remove any heatmap rings
            if (selectionOverlay.rings && selectionOverlay.rings.length) {
                selectionOverlay.rings.forEach(r => { try { r.remove(); } catch (e) {} });
            }
            selectionOverlay.rings = [];
        }

        function drawSelectionOverlay(ap) {
            clearSelectionOverlay();
            if (!ap) return;

            const strongLat = Number(ap.strong_lat || ap.est_lat || ap.lat || 0);
            const strongLon = Number(ap.strong_lon || ap.est_lon || ap.lon || 0);
            if (!strongLat || !strongLon) return;

            // Color keyed to SOI type
            const type = String(ap.device_category || ap.type || '').toLowerCase();
            const typeColor = (function() {
                const m = {
                    wifi: '#3b82f6',
                    ble: '#ec4899',
                    bluetooth: '#ec4899',
                    thread: '#f59e0b',
                    zigbee: '#f59e0b',
                    matter: '#22c55e',
                    halow: '#a3e635'
                };
                return m[type] || '#93c5fd';
            })();

            // Center marker = strongest observed point
            const icon = L.divIcon({
                html: `<div style="
                    width: 18px;
                    height: 18px;
                    border: 2px dashed ${typeColor};
                    border-radius: 50%;
                    background: rgba(255,255,255,0.06);
                    box-shadow: 0 0 14px ${typeColor}55;
                "></div>`,
                className: '',
                iconSize: [18, 18],
                iconAnchor: [9, 9]
            });
            selectionOverlay.estMarker = L.marker([strongLat, strongLon], { icon, opacity: 0.95 }).addTo(map);

            selectionOverlay.rings = [];

            const strongRssi = (ap.strong_rssi !== undefined && ap.strong_rssi !== null) ? Number(ap.strong_rssi) : null;

            // A1: Strongest point region (smallest/high-confidence radius)
            const strongRegionR = Number(ap.strong_region_radius_m || 0);
            if (strongRegionR > 0) {
                const ring = L.circle([strongLat, strongLon], {
                    radius: strongRegionR,
                    color: typeColor,
                    weight: 2,
                    opacity: 0.85,
                    fillColor: typeColor,
                    fillOpacity: 0.10
                }).addTo(map);
                const th = Number(ap.strong_region_threshold_db || 6);
                ring.bindTooltip(
                    `<div style="font-size:12px;"><b>Strongest point</b><br>` +
                    `${strongRssi !== null ? `Strongest RSSI: ${strongRssi.toFixed(0)} dBm<br>` : ''}` +
                    `Strong-signal region (within ${th} dB): radius ${Math.round(strongRegionR)} m</div>`,
                    { sticky: true, direction: 'top', opacity: 0.95, className: 'wa-tooltip' }
                );
                selectionOverlay.rings.push(ring);
            }

            // A2: Weighted ellipsoid of most-likely physical location
            const eLat = Number(ap.ellipse_center_lat || ap.est_lat || strongLat);
            const eLon = Number(ap.ellipse_center_lon || ap.est_lon || strongLon);
            const a = Number(ap.ellipse_a_m || 0);
            const b = Number(ap.ellipse_b_m || 0);
            const ang = Number(ap.ellipse_angle_deg || 0);
            if (a > 0 && b > 0 && eLat && eLon) {
                const pts = [];
                const n = 64;
                const lat0 = eLat * Math.PI / 180;
                const mPerDegLat = 111111.0;
                const mPerDegLon = 111111.0 * Math.cos(lat0);
                const rot = ang * Math.PI / 180;
                for (let i = 0; i < n; i++) {
                    const t = (i / n) * Math.PI * 2;
                    let x = a * Math.cos(t); // east
                    let y = b * Math.sin(t); // north
                    // rotate
                    const xr = x * Math.cos(rot) - y * Math.sin(rot);
                    const yr = x * Math.sin(rot) + y * Math.cos(rot);
                    const latP = eLat + (yr / mPerDegLat);
                    const lonP = eLon + (xr / mPerDegLon);
                    pts.push([latP, lonP]);
                }
                const poly = L.polygon(pts, {
                    color: typeColor,
                    weight: 2,
                    opacity: 0.45,
                    fillColor: typeColor,
                    fillOpacity: 0.14
                }).addTo(map);
                poly.bindTooltip(
                    `<div style="font-size:12px;"><b>Most likely location area</b><br>` +
                    `RSSI-weighted ellipse (≈2σ)<br>` +
                    `Major axis: ${Math.round(a)} m, Minor axis: ${Math.round(b)} m<br>` +
                    `Angle: ${Math.round(ang)}°<br>` +
                    `${strongRssi !== null ? `Strongest heard: ${strongRssi.toFixed(0)} dBm` : ''}</div>`,
                    { sticky: true, direction: 'top', opacity: 0.95, className: 'wa-tooltip' }
                );
                selectionOverlay.rings.push(poly);
            }

            // A3: First seen / last seen ring (dotted)
            const flR = Number(ap.first_last_radius_m || 0);
            if (flR > 0) {
                const ring = L.circle([strongLat, strongLon], {
                    radius: flR,
                    color: typeColor,
                    weight: 2,
                    opacity: 0.70,
                    fillOpacity: 0,
                    dashArray: '6,6'
                }).addTo(map);

                const firstTs = ap.first_seen_ts || ap.first_seen || '';
                const lastTs = ap.last_seen_ts || ap.last_seen || '';
                const firstRssi = (ap.first_seen_rssi !== undefined && ap.first_seen_rssi !== null) ? Number(ap.first_seen_rssi) : null;
                const lastRssi = (ap.last_seen_rssi !== undefined && ap.last_seen_rssi !== null) ? Number(ap.last_seen_rssi) : null;

                ring.bindTooltip(
                    `<div style="font-size:12px;"><b>First/Last heard boundary</b><br>` +
                    `Radius from strongest point: ${Math.round(flR)} m<br>` +
                    `${firstTs ? `First seen: ${firstTs}${firstRssi !== null ? ` @ ${firstRssi.toFixed(0)} dBm` : ''}<br>` : ''}` +
                    `${lastTs ? `Last seen: ${lastTs}${lastRssi !== null ? ` @ ${lastRssi.toFixed(0)} dBm` : ''}` : ''}` +
                    `</div>`,
                    { sticky: true, direction: 'top', opacity: 0.95, className: 'wa-tooltip' }
                );
                selectionOverlay.rings.push(ring);
                selectionOverlay.radiusCircle = ring;
            }
        }

        function centerOnSelected() {
            if (!selectedAP) return;
            const lat = selectedAP.strong_lat || selectedAP.est_lat || selectedAP.lat;
            const lon = selectedAP.strong_lon || selectedAP.est_lon || selectedAP.lon;
            if (!lat || !lon) return;
            map.setView([lat, lon], 17);
        }

        // ------------------- Playback -------------------
        function togglePlaybackPopup() {
            const dock = document.getElementById('playback-dock');
            if (!dock) return;
            const willCollapse = !dock.classList.contains('collapsed') ? true : false;
            if (willCollapse) {
                dock.classList.add('collapsed');
                playback.open = false;
                try { pbStop(); } catch (e) { /* no-op */ }
            } else {
                dock.classList.remove('collapsed');
                playback.open = true;
                try { ensureDetectionsLoaded(); } catch (e) { /* no-op */ }
            }
        }

        function ensureDetectionsLoaded() {
            if (playback._loading) return;
            if (playback.detections && playback.detections.length > 0) {
                if (!playback.frames || playback.frames.length === 0) buildPlaybackFrames();
                return;
            }

            playback._loading = true;

            // Prefer socket request if available, otherwise fall back to REST.
            if (socket && socket.connected) {
                socket.emit('request_detections', { includeNoGps: false });
                // socket handler will clear _loading
                return;
            }

            fetch('/api/data/detections?include_no_gps=false')
                .then(r => r.json())
                .then(payload => {
                    const dets = payload.detections || payload.data || payload.items || [];
                    playback.detections = Array.isArray(dets) ? dets : [];
                    buildPlaybackFrames();
                    updatePlaybackTimeLabel();
                    // Auto-start if requested
                    if (playback._autoplay && playback.frames && playback.frames.length > 0) {
                        const dir = playback._autoplay_dir || 1;
                        playback._autoplay = false;
                        playback._autoplay_dir = 1;
                        playback.direction = dir;
                        pbTogglePlay(true);
                    }
                })
                .catch(() => {
                    playback.detections = [];
                    buildPlaybackFrames();
                    updatePlaybackTimeLabel();
                })
                .finally(() => {
                    playback._loading = false;
                });
        }


        function buildPlaybackFrames() {
            playback.frames = [];
            playback.frameMap = new Map();

            // Group detections into 1-second frames
            const groups = new Map();
            (playback.detections || []).forEach(d => {
                if (!d || !d.ts_ms) return;
                const t = Math.floor(d.ts_ms / 1000) * 1000;
                if (!groups.has(t)) groups.set(t, []);
                groups.get(t).push(d);
            });
            playback.frames = Array.from(groups.keys()).sort((a,b) => a - b);
            playback.frames.forEach(t => playback.frameMap.set(t, groups.get(t)));
            playback.frameIndex = 0;
        }

        function updatePlaybackTimeLabel() {
            const el = document.getElementById('pb-time');
            if (!el) return;
            if (!playback.frames || playback.frames.length === 0) {
                el.textContent = 'Playback: (no GPS detections)';
                return;
            }
            const t = playback.frames[playback.frameIndex] || playback.frames[0];
            const dt = new Date(t);
            el.textContent = `Playback: ${dt.toISOString()}`;
        }

        function enterPlaybackMode() {
            if (playback.active) return;
            playback.active = true;
            selectedAP = null;
            clearStaticLayers();
            clearPlaybackLayers();
            updatePlaybackTimeLabel();
        }

        function exitPlaybackMode() {
            playback.active = false;
            playback.playing = false;
            if (playback.timer) {
                clearInterval(playback.timer);
                playback.timer = null;
            }
            clearPlaybackLayers();
            updateMarkers();
            updateTable();
        }

        function renderPlaybackFrame() {
            if (!playback.active) return;
            clearPlaybackLayers();
            if (!playback.frames || playback.frames.length === 0) return;

            const t = playback.frames[playback.frameIndex];
            const frame = playback.frameMap.get(t) || [];

            // Condense to one marker per MAC for the frame (pick strongest RSSI)
            const byMac = new Map();
            frame.forEach(d => {
                const mac = String(d.mac || '').toLowerCase();
                if (!mac) return;
                if (isWhitelisted(mac)) return;
                const apMeta = accessPoints.find(a => String(a.id).toLowerCase() === mac);
                if (!apMeta) return; // honor current filters
                if (!passesSecurityFilter(apMeta)) return;

                // honor worker settings
                const w = String(d.worker || 'unknown');
                if (workerEnabled[w] === false) return;
                const baseA = workerOpacity[w] !== undefined ? workerOpacity[w] : 1.0;
                if (baseA <= 0.01) return;

                const prev = byMac.get(mac);
                if (!prev || (d.rssi || -999) > (prev.rssi || -999)) {
                    byMac.set(mac, { d, apMeta, alpha: baseA });
                }
            });

            byMac.forEach(({d, apMeta, alpha}, mac) => {
                const color = apMeta.classification === 'static' ? '#10b981' : 
                             apMeta.classification === 'mobile' ? '#f59e0b' : '#ef4444';
                const icon = L.divIcon({
                    html: `<div style="
                        width: 16px;
                        height: 16px;
                        background: ${color};
                        border: 2px solid rgba(255,255,255,0.28);
                        border-radius: 50%;
                        box-shadow: 0 0 10px ${color}80;
                    "></div>`,
                    className: '',
                    iconSize: [16, 16],
                    iconAnchor: [8, 8]
                });
                const m = L.marker([d.lat, d.lon], { icon, opacity: alpha }).addTo(map);
                m.on('click', () => showAPDetail(apMeta));
                playback.markers[mac] = m;
            });

            updatePlaybackTimeLabel();
        }

        function pbSetSpeed(val) {
            playback.speed = Math.max(0.25, Math.min(10, Number(val)));
            document.getElementById('pb-speed-label').textContent = playback.speed.toFixed(2) + 'x';
            if (playback.playing) {
                // Restart timer with new interval
                pbStartTimer();
            }
        }

        function pbStartTimer() {
            if (playback.timer) {
                clearInterval(playback.timer);
                playback.timer = null;
            }
            const interval = Math.max(30, Math.floor(500 / playback.speed));
            playback.timer = setInterval(() => {
                pbStep(playback.direction);
            }, interval);
        }

        function pbStep(dir) {
            if (!playback.frames || playback.frames.length === 0) return;
            playback.frameIndex += dir;
            if (playback.frameIndex < 0) playback.frameIndex = 0;
            if (playback.frameIndex >= playback.frames.length) {
                playback.frameIndex = playback.frames.length - 1;
                pbTogglePlay(false);
            }
            renderPlaybackFrame();
        }

        function pbTogglePlay(force) {
            if (!playback.frames || playback.frames.length === 0) {
                playback._autoplay = true;
                playback._autoplay_dir = playback.direction || 1;
                ensureDetectionsLoaded();
                return;
            }
            enterPlaybackMode();
            if (typeof force === 'boolean') {
                playback.playing = force;
            } else {
                playback.playing = !playback.playing;
            }
            document.getElementById('pb-playpause').textContent = playback.playing ? '⏸️' : '▶️';

            if (playback.playing) {
                pbStartTimer();
                renderPlaybackFrame();
            } else {
                if (playback.timer) {
                    clearInterval(playback.timer);
                    playback.timer = null;
                }
            }
        }

        function pbStop() {
            // Stop playback and reset to start
            if (playback.timer) {
                clearInterval(playback.timer);
                playback.timer = null;
            }
            playback.playing = false;
            playback.frameIndex = 0;
            document.getElementById('pb-playpause').textContent = '▶️';
            if (playback.active) {
                exitPlaybackMode();
            } else {
                updatePlaybackTimeLabel();
            }
        }

        function pbPlayForward() {
            playback.direction = 1;
            pbTogglePlay(true);
        }

        function pbPlayBackward() {
            playback.direction = -1;
            pbTogglePlay(true);
        }

        function pbJumpStart() {
            if (!playback.frames || playback.frames.length === 0) return;
            enterPlaybackMode();
            playback.frameIndex = 0;
            renderPlaybackFrame();
        }

        function pbJumpEnd() {
            if (!playback.frames || playback.frames.length === 0) return;
            enterPlaybackMode();
            playback.frameIndex = playback.frames.length - 1;
            renderPlaybackFrame();
        }
        
        // Update map markers
        function updateMarkers() {
            // Playback mode owns the map rendering
            if (playback.active) {
                renderPlaybackFrame();
                return;
            }

            // Clear existing
            clearStaticLayers();

            // Add new markers for APs with GPS
            let apsWithGps = accessPoints.filter(ap => ap.lat && ap.lon && ap.lat !== 0 && ap.lon !== 0);
            
            // If showOnlySelected mode is active, filter to only selected SOIs
            if (showOnlySelected && tableSelection.size > 0) {
                apsWithGps = apsWithGps.filter(ap => tableSelection.has(String(ap.id).toLowerCase()));
            }

            apsWithGps.forEach(ap => {
                if (isWhitelisted(ap.id)) return;
                if (!passesSecurityFilter(ap)) return;

                const alpha = getEffectiveOpacityForAP(ap);
                if (alpha <= 0.01) return;

                const color = ap.classification === 'static' ? '#10b981' : 
                             ap.classification === 'mobile' ? '#f59e0b' : '#ef4444';

                // Create marker
                const icon = L.divIcon({
                    html: `<div style="
                        width: 20px;
                        height: 20px;
                        background: ${color};
                        border: 2px solid rgba(255,255,255,0.3);
                        border-radius: 50%;
                        box-shadow: 0 0 10px ${color}80;
                    "></div>`,
                    className: '',
                    iconSize: [20, 20],
                    iconAnchor: [10, 10]
                });

                const marker = L.marker([ap.lat, ap.lon], { icon, opacity: alpha }).addTo(map);
                marker.on('click', () => showAPDetail(ap));
                markers[ap.id] = marker;

                // Create coverage circle
                const circle = L.circle([ap.lat, ap.lon], {
                    radius: ap.coverage || 50,
                    color: color,
                    weight: 1,
                    opacity: 0.3 * alpha,
                    fillColor: color,
                    fillOpacity: 0.1 * alpha
                }).addTo(map);
                circles[ap.id] = circle;
            });

            // Fit bounds if we have markers (only when no single selection and not skipping)
            if (!selectedAP && apsWithGps.length > 0 && !showOnlySelected && !skipNextFitBounds) {
                const bounds = L.latLngBounds(apsWithGps.map(ap => [ap.lat, ap.lon]));
                map.fitBounds(bounds, { padding: [50, 50] });
            }
            skipNextFitBounds = false;  // Reset the flag after use

            // Draw selection overlay (estimated location + observed radius)
            drawSelectionOverlay(selectedAP);
            
            // Update the selection mode indicator
            updateSelectionModeIndicator();
        }
        
        // Show/hide the "showing selected only" indicator
        function updateSelectionModeIndicator() {
            let indicator = document.getElementById('selection-mode-indicator');
            if (showOnlySelected && tableSelection.size > 0) {
                if (!indicator) {
                    indicator = document.createElement('div');
                    indicator.id = 'selection-mode-indicator';
                    indicator.style.cssText = 'position: absolute; top: 10px; left: 50%; transform: translateX(-50%); z-index: 1000; background: var(--accent-blue); color: white; padding: 8px 16px; border-radius: 20px; font-size: 12px; display: flex; align-items: center; gap: 10px;';
                    indicator.innerHTML = `
                        <span>Showing ${tableSelection.size} selected SOI(s)</span>
                        <button onclick="showAllSOIs()" style="background: white; color: var(--accent-blue); border: none; padding: 4px 8px; border-radius: 4px; cursor: pointer; font-size: 11px;">Show All</button>
                    `;
                    document.getElementById('map').appendChild(indicator);
                } else {
                    indicator.querySelector('span').textContent = `Showing ${tableSelection.size} selected SOI(s)`;
                    indicator.style.display = 'flex';
                }
            } else if (indicator) {
                indicator.style.display = 'none';
            }
        }
        
        // ------------------- Table helpers -------------------
        function apPassesWorkerEnableFilter(ap) {
            const ws = (ap.workers && ap.workers.length) ? ap.workers : [ap.worker || 'unknown'];
            for (const w of ws) {
                const key = String(w || 'unknown');
                if (workerEnabled[key] !== false) return true;
            }
            return false;
        }

        function setTableFilter(key, val) {
            if (!(key in tableFilters)) return;
            tableFilters[key] = (val === undefined || val === null) ? '' : String(val);
            updateTable();
        }
        
        // Map search functionality
        let mapSearchTimeout = null;
        
        function handleMapSearch(query) {
            clearTimeout(mapSearchTimeout);
            const resultsDiv = document.getElementById('map-search-results');
            
            if (!query || query.trim().length < 2) {
                resultsDiv.style.display = 'none';
                return;
            }
            
            // Debounce search
            mapSearchTimeout = setTimeout(() => {
                const q = query.trim().toLowerCase();
                const matches = accessPoints.filter(ap => {
                    const name = (ap.name || '').toLowerCase();
                    const mac = (ap.mac || ap.id || '').toLowerCase();
                    return name.includes(q) || mac.includes(q);
                }).slice(0, 10);  // Limit to 10 results
                
                if (matches.length === 0) {
                    resultsDiv.innerHTML = '<div class="map-search-item" style="color: var(--text-muted);">No matches found</div>';
                    resultsDiv.style.display = 'block';
                    return;
                }
                
                resultsDiv.innerHTML = matches.map(ap => {
                    const classColor = ap.classification === 'static' ? 'var(--accent-green)' :
                                      ap.classification === 'mobile' ? 'var(--accent-orange)' : 'var(--accent-red)';
                    return `
                        <div class="map-search-item" onclick="selectAPFromSearch('${ap.id}')">
                            <div class="name">${ap.name || '(Hidden)'}</div>
                            <div class="mac">${ap.mac}</div>
                            <div class="meta">
                                <span style="color: ${classColor}">${(ap.classification || 'unknown').toUpperCase()}</span>
                                &bull; ${ap.rssi_mean?.toFixed(0) || '-'} dBm
                                &bull; ${ap.detections || 0} detections
                            </div>
                        </div>
                    `;
                }).join('');
                resultsDiv.style.display = 'block';
            }, 150);
        }
        
        function selectAPFromSearch(apId) {
            const ap = accessPoints.find(a => String(a.id).toLowerCase() === String(apId).toLowerCase());
            if (ap) {
                showAPDetail(ap);
                
                // Zoom to AP if it has GPS
                if (ap.lat && ap.lon && ap.lat !== 0 && ap.lon !== 0) {
                    map.setView([ap.lat, ap.lon], 17);
                }
            }
            
            // Clear and hide search
            document.getElementById('map-search').value = '';
            document.getElementById('map-search-results').style.display = 'none';
        }
        
        // Close search results when clicking outside
        document.addEventListener('click', (e) => {
            const searchBar = document.getElementById('map-search-bar');
            if (searchBar && !searchBar.contains(e.target)) {
                document.getElementById('map-search-results').style.display = 'none';
            }
        });

        function clearTableSelection() {
            tableSelection = new Set();
            const selAll = document.getElementById('tb-select-all');
            if (selAll) selAll.checked = false;
            updateTable();
        }

        function toggleSelectAllTable(checked) {
            const visible = getVisibleTableRows();
            if (checked) {
                visible.forEach(ap => tableSelection.add(String(ap.id).toLowerCase()));
            } else {
                visible.forEach(ap => tableSelection.delete(String(ap.id).toLowerCase()));
            }
            updateTable();
            updateSelectionStats();
        }

        function getVisibleTableRows() {
            // Apply the same filters used in updateTable, return AP objects
            const q = (tableFilters.q || '').trim().toLowerCase();
            const type = (tableFilters.type || 'all').toLowerCase();
            const cls = (tableFilters.class || 'all').toLowerCase();
            const ch = (tableFilters.channel || '').trim().toLowerCase();
            const sec = (tableFilters.security || 'all').toLowerCase();
            const rssiMin = tableFilters.rssiMin !== '' ? Number(tableFilters.rssiMin) : null;
            const rssiMax = tableFilters.rssiMax !== '' ? Number(tableFilters.rssiMax) : null;

            const out = [];
            accessPoints.forEach(ap => {
                if (isWhitelisted(ap.id)) return;
                // When in selection-only mode, mirror the map view: only show the selected set.
                if (showOnlySelected && tableSelection.size > 0 && !tableSelection.has(String(ap.id).toLowerCase())) return;
                // Worker enable checkboxes are a true inclusion filter (opacity sliders are not).
                if (!apPassesWorkerEnableFilter(ap)) return;
                // Apply global security filter (settings in the right sidebar) in addition to table filters.
                if (!passesSecurityFilter(ap)) return;

                const name = String(ap.name || '').toLowerCase();
                const mac = String(ap.mac || ap.id || '').toLowerCase();
                const dev = String(ap.device_category || ap.type || '').toLowerCase();
                const apCls = String(ap.classification || '').toLowerCase();

                if (q && !(name.includes(q) || mac.includes(q))) return;
                if (type !== 'all' && dev !== type) return;
                if (cls !== 'all' && apCls !== cls) return;
                if (sec !== 'all') {
                    if (dev === 'wifi') {
                        const wantMap = {
                            open: 'OPEN',
                            owe: 'OWE',
                            wep: 'WEP',
                            wpa1: 'WPA1',
                            wpa2: 'WPA2',
                            wpa3: 'WPA3',
                            wpa2wpa3: 'WPA2WPA3',
                            unknown: 'UNKNOWN'
                        };
                        const want = wantMap[sec] || String(sec).toUpperCase();
                        if (_canonSecurity(ap.security) !== want) return;
                    }
                }
                if (ch) {
                    const chans = (ap.channels || []).map(x => String(x).toLowerCase());
                    if (!chans.some(x => x.includes(ch))) return;
                }
                const mean = Number(ap.rssi_mean || -999);
                if (rssiMin !== null && mean < rssiMin) return;
                if (rssiMax !== null && mean > rssiMax) return;

                out.push(ap);
            });
            return out;
        }

        function viewSelectedFromTable() {
            if (tableSelection.size === 0) {
                alert('Select one or more SOIs first (use checkboxes in the table).');
                return;
            }
            
            // Enable "show only selected" mode
            showOnlySelected = true;
            
            setView('map');

            const selected = accessPoints.filter(ap => tableSelection.has(String(ap.id).toLowerCase()));
            
            // Update markers to show only selected
            updateMarkers();
            
            if (selected.length === 1) {
                showAPDetail(selected[0]);
            }

            // Zoom to bounds of selected items with GPS
            const pts = selected
                .filter(ap => ap.lat && ap.lon && ap.lat !== 0 && ap.lon !== 0)
                .map(ap => [ap.lat, ap.lon]);
            if (pts.length > 0) {
                const bounds = L.latLngBounds(pts);
                map.fitBounds(bounds, { padding: [50, 50] });
            }
        }
        
        // Exit selection-only mode (show all SOIs again)
        function showAllSOIs() {
            showOnlySelected = false;
            updateMarkers();
            updateTable();
        }

// Update table view
        function updateTable() {
            const tbody = document.getElementById('table-body');
            if (!tbody) return;
            tbody.innerHTML = '';

            let rows = getVisibleTableRows();
            
            // Apply sorting
            rows = sortAccessPoints(rows, currentSort);

            // Keep select-all checkbox in sync with visible rows
            const selAll = document.getElementById('tb-select-all');
            if (selAll) {
                if (rows.length === 0) {
                    selAll.checked = false;
                    selAll.indeterminate = false;
                } else {
                    const selectedVisible = rows.filter(ap => tableSelection.has(String(ap.id).toLowerCase())).length;
                    selAll.checked = selectedVisible === rows.length;
                    selAll.indeterminate = selectedVisible > 0 && selectedVisible < rows.length;
                }
            }
            
            let selectedRowElement = null;

            rows.forEach(ap => {
                const alpha = getEffectiveOpacityForAP(ap);
                const row = document.createElement('tr');
                row.style.opacity = alpha.toFixed(2);
                row.dataset.apId = ap.id;

                const classColor = ap.classification === 'static' ? 'var(--accent-green)' :
                                  ap.classification === 'mobile' ? 'var(--accent-orange)' : 'var(--accent-red)';

                const rssiPercent = Math.max(0, Math.min(100, (ap.rssi_mean + 100) * 1.5));
                const key = String(ap.id).toLowerCase();
                
                // Get security display (N/A for non-WiFi)
                const deviceCat = (ap.device_category || ap.type || '').toLowerCase();
                const isWifi = deviceCat.includes('wifi') || deviceCat.includes('wlan');
                const securityDisplay = isWifi ? (ap.security || 'UNKNOWN') : 'N/A';
                
                // Highlight if this is the selected AP
                if (selectedAP && String(selectedAP.id).toLowerCase() === key) {
                    row.style.background = 'rgba(59, 130, 246, 0.2)';
                    row.style.borderLeft = '3px solid var(--accent-blue)';
                    selectedRowElement = row;
                }

                // Build row with all cells properly
                // Selection checkbox cell
                const selTd = document.createElement('td');
                selTd.className = 'sel-col';
                const cb = document.createElement('input');
                cb.type = 'checkbox';
                cb.checked = tableSelection.has(key);
                cb.addEventListener('change', (e) => {
                    e.stopPropagation();
                    if (cb.checked) {
                        tableSelection.add(key);
                    } else {
                        tableSelection.delete(key);
                    }
                    updateSelectionStats();
                    syncSelectAllCheckbox();
                });
                selTd.appendChild(cb);
                row.appendChild(selTd);

                // Name cell
                const nameTd = document.createElement('td');
                nameTd.textContent = ap.name || '(Hidden)';
                row.appendChild(nameTd);

                // MAC cell
                const macTd = document.createElement('td');
                macTd.style.fontFamily = 'monospace';
                macTd.style.fontSize = '11px';
                macTd.textContent = ap.mac;
                row.appendChild(macTd);

                // Type cell
                const typeTd = document.createElement('td');
                typeTd.textContent = ap.device_category || ap.type || 'Unknown';
                row.appendChild(typeTd);
                
                // First Seen cell (Zulu time)
                const firstSeenTd = document.createElement('td');
                firstSeenTd.style.fontSize = '10px';
                firstSeenTd.style.fontFamily = 'monospace';
                firstSeenTd.style.color = 'var(--text-secondary)';
                firstSeenTd.textContent = formatZuluTime(ap.first_seen);
                row.appendChild(firstSeenTd);
                
                // Last Seen cell (Zulu time)
                const lastSeenTd = document.createElement('td');
                lastSeenTd.style.fontSize = '10px';
                lastSeenTd.style.fontFamily = 'monospace';
                lastSeenTd.style.color = 'var(--text-secondary)';
                lastSeenTd.textContent = formatZuluTime(ap.last_seen);
                row.appendChild(lastSeenTd);

                // Security cell
                const secTd = document.createElement('td');
                secTd.textContent = securityDisplay;
                secTd.style.fontSize = '11px';
                if (securityDisplay === 'OPEN' || securityDisplay === 'WEP') {
                    secTd.style.color = 'var(--accent-red)';
                } else if (securityDisplay === 'WPA3' || securityDisplay === 'WPA2/WPA3') {
                    secTd.style.color = 'var(--accent-green)';
                }
                row.appendChild(secTd);

                // Worker cell
                const workerTd = document.createElement('td');
                workerTd.textContent = ap.worker || '-';
                row.appendChild(workerTd);

                // RSSI cell
                const rssiTd = document.createElement('td');
                rssiTd.innerHTML = `
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <div class="rssi-bar">
                            <div class="rssi-fill" style="width: ${rssiPercent}%"></div>
                        </div>
                        <span>${ap.rssi_mean?.toFixed(0) || '-'} dBm</span>
                    </div>
                `;
                row.appendChild(rssiTd);

                // Detections cell
                const detTd = document.createElement('td');
                detTd.textContent = ap.detections || 0;
                row.appendChild(detTd);

                // Classification cell
                const classTd = document.createElement('td');
                classTd.innerHTML = `<span style="color: ${classColor}; font-weight: 600;">${(ap.classification || 'unknown').toUpperCase()}</span>`;
                row.appendChild(classTd);

                row.style.cursor = 'pointer';
                row.addEventListener('click', (e) => {
                    // Don't trigger detail view if clicking checkbox
                    if (e.target.tagName !== 'INPUT') {
                        showAPDetail(ap);
                    }
                });
                tbody.appendChild(row);
            });
            
            // Auto-scroll to selected AP row
            if (selectedRowElement) {
                setTimeout(() => {
                    selectedRowElement.scrollIntoView({ behavior: 'smooth', block: 'center' });
                }, 50);
            }
        }
        
        // Format timestamp to Zulu time
        function formatZuluTime(ts) {
            if (!ts) return '-';
            try {
                const d = new Date(ts);
                if (isNaN(d.getTime())) return '-';
                return d.toISOString().replace('T', ' ').slice(0, 19) + 'Z';
            } catch {
                return '-';
            }
        }
        
        // Sort access points based on current sort setting
        function sortAccessPoints(aps, sortBy) {
            const sorted = [...aps];
            switch (sortBy) {
                case 'recent':
                    sorted.sort((a, b) => {
                        const ta = a.last_seen ? new Date(a.last_seen).getTime() : 0;
                        const tb = b.last_seen ? new Date(b.last_seen).getTime() : 0;
                        return tb - ta;
                    });
                    break;
                case 'oldest':
                    sorted.sort((a, b) => {
                        const ta = a.first_seen ? new Date(a.first_seen).getTime() : Infinity;
                        const tb = b.first_seen ? new Date(b.first_seen).getTime() : Infinity;
                        return ta - tb;
                    });
                    break;
                case 'most-detections':
                    sorted.sort((a, b) => (b.detections || 0) - (a.detections || 0));
                    break;
                case 'fewest-detections':
                    sorted.sort((a, b) => (a.detections || 0) - (b.detections || 0));
                    break;
                case 'strongest':
                    sorted.sort((a, b) => (b.rssi_mean || -999) - (a.rssi_mean || -999));
                    break;
                case 'weakest':
                    sorted.sort((a, b) => (a.rssi_mean || -999) - (b.rssi_mean || -999));
                    break;
                case 'alpha-asc':
                    sorted.sort((a, b) => (a.name || '').localeCompare(b.name || ''));
                    break;
                case 'alpha-desc':
                    sorted.sort((a, b) => (b.name || '').localeCompare(a.name || ''));
                    break;
                case 'class':
                    const classOrder = { 'static': 0, 'mobile': 1, 'uncertain': 2 };
                    sorted.sort((a, b) => (classOrder[a.classification] || 3) - (classOrder[b.classification] || 3));
                    break;
            }
            return sorted;
        }
        
        // Set table sort order
        function setTableSort(sortBy) {
            currentSort = sortBy;
            updateTable();
        }
        
        // Sync select-all checkbox state
        function syncSelectAllCheckbox() {
            const selAll = document.getElementById('tb-select-all');
            if (!selAll) return;
            
            const rows = getVisibleTableRows();
            if (rows.length === 0) {
                selAll.checked = false;
                selAll.indeterminate = false;
            } else {
                const selectedVisible = rows.filter(ap => tableSelection.has(String(ap.id).toLowerCase())).length;
                selAll.checked = selectedVisible === rows.length;
                selAll.indeterminate = selectedVisible > 0 && selectedVisible < rows.length;
            }
        }
        
        // Update selection statistics display
        function updateSelectionStats() {
            const selectedCount = tableSelection.size;
            const statEl = document.getElementById('stat-selected');
            if (statEl) {
                statEl.textContent = selectedCount;
            }
            
            // Update displayed stats if we have selection
            updateFilteredStats();
        }
        
        // Update stats based on current filters/selection
        function updateFilteredStats() {
            const visible = getVisibleTableRows();
            const selected = tableSelection.size > 0 
                ? accessPoints.filter(ap => tableSelection.has(String(ap.id).toLowerCase()))
                : visible;
            
            const displayedEl = document.getElementById('stat-displayed');
            if (displayedEl) {
                displayedEl.textContent = visible.length;
            }
            
            // Count by classification in displayed/selected
            let staticCount = 0, mobileCount = 0, uncertainCount = 0;
            selected.forEach(ap => {
                if (ap.classification === 'static') staticCount++;
                else if (ap.classification === 'mobile') mobileCount++;
                else uncertainCount++;
            });
            
            const filtStaticEl = document.getElementById('stat-filtered-static');
            const filtMobileEl = document.getElementById('stat-filtered-mobile');
            const filtUncertainEl = document.getElementById('stat-filtered-uncertain');
            
            if (filtStaticEl) filtStaticEl.textContent = staticCount;
            if (filtMobileEl) filtMobileEl.textContent = mobileCount;
            if (filtUncertainEl) filtUncertainEl.textContent = uncertainCount;
        }

        // Show AP detail
        function showAPDetail(ap) {
            selectedAP = ap;
            const panel = document.getElementById('detail-panel');
            
            const classColor = ap.classification === 'static' ? 'static' : 
                              ap.classification === 'mobile' ? 'mobile' : 'uncertain';
            
            panel.innerHTML = `
                <div class="ap-header">
                    <div class="ap-name">${ap.name || '(Hidden Network)'}</div>
                    <div class="ap-mac">${ap.mac}</div>
                    <span class="classification-badge ${classColor}">
                        ${(ap.classification || 'Unknown').toUpperCase()}
                    </span>
                </div>
                
                <div class="confidence-section">
                    <div class="filter-label">Confidence: ${((ap.confidence || 0) * 100).toFixed(1)}%</div>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: ${(ap.confidence || 0) * 100}%"></div>
                    </div>
                </div>
                
                <div class="ap-details-grid">
                    <div class="detail-item">
                        <div class="detail-item-label">Type</div>
                        <div class="detail-item-value">${ap.type || 'Unknown'}</div>
                    </div>
                    <div class="detail-item">
                        <div class="detail-item-label">Worker</div>
                        <div class="detail-item-value">${ap.workers?.join(', ') || ap.worker || '-'}</div>
                    </div>
                    <div class="detail-item">
                        <div class="detail-item-label">Detections</div>
                        <div class="detail-item-value">${ap.detections || 0}</div>
                    </div>
                    <div class="detail-item">
                        <div class="detail-item-label">Runs Seen</div>
                        <div class="detail-item-value">${ap.runs || 1}</div>
                    </div>
                    <div class="detail-item">
                        <div class="detail-item-label">RSSI Mean</div>
                        <div class="detail-item-value">${ap.rssi_mean?.toFixed(1) || '-'} dBm</div>
                    </div>
                    <div class="detail-item">
                        <div class="detail-item-label">RSSI Range</div>
                        <div class="detail-item-value">${ap.rssi_min || '-'} to ${ap.rssi_max || '-'}</div>
                    </div>
                    <div class="detail-item">
                        <div class="detail-item-label">Std Dev</div>
                        <div class="detail-item-value">${ap.rssi_std_dev?.toFixed(2) || '-'} dB</div>
                    </div>
                    <div class="detail-item">
                        <div class="detail-item-label">Coverage</div>
                        <div class="detail-item-value">${ap.coverage || '-'}m</div>
                    </div>
                    ${(String(ap.device_category || '').toLowerCase() === 'wifi') ? `
                    <div class="detail-item">
                        <div class="detail-item-label">Security</div>
                        <div class="detail-item-value">${(ap.security || 'UNKNOWN')}</div>
                    </div>
                    ` : ''}
                </div>
                
                ${ap.lat && ap.lon && ap.lat !== 0 ? `
                <div style="margin-top: 16px;">
                    <div class="filter-label">Strongest observed point (best guess for AP proximity)</div>
                    <div style="font-family: monospace; font-size: 12px; color: var(--text-primary);">
                        ${(ap.strong_lat || ap.est_lat || ap.lat)?.toFixed(6)}, ${(ap.strong_lon || ap.est_lon || ap.lon)?.toFixed(6)}
                    </div>
                    <div style="font-size: 11px; color: var(--text-secondary); margin-top: 6px;">
                        Captured radius: <span style="color: var(--text-primary);">${formatMeters(ap.strong_radius_m || ap.observed_radius_m || 0)}</span>
                    </div>
                    <div style="font-size: 11px; color: var(--text-secondary);">
                        Map centroid: <span style="font-family: monospace; color: var(--text-primary);">${ap.lat?.toFixed(6)}, ${ap.lon?.toFixed(6)}</span>
                    </div>
                </div>
                ` : `
                <div class="no-gps-warning" style="margin-top: 16px;">
                    No GPS coordinates available for this AP
                </div>
                `}


                <div style="margin-top: 12px; padding: 10px; border: 1px solid var(--border-color); border-radius: 10px; background: rgba(26, 32, 41, 0.55);">
                    <div class="filter-label" style="margin-bottom: 6px;">Dim other SOIs while focused</div>
                    <div style="display:flex; align-items:center; gap:10px;">
                        <input type="range" min="0" max="100" step="1" value="${Math.round(dimOthersOpacity*100)}" oninput="setDimOthersOpacity(this.value)">
                        <div style="font-size: 11px; color: var(--text-secondary); width: 42px; text-align:right;">${Math.round(dimOthersOpacity*100)}%</div>
                    </div>
                    <div style="font-size: 11px; color: var(--text-secondary); margin-top: 6px;">Tip: set to ~10–30% for congested areas.</div>
                </div>

                <div class="detail-actions">
                    <button class="action-btn" onclick="centerOnSelected()">🎯 Center</button>
                    <button class="action-btn" onclick="generateReport([ap.id])">📑 Report</button>
                    <button class="action-btn" onclick="toggleWhitelistSelected()">
                        ${isWhitelisted(ap.id) ? 'Unhide' : 'Hide (Whitelist)'}
                    </button>
                </div>
                
                ${ap.channels && ap.channels.length > 0 ? `
                <div style="margin-top: 12px;">
                    <div class="filter-label">Channels</div>
                    <div style="color: var(--text-primary);">${ap.channels.join(', ')}</div>
                </div>
                ` : ''}
            `;
            
            // Do NOT move/zoom the map camera on selection (use the Center button instead).

            // Re-render to apply selection dimming and overlays
            updateMarkers();
            updateTable();
        }
        
        // Filter initialization
        function initFilters() {
            document.querySelectorAll('#classification-filters .filter-btn').forEach(btn => {
                btn.addEventListener('click', () => {
                    document.querySelectorAll('#classification-filters .filter-btn').forEach(b => b.classList.remove('active'));
                    btn.classList.add('active');
                    filters.classification = btn.dataset.filter;
                    requestAPs();
                    updateFilteredStats();
                });
            });

            // Worker opacity UI is built from data once APs are loaded.
            // Initialize with a reasonable default set so the controls exist immediately.
            initWorkerOpacityUI(['A','B','C','D','E','F','M','unknown']);
            loadSettings();
            renderWhitelistList();
            
            document.querySelectorAll('#device-filters .filter-btn').forEach(btn => {
                btn.addEventListener('click', () => {
                    document.querySelectorAll('#device-filters .filter-btn').forEach(b => b.classList.remove('active'));
                    btn.classList.add('active');
                    filters.deviceType = btn.dataset.filter;
                    
                    // Reset security filter to "All" if switching away from WiFi
                    if (btn.dataset.filter !== 'wifi' && btn.dataset.filter !== 'all') {
                        document.querySelectorAll('#security-filters .filter-btn').forEach(b => b.classList.remove('active'));
                        const allSecBtn = document.querySelector('#security-filters .filter-btn[data-filter="all"]');
                        if (allSecBtn) {
                            allSecBtn.classList.add('active');
                            filters.security = 'all';
                        }
                    }
                    
                    requestAPs();
                    updateFilteredStats();
                });
            });

            document.querySelectorAll('#security-filters .filter-btn').forEach(btn => {
                btn.addEventListener('click', () => {
                    document.querySelectorAll('#security-filters .filter-btn').forEach(b => b.classList.remove('active'));
                    btn.classList.add('active');
                    filters.security = btn.dataset.filter;
                    
                    // Auto-switch to WiFi when selecting a security filter (except "All")
                    if (btn.dataset.filter !== 'all') {
                        document.querySelectorAll('#device-filters .filter-btn').forEach(b => b.classList.remove('active'));
                        const wifiBtn = document.querySelector('#device-filters .filter-btn[data-filter="wifi"]');
                        if (wifiBtn) {
                            wifiBtn.classList.add('active');
                            filters.deviceType = 'wifi';
                        }
                    }
                    
                    updateMarkers();
                    updateTable();
                    updateFilteredStats();
                });
            });
        }
        
        // View toggle
        function togglePanel(panelId) {
            const el = document.getElementById(panelId);
            if (!el) return;
            el.classList.toggle('collapsed');
            try {
                localStorage.setItem(`wa_panel_collapsed_${panelId}`, el.classList.contains('collapsed') ? '1' : '0');
            } catch (e) {}
            // Allow Leaflet to recalc size after layout changes
            if (window.map) {
                setTimeout(() => map.invalidateSize(), 120);
            }
        }

        function restorePanelCollapseState() {
            ['left-panel', 'right-panel'].forEach(pid => {
                try {
                    const v = localStorage.getItem(`wa_panel_collapsed_${pid}`);
                    if (v === '1') {
                        const el = document.getElementById(pid);
                        if (el) el.classList.add('collapsed');
                    }
                } catch (e) {}
            });
        }

        function setView(view) {
            currentView = view;
            document.getElementById('btn-map-view').classList.toggle('active', view === 'map');
            document.getElementById('btn-table-view').classList.toggle('active', view === 'table');
            document.getElementById('btn-gnss-view').classList.toggle('active', view === 'gnss');
            document.getElementById('map-section').style.display = view === 'map' ? 'block' : 'none';
            document.getElementById('table-section').style.display = view === 'table' ? 'block' : 'none';
            document.getElementById('gnss-section').style.display = view === 'gnss' ? 'block' : 'none';
            
            if (view === 'map') {
                setTimeout(() => map.invalidateSize(), 100);
            }
            if (view === 'gnss') {
                loadGNSSStatus();
            }
        }
        
        // ============= GNSS Link Functions =============
        
        let gnssSession = null;
        
        async function loadGNSSStatus() {
            try {
                const response = await fetch('/api/gnss/status');
                const data = await response.json();
                
                if (data.session_active) {
                    gnssSession = data;
                    await loadGNSSData();
                }
            } catch (error) {
                console.error('Failed to load GNSS status:', error);
            }
        }
        
        async function handleGNSSFileUpload(event) {
            const files = event.target.files;
            if (files.length === 0) return;
            
            const formData = new FormData();
            for (const file of files) {
                formData.append('files', file);
            }
            formData.append('session_id', `session_${Date.now()}`);
            formData.append('username', 'analyst');
            
            try {
                showNotification('Uploading GNSS data...', 'info');
                
                const response = await fetch('/api/gnss/load-csv', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                if (data.success) {
                    gnssSession = data;
                    await loadGNSSData();
                    showNotification(`Loaded ${data.positions} positions from ${data.receivers.length} receivers`, 'success');
                } else {
                    showNotification('Failed to load GNSS data: ' + data.error, 'error');
                }
            } catch (error) {
                showNotification('Error uploading GNSS files: ' + error.message, 'error');
            }
            
            // Reset file input
            event.target.value = '';
        }
        
        async function loadGNSSData() {
            try {
                // Load metrics
                const metricsRes = await fetch('/api/gnss/metrics');
                const metrics = await metricsRes.json();
                
                // Load receivers
                const receiversRes = await fetch('/api/gnss/receivers');
                const receiversData = await receiversRes.json();
                
                // Update UI
                updateGNSSUI(metrics, receiversData.receivers);
            } catch (error) {
                console.error('Failed to load GNSS data:', error);
            }
        }
        
        function updateGNSSUI(metrics, receivers) {
            // Show session info
            document.getElementById('gnss-session-info').style.display = 'block';
            document.getElementById('gnss-session-id').textContent = gnssSession.session_id || 'Active';
            document.getElementById('gnss-receiver-count').textContent = receivers.length;
            document.getElementById('gnss-position-count').textContent = gnssSession.positions || 0;
            
            if (gnssSession.time_range) {
                const start = gnssSession.time_range.start ? new Date(gnssSession.time_range.start).toLocaleString() : '';
                const end = gnssSession.time_range.end ? new Date(gnssSession.time_range.end).toLocaleString() : '';
                document.getElementById('gnss-time-range').textContent = start && end ? `${start} - ${end}` : '-';
            }
            
            // Show metrics
            document.getElementById('gnss-metrics').style.display = 'block';
            document.getElementById('gnss-std-h').textContent = metrics.all_receivers.std_h?.toFixed(3) || '-';
            document.getElementById('gnss-std-v').textContent = metrics.all_receivers.std_v?.toFixed(3) || '-';
            document.getElementById('gnss-cep-50').textContent = metrics.all_receivers.cep_50?.toFixed(3) || '-';
            document.getElementById('gnss-2drms').textContent = metrics.all_receivers.drms_2?.toFixed(3) || '-';
            document.getElementById('gnss-max-dev').textContent = metrics.all_receivers.max_deviation?.toFixed(3) || '-';
            
            // Show receivers table
            document.getElementById('gnss-receivers').style.display = 'block';
            const tbody = document.getElementById('gnss-receiver-tbody');
            tbody.innerHTML = '';
            
            for (const rcvr of receivers) {
                const row = document.createElement('tr');
                const stats = rcvr.stats || {};
                row.innerHTML = `
                    <td><span style="color: ${rcvr.icon_color}">●</span> ${rcvr.nickname || rcvr.id}</td>
                    <td><input type="checkbox" ${rcvr.is_truth ? 'checked' : ''} onchange="toggleGNSSTruth('${rcvr.id}', this.checked)"></td>
                    <td>${rcvr.positions || 0}</td>
                    <td>${stats.sat_count_mean?.toFixed(1) || '-'}</td>
                    <td>${stats.hdop_mean?.toFixed(2) || '-'}</td>
                    <td>${stats.deviation_mean?.toFixed(3) || '-'}</td>
                    <td>${stats.cep_50?.toFixed(3) || '-'}</td>
                    <td>${stats.drms_2?.toFixed(3) || '-'}</td>
                `;
                tbody.appendChild(row);
            }
            
            // Show actions
            document.getElementById('gnss-actions').style.display = 'flex';
        }
        
        async function toggleGNSSTruth(receiverId, isTruth) {
            try {
                await fetch(`/api/gnss/receivers/${receiverId}`, {
                    method: 'PUT',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ is_truth: isTruth })
                });
                
                await loadGNSSData();
                showNotification(`Receiver ${receiverId} ${isTruth ? 'set as' : 'removed from'} truth`, 'success');
            } catch (error) {
                showNotification('Failed to update receiver: ' + error.message, 'error');
            }
        }
        
        async function generateGNSSReport() {
            try {
                showNotification('Generating GNSS report...', 'info');
                
                const response = await fetch('/api/gnss/report', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        title: 'GNSS Link Analysis Report',
                        dark_mode: true,
                        include_map: true,
                        include_tracks: true
                    })
                });
                
                if (response.ok) {
                    const blob = await response.blob();
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = `gnss_report_${Date.now()}.pdf`;
                    a.click();
                    URL.revokeObjectURL(url);
                    showNotification('GNSS report downloaded', 'success');
                } else {
                    const error = await response.json();
                    showNotification('Failed to generate report: ' + error.error, 'error');
                }
            } catch (error) {
                showNotification('Error generating report: ' + error.message, 'error');
            }
        }
        
        async function exportGNSSCSV() {
            try {
                const response = await fetch('/api/gnss/export/csv');
                
                if (response.ok) {
                    const blob = await response.blob();
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = `gnss_export_${Date.now()}.csv`;
                    a.click();
                    URL.revokeObjectURL(url);
                    showNotification('GNSS CSV exported', 'success');
                } else {
                    const error = await response.json();
                    showNotification('Failed to export CSV: ' + error.error, 'error');
                }
            } catch (error) {
                showNotification('Error exporting CSV: ' + error.message, 'error');
            }
        }
        
        async function clearGNSSSession() {
            if (!confirm('Clear GNSS session data?')) return;
            
            try {
                await fetch('/api/gnss/clear', { method: 'POST' });
                
                gnssSession = null;
                document.getElementById('gnss-session-info').style.display = 'none';
                document.getElementById('gnss-metrics').style.display = 'none';
                document.getElementById('gnss-receivers').style.display = 'none';
                document.getElementById('gnss-actions').style.display = 'none';
                
                showNotification('GNSS session cleared', 'success');
            } catch (error) {
                showNotification('Failed to clear session: ' + error.message, 'error');
            }
        }
        
        // Drag and drop
        function initDragDrop() {
            const zone = document.getElementById('upload-zone');
            
            zone.addEventListener('dragover', (e) => {
                e.preventDefault();
                zone.classList.add('dragover');
            });
            
            zone.addEventListener('dragleave', () => {
                zone.classList.remove('dragover');
            });
            
            zone.addEventListener('drop', (e) => {
                e.preventDefault();
                zone.classList.remove('dragover');
                if (e.dataTransfer.files.length > 0) {
                    uploadFile(e.dataTransfer.files[0]);
                }
            });
        }
        
        // File upload handler
        function handleFileUpload(event) {
            const files = event.target.files;
            if (files.length > 0) {
                uploadFiles(Array.from(files));
            }
        }
        
        function uploadFiles(files) {
            const formData = new FormData();
            files.forEach(file => {
            if (activeProfileId) formData.append('profile_id', activeProfileId);
                formData.append('files', file);
            });
            
            const fileNames = files.map(f => f.name).join(', ');
            const fileCount = files.length;
            
            // Show loading
            document.getElementById('upload-zone').innerHTML = `
                <div class="loading">
                    <div class="spinner"></div>
                    <div style="margin-top: 12px;">Analyzing ${fileCount} file${fileCount > 1 ? 's' : ''}...</div>
                    <div style="font-size: 11px; color: var(--text-muted); margin-top: 4px;">${fileNames}</div>
                </div>
            `;
            
            fetch('/api/analysis/load-multi', {
                method: 'POST',
                body: formData
            })
            .then(res => res.json())
            .then(data => {
                // Reset upload zone
                document.getElementById('upload-zone').innerHTML = `
                    <div class="upload-icon">📤</div>
                    <div class="upload-text">
                        <strong>Click to upload</strong> or drag CSV files
                        <div style="font-size: 11px; color: var(--text-muted); margin-top: 4px;">
                            Multiple files supported for multi-day analysis
                        </div>
                    </div>
                `;
                
                if (data.success) {
                    console.log('Analysis complete:', data);
                    // Show loaded runs
                    updateLoadedRuns(data.results);
                } else {
                    alert('Error: ' + (data.error || 'Unknown error'));
                }
            })
            .catch(err => {
                document.getElementById('upload-zone').innerHTML = `
                    <div class="upload-icon">📤</div>
                    <div class="upload-text">
                        <strong>Click to upload</strong> or drag CSV files
                        <div style="font-size: 11px; color: var(--text-muted); margin-top: 4px;">
                            Multiple files supported for multi-day analysis
                        </div>
                    </div>
                `;
                alert('Upload failed: ' + err.message);
            });
        }
        
        function updateLoadedRuns(results) {
            const container = document.getElementById('loaded-runs');
            const list = document.getElementById('runs-list');
            
            if (!results || results.length === 0) {
                return;
            }
            
            // Append new runs to the tracked list
            results.forEach(r => {
                // Check if this run already exists (by run_id)
                const existing = loadedRuns.find(lr => lr.run_id === r.run_id);
                if (!existing) {
                    loadedRuns.push(r);
                }
            });
            
            // Re-render the full list
            container.style.display = 'block';
            list.innerHTML = loadedRuns.map(r => 
                `<div class="run-item">
                    <input type="checkbox" class="run-checkbox" data-run-id="${r.run_id}" data-filename="${r.filename}" checked style="margin-right: 8px;">
                    <span class="run-name">${r.filename}</span>
                    <span class="run-count">${r.result?.total_rows || '?'} rows</span>
                </div>`
            ).join('');
        }
        
        // Clear selected runs (remove unchecked runs from data)
        function clearSelectedRuns() {
            const checkboxes = document.querySelectorAll('.run-checkbox:not(:checked)');
            if (checkboxes.length === 0) {
                alert('Uncheck the runs you want to remove, then click Clear Selected.');
                return;
            }
            
            const toRemove = Array.from(checkboxes).map(cb => cb.dataset.runId);
            const filenames = Array.from(checkboxes).map(cb => cb.dataset.filename).join(', ');
            
            if (confirm(`Remove ${toRemove.length} run(s)?\n\nFiles: ${filenames}\n\nThis will remove all detections from these runs.`)) {
                fetch('/api/analysis/remove-runs', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ run_ids: toRemove })
                })
                .then(res => res.json())
                .then(data => {
                    if (data.success) {
                        console.log('Runs removed:', data);
                        
                        // Update the loadedRuns state
                        loadedRuns = loadedRuns.filter(r => !toRemove.includes(r.run_id));
                        
                        // Remove the unchecked items from the UI
                        checkboxes.forEach(cb => {
                            cb.closest('.run-item').remove();
                        });
                        
                        // Hide runs container if empty
                        const list = document.getElementById('runs-list');
                        if (list && list.children.length === 0) {
                            document.getElementById('loaded-runs').style.display = 'none';
                        }
                        
                        // Request fresh data
                        requestAPs();
                    } else {
                        alert('Error removing runs: ' + (data.error || 'Unknown error'));
                    }
                })
                .catch(err => {
                    console.error('Error removing runs:', err);
                    alert('Error removing runs: ' + err.message);
                });
            }
        }
        
        // Clear data
        function clearData() {
            if (confirm('Clear all loaded data?')) {
                fetch('/api/clear', { method: 'POST' })
                    .then(res => res.json())
                    .then(data => {
                        if (!data.success) {
                            alert('Error clearing data');
                        }
                    });
            }
        }

        // ------------------- Reports -------------------
        
        // Generate report for selected SOIs (or all visible if none selected)
        function generateSelectedReport() {
            const format = document.getElementById('report-format')?.value || 'pdf';
            
            // Use selection if any, otherwise use visible filtered results
            let ids;
            if (tableSelection.size > 0) {
                ids = Array.from(tableSelection);
            } else {
                // Use currently visible/filtered APs
                const visible = getVisibleTableRows();
                ids = visible.map(ap => String(ap.id).toLowerCase());
            }
            
            if (ids.length === 0) {
                alert('No SOIs to include in report. Load data or adjust filters.');
                return;
            }
            
            if (format === 'pdf') {
                generatePDFReport(ids);
            } else {
                generateHTMLReport(ids);
            }
        }
        
        async function generateHTMLReport(ids) {
            try {
                const list = (ids || []).map(x => String(x).toLowerCase()).filter(Boolean);
                if (list.length === 0) return;

                const res = await fetch('/api/report/html', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ ids: list })
                });
                if (!res.ok) {
                    const msg = await res.text();
                    throw new Error(msg || 'Report generation failed');
                }
                const blob = await res.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                const ts = new Date();
                const stamp = ts.toISOString().replace(/[:.]/g, '').slice(0, 15);
                a.href = url;
                a.download = `wardriving_report_${stamp}.html`;
                document.body.appendChild(a);
                a.click();
                a.remove();
                window.URL.revokeObjectURL(url);
            } catch (err) {
                console.error(err);
                alert('Report failed: ' + (err.message || err));
            }
        }
        
        async function generatePDFReport(ids) {
            try {
                const list = (ids || []).map(x => String(x).toLowerCase()).filter(Boolean);
                
                const res = await fetch('/api/report/pdf', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        title: 'RF Site Survey Analysis Report',
                        include_details: true,
                        ids: list.length > 0 ? list : null  // null means all
                    })
                });
                
                if (!res.ok) {
                    const data = await res.json();
                    throw new Error(data.error || 'PDF generation failed');
                }
                
                const blob = await res.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                const stamp = new Date().toISOString().slice(0,19).replace(/[:-]/g, '');
                a.href = url;
                a.download = `wardriving_report_${stamp}.pdf`;
                document.body.appendChild(a);
                a.click();
                window.URL.revokeObjectURL(url);
                a.remove();
            } catch (err) {
                console.error(err);
                alert('PDF export failed: ' + err.message);
            }
        }
        
        // Legacy function for compatibility
        async function generateReport(ids) {
            const format = document.getElementById('report-format')?.value || 'pdf';
            if (format === 'pdf') {
                await generatePDFReport(ids);
            } else {
                await generateHTMLReport(ids);
            }
        }

        function generateReportFromTable() {
            if (tableSelection.size === 0) {
                alert('Select one or more SOIs first (use checkboxes in the table).');
                return;
            }
            generateReport(Array.from(tableSelection));
        }
        
        // Export functions
        function exportGeoJSON() {
            window.location.href = '/api/export/geojson';
        }
        
        function exportKML() {
            window.location.href = '/api/export/kml';
        }
        
        function exportCSV() {
            window.location.href = '/api/export/csv';
        }
    </script>
</body>
</html>
'''


def initialize_app(mqtt_config=None):
    """Initialize the application"""
    logger.info("=" * 50)
    logger.info("Wardriving Analysis Server Starting")
    logger.info("=" * 50)
    
    if mqtt_config and mqtt_config.get('host'):
        if initialize_mqtt(mqtt_config):
            logger.info(f"MQTT connected to {mqtt_config['host']}:{mqtt_config.get('port', 1883)}")
        else:
            logger.info("MQTT not available - continuing with offline mode")
    else:
        logger.info("MQTT not configured - running in offline mode")
    
    logger.info("Server ready for requests")
    logger.info("=" * 50)


if __name__ == '__main__':
    # Load config from environment
    mqtt_config = {
        'host': os.getenv('MQTT_HOST', ''),
        'port': int(os.getenv('MQTT_PORT', 1883)),
        'username': os.getenv('MQTT_USER'),
        'password': os.getenv('MQTT_PASS')
    }
    
    initialize_app(mqtt_config)
    
    # Run server
    debug = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    port = int(os.getenv('FLASK_PORT', 5000))
    
    print(f"\n🌐 Open http://localhost:{port} in your browser\n")
    
    socketio.run(app, host='0.0.0.0', port=port, debug=debug, allow_unsafe_werkzeug=True)
