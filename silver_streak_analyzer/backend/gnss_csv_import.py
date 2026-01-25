#!/usr/bin/env python3
"""
GNSS Link - CSV Import Module
Version: 4.8.0

Supports CSV import in Option B format (combined CSV with receiver column).
Also supports single-receiver CSV files with receiver ID from filename.
"""

import csv
import re
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Generator
from dataclasses import dataclass

from gnss_models import (
    GNSSPosition, GNSSSession, ReceiverConfig, 
    ProtocolType, FixType, haversine_distance
)

logger = logging.getLogger(__name__)


@dataclass
class CSVColumnProfile:
    """
    Column mapping profile for CSV import.
    Maps CSV column names to internal field names.
    """
    name: str
    description: str = ""
    header_rows: int = 0  # Number of metadata rows before column headers
    
    # Required columns
    timestamp: str = "Timestamp"
    lat: str = "Latitude"
    lon: str = "Longitude"
    
    # Optional receiver ID column (for Option B multi-receiver CSVs)
    receiver_id: Optional[str] = None  # If None, derive from filename
    
    # Optional position columns
    alt_msl: Optional[str] = None
    alt_hae: Optional[str] = None
    
    # Fix quality columns
    fix_type: Optional[str] = None
    fix_quality: Optional[str] = None  # NMEA-style fix quality
    carrier_solution: Optional[str] = None
    sat_count: Optional[str] = None
    nav_mode: Optional[str] = None
    
    # DOP columns
    hdop: Optional[str] = None
    vdop: Optional[str] = None
    pdop: Optional[str] = None
    
    # Accuracy columns (in various units)
    accuracy_h_mm: Optional[str] = None
    accuracy_h_m: Optional[str] = None
    accuracy_v_mm: Optional[str] = None
    accuracy_v_m: Optional[str] = None
    
    # Velocity columns
    speed_knots: Optional[str] = None
    speed_mps: Optional[str] = None
    ground_speed_mmps: Optional[str] = None
    course_deg: Optional[str] = None
    heading_motion: Optional[str] = None
    heading_vehicle: Optional[str] = None
    vel_n_mmps: Optional[str] = None
    vel_e_mmps: Optional[str] = None
    vel_d_mmps: Optional[str] = None
    
    # Extra metadata
    geoid_sep: Optional[str] = None
    mag_var: Optional[str] = None
    diff_age: Optional[str] = None
    diff_station: Optional[str] = None
    flags: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON storage"""
        return {
            'name': self.name,
            'description': self.description,
            'header_rows': self.header_rows,
            'timestamp': self.timestamp,
            'lat': self.lat,
            'lon': self.lon,
            'receiver_id': self.receiver_id,
            'alt_msl': self.alt_msl,
            'alt_hae': self.alt_hae,
            'fix_type': self.fix_type,
            'fix_quality': self.fix_quality,
            'carrier_solution': self.carrier_solution,
            'sat_count': self.sat_count,
            'nav_mode': self.nav_mode,
            'hdop': self.hdop,
            'vdop': self.vdop,
            'pdop': self.pdop,
            'accuracy_h_mm': self.accuracy_h_mm,
            'accuracy_h_m': self.accuracy_h_m,
            'accuracy_v_mm': self.accuracy_v_mm,
            'accuracy_v_m': self.accuracy_v_m,
            'speed_knots': self.speed_knots,
            'speed_mps': self.speed_mps,
            'ground_speed_mmps': self.ground_speed_mmps,
            'course_deg': self.course_deg,
            'heading_motion': self.heading_motion,
            'heading_vehicle': self.heading_vehicle,
            'vel_n_mmps': self.vel_n_mmps,
            'vel_e_mmps': self.vel_e_mmps,
            'vel_d_mmps': self.vel_d_mmps,
            'geoid_sep': self.geoid_sep,
            'mag_var': self.mag_var,
            'diff_age': self.diff_age,
            'diff_station': self.diff_station,
            'flags': self.flags
        }


# Built-in profiles
PROFILE_NMEA_STANDARD = CSVColumnProfile(
    name="NMEA-Standard",
    description="Standard NMEA-derived CSV format",
    header_rows=2,
    timestamp="Timestamp",
    lat="Latitude",
    lon="Longitude",
    receiver_id="ReceiverID",
    alt_msl="AltitudeM",
    fix_quality="FixQuality",
    sat_count="NumSats",
    hdop="HDOP",
    vdop="VDOP",
    pdop="PDOP",
    geoid_sep="GeoidSepM",
    speed_knots="SpeedKnots",
    course_deg="CourseDeg",
    mag_var="MagVarDeg",
    diff_age="DiffAge",
    diff_station="DiffStation",
    nav_mode="NavMode"
)

PROFILE_UBX_STANDARD = CSVColumnProfile(
    name="UBX-Standard",
    description="Standard UBX-derived CSV format (NAV-PVT style)",
    header_rows=2,
    timestamp="Timestamp",
    lat="Latitude",
    lon="Longitude",
    receiver_id="ReceiverID",
    alt_hae="HeightM",
    alt_msl="HeightMSL",
    fix_type="FixType",
    carrier_solution="CarrierSoln",
    sat_count="NumSV",
    hdop="HDOP",
    vdop="VDOP",
    pdop="PDOP",
    accuracy_h_mm="HAcc_mm",
    accuracy_v_mm="VAcc_mm",
    ground_speed_mmps="GSpeed_mmps",
    heading_motion="HeadMot_deg",
    heading_vehicle="HeadVeh_deg",
    vel_n_mmps="VelN_mmps",
    vel_e_mmps="VelE_mmps",
    vel_d_mmps="VelD_mmps",
    flags="Flags"
)

PROFILE_GNSS_LINK_OPTION_B = CSVColumnProfile(
    name="GNSS-Link-OptionB",
    description="Multi-receiver CSV with ReceiverID column",
    header_rows=0,
    timestamp="Timestamp",
    lat="Latitude",
    lon="Longitude",
    receiver_id="ReceiverID",
    alt_msl="AltitudeMSL",
    alt_hae="AltitudeHAE",
    fix_type="FixType",
    carrier_solution="CarrierSolution",
    sat_count="SatCount",
    hdop="HDOP",
    vdop="VDOP",
    pdop="PDOP",
    accuracy_h_m="AccuracyH_m",
    accuracy_v_m="AccuracyV_m",
    speed_mps="Speed_mps",
    heading_motion="Heading_deg"
)

# Profile registry
BUILTIN_PROFILES: Dict[str, CSVColumnProfile] = {
    'NMEA-Standard': PROFILE_NMEA_STANDARD,
    'UBX-Standard': PROFILE_UBX_STANDARD,
    'GNSS-Link-OptionB': PROFILE_GNSS_LINK_OPTION_B,
}


class GNSSCSVImporter:
    """
    Imports GNSS data from CSV files.
    
    Supports:
    - Option B: Combined multi-receiver CSV with ReceiverID column
    - Single-receiver CSVs (receiver ID from filename or specified)
    - Auto-detection of file format
    - Custom column mapping profiles
    """
    
    def __init__(self, profile: Optional[CSVColumnProfile] = None):
        """
        Initialize importer.
        
        Args:
            profile: Column mapping profile. If None, will auto-detect.
        """
        self.profile = profile
        self.detected_receivers: Dict[str, ReceiverConfig] = {}
        self.metadata: Dict[str, str] = {}
        
    def detect_format(self, filepath: Path) -> Tuple[CSVColumnProfile, ProtocolType]:
        """
        Auto-detect CSV format by examining headers.
        
        Args:
            filepath: Path to CSV file
            
        Returns:
            Tuple of (profile, protocol_type)
        """
        with open(filepath, 'r', encoding='utf-8-sig') as f:
            # Read first few lines
            lines = [f.readline() for _ in range(5)]
        
        # Check for metadata header (like "Version,AppName,Model...")
        has_metadata_header = False
        header_row = 0
        
        if lines[0] and 'Version' in lines[0] and 'AppName' in lines[0]:
            has_metadata_header = True
            header_row = 2
            # Parse metadata
            meta_keys = lines[0].strip().split(',')
            meta_vals = lines[1].strip().split(',')
            self.metadata = dict(zip(meta_keys, meta_vals))
        
        # Get actual column headers
        header_line = lines[header_row] if len(lines) > header_row else ""
        columns = [c.strip() for c in header_line.split(',')]
        
        # Detect protocol type
        protocol = ProtocolType.UNKNOWN
        if 'UBX' in self.metadata.get('Version', '').upper():
            protocol = ProtocolType.UBX
        elif 'NMEA' in self.metadata.get('Version', '').upper():
            protocol = ProtocolType.NMEA
        elif 'HeightM' in columns or 'HAcc_mm' in columns or 'CarrierSoln' in columns:
            protocol = ProtocolType.UBX
        elif 'FixQuality' in columns or 'SpeedKnots' in columns or 'GeoidSepM' in columns:
            protocol = ProtocolType.NMEA
        
        # Check for ReceiverID column (Option B)
        has_receiver_id = 'ReceiverID' in columns or 'Receiver_ID' in columns or 'receiver_id' in columns
        
        # Select appropriate profile
        if protocol == ProtocolType.UBX:
            profile = CSVColumnProfile(
                name="Auto-UBX",
                header_rows=header_row,
                timestamp=self._find_column(columns, ['Timestamp', 'Time', 'UTC']),
                lat=self._find_column(columns, ['Latitude', 'Lat', 'lat']),
                lon=self._find_column(columns, ['Longitude', 'Lon', 'lon', 'Long']),
                receiver_id=self._find_column(columns, ['ReceiverID', 'Receiver_ID', 'receiver_id']) if has_receiver_id else None,
                alt_hae=self._find_column(columns, ['HeightM', 'Height', 'AltHAE']),
                alt_msl=self._find_column(columns, ['HeightMSL', 'AltMSL', 'AltitudeM']),
                fix_type=self._find_column(columns, ['FixType', 'Fix']),
                carrier_solution=self._find_column(columns, ['CarrierSoln', 'CarrierSolution']),
                sat_count=self._find_column(columns, ['NumSV', 'NumSats', 'SatCount']),
                hdop=self._find_column(columns, ['HDOP']),
                vdop=self._find_column(columns, ['VDOP']),
                pdop=self._find_column(columns, ['PDOP']),
                accuracy_h_mm=self._find_column(columns, ['HAcc_mm', 'HAccuracy_mm']),
                accuracy_v_mm=self._find_column(columns, ['VAcc_mm', 'VAccuracy_mm']),
                ground_speed_mmps=self._find_column(columns, ['GSpeed_mmps', 'GroundSpeed_mmps']),
                heading_motion=self._find_column(columns, ['HeadMot_deg', 'HeadingMotion']),
                heading_vehicle=self._find_column(columns, ['HeadVeh_deg', 'HeadingVehicle']),
                vel_n_mmps=self._find_column(columns, ['VelN_mmps']),
                vel_e_mmps=self._find_column(columns, ['VelE_mmps']),
                vel_d_mmps=self._find_column(columns, ['VelD_mmps']),
                flags=self._find_column(columns, ['Flags', 'flags'])
            )
        else:
            # NMEA or unknown
            profile = CSVColumnProfile(
                name="Auto-NMEA",
                header_rows=header_row,
                timestamp=self._find_column(columns, ['Timestamp', 'Time', 'UTC']),
                lat=self._find_column(columns, ['Latitude', 'Lat', 'lat']),
                lon=self._find_column(columns, ['Longitude', 'Lon', 'lon', 'Long']),
                receiver_id=self._find_column(columns, ['ReceiverID', 'Receiver_ID', 'receiver_id']) if has_receiver_id else None,
                alt_msl=self._find_column(columns, ['AltitudeM', 'Altitude', 'Alt', 'AltMSL']),
                fix_quality=self._find_column(columns, ['FixQuality', 'Quality', 'Fix']),
                sat_count=self._find_column(columns, ['NumSats', 'Satellites', 'SatCount']),
                hdop=self._find_column(columns, ['HDOP']),
                vdop=self._find_column(columns, ['VDOP']),
                pdop=self._find_column(columns, ['PDOP']),
                speed_knots=self._find_column(columns, ['SpeedKnots', 'Speed_kts']),
                course_deg=self._find_column(columns, ['CourseDeg', 'Course', 'Heading']),
                geoid_sep=self._find_column(columns, ['GeoidSepM', 'GeoidSep']),
                mag_var=self._find_column(columns, ['MagVarDeg', 'MagVar']),
                nav_mode=self._find_column(columns, ['NavMode'])
            )
        
        return profile, protocol
    
    def _find_column(self, columns: List[str], candidates: List[str]) -> Optional[str]:
        """Find first matching column name from candidates"""
        for candidate in candidates:
            if candidate in columns:
                return candidate
        return None
    
    def _parse_timestamp(self, ts_str: str) -> Optional[datetime]:
        """Parse various timestamp formats"""
        if not ts_str or ts_str.strip() == '':
            return None
            
        ts_str = ts_str.strip()
        
        # Common formats
        formats = [
            '%Y-%m-%dT%H:%M:%S.%fZ',      # ISO with Z
            '%Y-%m-%dT%H:%M:%S.%f',        # ISO without Z
            '%Y-%m-%dT%H:%M:%SZ',          # ISO no ms with Z
            '%Y-%m-%dT%H:%M:%S',           # ISO no ms
            '%Y-%m-%d %H:%M:%S.%f',        # Space separator
            '%Y-%m-%d %H:%M:%S',           # Space separator no ms
            '%d/%m/%Y %H:%M:%S',           # DD/MM/YYYY
            '%m/%d/%Y %H:%M:%S',           # MM/DD/YYYY
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(ts_str, fmt)
            except ValueError:
                continue
        
        logger.warning(f"Could not parse timestamp: {ts_str}")
        return None
    
    def _safe_float(self, value: str) -> Optional[float]:
        """Safely parse float, returning None on failure"""
        if not value or value.strip() == '' or value.strip().lower() == 'nan':
            return None
        try:
            return float(value.strip())
        except (ValueError, TypeError):
            return None
    
    def _safe_int(self, value: str) -> Optional[int]:
        """Safely parse int, returning None on failure"""
        if not value or value.strip() == '':
            return None
        try:
            return int(float(value.strip()))
        except (ValueError, TypeError):
            return None
    
    def import_file(
        self,
        filepath: Path,
        default_receiver_id: Optional[str] = None,
        profile: Optional[CSVColumnProfile] = None
    ) -> Generator[GNSSPosition, None, None]:
        """
        Import positions from a CSV file.
        
        Args:
            filepath: Path to CSV file
            default_receiver_id: Receiver ID to use if not in CSV (e.g., from filename)
            profile: Column mapping profile. If None, auto-detect.
            
        Yields:
            GNSSPosition objects
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"CSV file not found: {filepath}")
        
        # Auto-detect format if no profile provided
        if profile is None and self.profile is None:
            profile, protocol = self.detect_format(filepath)
            logger.info(f"Auto-detected format: {profile.name}, protocol: {protocol.value}")
        else:
            profile = profile or self.profile
            protocol = ProtocolType.UNKNOWN
        
        # Derive receiver ID from filename if not specified
        if default_receiver_id is None and profile.receiver_id is None:
            # Try to extract from filename
            stem = filepath.stem
            match = re.search(r'(rcvr_?\d+|receiver_?\d+|gps_?\d+)', stem, re.IGNORECASE)
            if match:
                default_receiver_id = match.group(1).lower()
            else:
                default_receiver_id = stem
        
        with open(filepath, 'r', encoding='utf-8-sig') as f:
            # Skip metadata header rows
            for _ in range(profile.header_rows):
                next(f)
            
            reader = csv.DictReader(f)
            
            for row_num, row in enumerate(reader, start=profile.header_rows + 2):
                try:
                    # Get receiver ID
                    if profile.receiver_id and profile.receiver_id in row:
                        receiver_id = row[profile.receiver_id].strip()
                    else:
                        receiver_id = default_receiver_id or "unknown"
                    
                    # Parse timestamp
                    ts = self._parse_timestamp(row.get(profile.timestamp, ''))
                    if ts is None:
                        logger.warning(f"Row {row_num}: Invalid timestamp, skipping")
                        continue
                    
                    # Parse coordinates
                    lat = self._safe_float(row.get(profile.lat, ''))
                    lon = self._safe_float(row.get(profile.lon, ''))
                    
                    if lat is None or lon is None:
                        logger.warning(f"Row {row_num}: Missing coordinates, skipping")
                        continue
                    
                    # Basic validation
                    if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
                        logger.warning(f"Row {row_num}: Invalid coordinates ({lat}, {lon}), skipping")
                        continue
                    
                    # Parse altitude
                    alt_msl = self._safe_float(row.get(profile.alt_msl, '')) if profile.alt_msl else None
                    alt_hae = self._safe_float(row.get(profile.alt_hae, '')) if profile.alt_hae else None
                    
                    # Parse fix type
                    fix_type = FixType.NO_FIX
                    if profile.fix_type and profile.fix_type in row:
                        ft = self._safe_int(row[profile.fix_type])
                        if ft is not None:
                            fix_type = FixType.from_value(ft)
                    elif profile.fix_quality and profile.fix_quality in row:
                        fq = self._safe_int(row[profile.fix_quality])
                        if fq is not None:
                            # NMEA fix quality mapping
                            nmea_map = {0: 0, 1: 2, 2: 4, 4: 5, 5: 6}
                            fix_type = FixType.from_value(nmea_map.get(fq, 0))
                    
                    # Parse carrier solution
                    carrier = 0
                    if profile.carrier_solution and profile.carrier_solution in row:
                        carrier = self._safe_int(row[profile.carrier_solution]) or 0
                    
                    # Parse satellite count
                    sat_count = 0
                    if profile.sat_count and profile.sat_count in row:
                        sat_count = self._safe_int(row[profile.sat_count]) or 0
                    
                    # Parse DOP values
                    hdop = self._safe_float(row.get(profile.hdop, '')) if profile.hdop else None
                    vdop = self._safe_float(row.get(profile.vdop, '')) if profile.vdop else None
                    pdop = self._safe_float(row.get(profile.pdop, '')) if profile.pdop else None
                    
                    # Parse accuracy (convert mm to m if needed)
                    accuracy_h = None
                    if profile.accuracy_h_mm and profile.accuracy_h_mm in row:
                        val = self._safe_float(row[profile.accuracy_h_mm])
                        if val is not None:
                            accuracy_h = val / 1000.0
                    elif profile.accuracy_h_m and profile.accuracy_h_m in row:
                        accuracy_h = self._safe_float(row[profile.accuracy_h_m])
                    
                    accuracy_v = None
                    if profile.accuracy_v_mm and profile.accuracy_v_mm in row:
                        val = self._safe_float(row[profile.accuracy_v_mm])
                        if val is not None:
                            accuracy_v = val / 1000.0
                    elif profile.accuracy_v_m and profile.accuracy_v_m in row:
                        accuracy_v = self._safe_float(row[profile.accuracy_v_m])
                    
                    # Parse velocity
                    speed_mps = None
                    if profile.ground_speed_mmps and profile.ground_speed_mmps in row:
                        val = self._safe_float(row[profile.ground_speed_mmps])
                        if val is not None:
                            speed_mps = val / 1000.0
                    elif profile.speed_mps and profile.speed_mps in row:
                        speed_mps = self._safe_float(row[profile.speed_mps])
                    elif profile.speed_knots and profile.speed_knots in row:
                        val = self._safe_float(row[profile.speed_knots])
                        if val is not None:
                            speed_mps = val * 0.514444  # knots to m/s
                    
                    # Parse heading
                    heading = None
                    if profile.heading_motion and profile.heading_motion in row:
                        heading = self._safe_float(row[profile.heading_motion])
                    elif profile.course_deg and profile.course_deg in row:
                        heading = self._safe_float(row[profile.course_deg])
                    
                    # Parse velocity components
                    vel_n = None
                    vel_e = None
                    vel_d = None
                    if profile.vel_n_mmps and profile.vel_n_mmps in row:
                        val = self._safe_float(row[profile.vel_n_mmps])
                        if val is not None:
                            vel_n = val / 1000.0
                    if profile.vel_e_mmps and profile.vel_e_mmps in row:
                        val = self._safe_float(row[profile.vel_e_mmps])
                        if val is not None:
                            vel_e = val / 1000.0
                    if profile.vel_d_mmps and profile.vel_d_mmps in row:
                        val = self._safe_float(row[profile.vel_d_mmps])
                        if val is not None:
                            vel_d = val / 1000.0
                    
                    # Track receiver
                    if receiver_id not in self.detected_receivers:
                        self.detected_receivers[receiver_id] = ReceiverConfig(
                            id=receiver_id,
                            nickname=receiver_id,
                            protocol=protocol
                        )
                    
                    # Create position object
                    position = GNSSPosition(
                        timestamp=ts,
                        receiver_id=receiver_id,
                        lat=lat,
                        lon=lon,
                        alt_msl=alt_msl,
                        alt_hae=alt_hae,
                        fix_type=fix_type,
                        carrier_solution=carrier,
                        sat_count=sat_count,
                        hdop=hdop,
                        vdop=vdop,
                        pdop=pdop,
                        accuracy_h_m=accuracy_h,
                        accuracy_v_m=accuracy_v,
                        speed_mps=speed_mps,
                        heading_deg=heading,
                        vel_n_mps=vel_n,
                        vel_e_mps=vel_e,
                        vel_d_mps=vel_d,
                        protocol=protocol,
                        is_valid=True
                    )
                    
                    yield position
                    
                except Exception as e:
                    logger.error(f"Row {row_num}: Error parsing row: {e}")
                    continue
    
    def import_session(
        self,
        filepaths: List[Path],
        session_id: str,
        username: str = "import",
        role: str = "analyst"
    ) -> GNSSSession:
        """
        Import multiple CSV files into a single session.
        
        Args:
            filepaths: List of CSV file paths
            session_id: Session identifier
            username: Username for session
            role: User role
            
        Returns:
            GNSSSession with all imported data
        """
        session = GNSSSession(
            session_id=session_id,
            username=username,
            role=role,
            source_files=[str(p) for p in filepaths]
        )
        
        for filepath in filepaths:
            logger.info(f"Importing: {filepath}")
            
            try:
                for position in self.import_file(Path(filepath)):
                    session.positions.append(position)
            except Exception as e:
                logger.error(f"Failed to import {filepath}: {e}")
                continue
        
        # Copy detected receivers to session
        session.receivers = self.detected_receivers.copy()
        
        # Set time bounds
        if session.positions:
            session.positions.sort(key=lambda p: p.timestamp)
            session.start_time = session.positions[0].timestamp
            session.end_time = session.positions[-1].timestamp
        
        logger.info(f"Imported {len(session.positions)} positions from {len(session.receivers)} receivers")
        
        return session


def create_option_b_csv(
    session: GNSSSession,
    output_path: Path,
    include_ubx_fields: bool = True
) -> None:
    """
    Export session data to Option B format CSV (multi-receiver with ReceiverID column).
    
    Args:
        session: GNSSSession to export
        output_path: Path for output CSV
        include_ubx_fields: Include UBX-specific fields
    """
    fieldnames = [
        'Timestamp', 'ReceiverID', 'Latitude', 'Longitude',
        'AltitudeMSL', 'AltitudeHAE', 'FixType', 'SatCount',
        'HDOP', 'VDOP', 'PDOP', 'AccuracyH_m', 'AccuracyV_m',
        'Speed_mps', 'Heading_deg'
    ]
    
    if include_ubx_fields:
        fieldnames.extend([
            'CarrierSolution', 'VelN_mps', 'VelE_mps', 'VelD_mps'
        ])
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for pos in sorted(session.positions, key=lambda p: (p.timestamp, p.receiver_id)):
            row = {
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
                'Heading_deg': f"{pos.heading_deg:.2f}" if pos.heading_deg else ''
            }
            
            if include_ubx_fields:
                row.update({
                    'CarrierSolution': pos.carrier_solution,
                    'VelN_mps': f"{pos.vel_n_mps:.3f}" if pos.vel_n_mps else '',
                    'VelE_mps': f"{pos.vel_e_mps:.3f}" if pos.vel_e_mps else '',
                    'VelD_mps': f"{pos.vel_d_mps:.3f}" if pos.vel_d_mps else ''
                })
            
            writer.writerow(row)
    
    logger.info(f"Exported {len(session.positions)} positions to {output_path}")
