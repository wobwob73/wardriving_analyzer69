#!/usr/bin/env python3
"""
GNSS Link - Data Models and Types
Version: 4.6.1

Defines data structures for GNSS receiver data, session management,
and variance analysis.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum
import math


class FixType(Enum):
    """GNSS fix type enumeration"""
    NO_FIX = 0
    FIX_2D = 1
    FIX_3D = 2
    DGPS = 4
    RTK_FLOAT = 5
    RTK_FIXED = 6
    
    @classmethod
    def from_value(cls, value: int) -> 'FixType':
        """Get FixType from integer value"""
        for fix_type in cls:
            if fix_type.value == value:
                return fix_type
        return cls.NO_FIX
    
    @property
    def name_friendly(self) -> str:
        """Human-readable name"""
        names = {
            0: "No Fix",
            1: "2D Fix",
            2: "3D Fix",
            4: "DGPS",
            5: "RTK Float",
            6: "RTK Fixed"
        }
        return names.get(self.value, "Unknown")


class ReceiverStatus(Enum):
    """Receiver connection status"""
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    POSITION_JUMP = "position_jump"
    UNKNOWN = "unknown"


class ProtocolType(Enum):
    """GNSS protocol type"""
    NMEA = "NMEA"
    UBX = "UBX"
    MIXED = "MIXED"
    UNKNOWN = "UNKNOWN"


@dataclass
class GNSSPosition:
    """Single GNSS position fix from a receiver"""
    timestamp: datetime
    receiver_id: str
    lat: float
    lon: float
    alt_msl: Optional[float] = None
    alt_hae: Optional[float] = None  # Height above ellipsoid
    fix_type: FixType = FixType.NO_FIX
    carrier_solution: int = 0
    sat_count: int = 0
    hdop: Optional[float] = None
    vdop: Optional[float] = None
    pdop: Optional[float] = None
    accuracy_h_m: Optional[float] = None  # Horizontal accuracy in meters
    accuracy_v_m: Optional[float] = None  # Vertical accuracy in meters
    speed_mps: Optional[float] = None
    heading_deg: Optional[float] = None
    vel_n_mps: Optional[float] = None  # Velocity North
    vel_e_mps: Optional[float] = None  # Velocity East
    vel_d_mps: Optional[float] = None  # Velocity Down
    
    # Raw data storage
    raw_nmea: Optional[str] = None
    raw_ubx: Optional[bytes] = None
    
    # Metadata
    protocol: ProtocolType = ProtocolType.UNKNOWN
    is_valid: bool = True
    notes: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'receiver_id': self.receiver_id,
            'lat': self.lat,
            'lon': self.lon,
            'alt_msl': self.alt_msl,
            'alt_hae': self.alt_hae,
            'fix_type': self.fix_type.value,
            'fix_type_name': self.fix_type.name_friendly,
            'carrier_solution': self.carrier_solution,
            'sat_count': self.sat_count,
            'hdop': self.hdop,
            'vdop': self.vdop,
            'pdop': self.pdop,
            'accuracy_h_m': self.accuracy_h_m,
            'accuracy_v_m': self.accuracy_v_m,
            'speed_mps': self.speed_mps,
            'heading_deg': self.heading_deg,
            'protocol': self.protocol.value,
            'is_valid': self.is_valid
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GNSSPosition':
        """Create from dictionary"""
        return cls(
            timestamp=datetime.fromisoformat(data['timestamp']),
            receiver_id=data['receiver_id'],
            lat=data['lat'],
            lon=data['lon'],
            alt_msl=data.get('alt_msl'),
            alt_hae=data.get('alt_hae'),
            fix_type=FixType.from_value(data.get('fix_type', 0)),
            carrier_solution=data.get('carrier_solution', 0),
            sat_count=data.get('sat_count', 0),
            hdop=data.get('hdop'),
            vdop=data.get('vdop'),
            pdop=data.get('pdop'),
            accuracy_h_m=data.get('accuracy_h_m'),
            accuracy_v_m=data.get('accuracy_v_m'),
            speed_mps=data.get('speed_mps'),
            heading_deg=data.get('heading_deg'),
            protocol=ProtocolType(data.get('protocol', 'UNKNOWN')),
            is_valid=data.get('is_valid', True)
        )


@dataclass
class ReceiverConfig:
    """Configuration for a single GNSS receiver"""
    id: str
    nickname: str = ""
    device_name: str = ""
    protocol: ProtocolType = ProtocolType.UNKNOWN
    chip_generation: Optional[str] = None  # M8, M9, F9, M10, X20
    refresh_rate_hz: float = 1.0
    is_truth_receiver: bool = False
    enabled: bool = True
    icon_color: str = "#00FF00"
    com_port: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'nickname': self.nickname,
            'device_name': self.device_name,
            'protocol': self.protocol.value,
            'chip_generation': self.chip_generation,
            'refresh_rate_hz': self.refresh_rate_hz,
            'is_truth_receiver': self.is_truth_receiver,
            'enabled': self.enabled,
            'icon_color': self.icon_color,
            'com_port': self.com_port
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ReceiverConfig':
        """Create from dictionary"""
        return cls(
            id=data['id'],
            nickname=data.get('nickname', ''),
            device_name=data.get('device_name', ''),
            protocol=ProtocolType(data.get('protocol', 'UNKNOWN')),
            chip_generation=data.get('chip_generation'),
            refresh_rate_hz=data.get('refresh_rate_hz', 1.0),
            is_truth_receiver=data.get('is_truth_receiver', False),
            enabled=data.get('enabled', True),
            icon_color=data.get('icon_color', '#00FF00'),
            com_port=data.get('com_port')
        )


@dataclass
class ReceiverStatistics:
    """Computed statistics for a single receiver"""
    receiver_id: str
    position_count: int = 0
    
    # Position statistics
    lat_mean: float = 0.0
    lat_std: float = 0.0
    lon_mean: float = 0.0
    lon_std: float = 0.0
    alt_mean: float = 0.0
    alt_std: float = 0.0
    
    # Fix quality
    fix_type_distribution: Dict[int, int] = field(default_factory=dict)
    sat_count_min: int = 0
    sat_count_max: int = 0
    sat_count_mean: float = 0.0
    
    # DOP statistics
    hdop_min: float = 0.0
    hdop_max: float = 0.0
    hdop_mean: float = 0.0
    vdop_min: float = 0.0
    vdop_max: float = 0.0
    vdop_mean: float = 0.0
    pdop_min: float = 0.0
    pdop_max: float = 0.0
    pdop_mean: float = 0.0
    
    # Deviation from centroid (meters)
    deviation_from_centroid_mean: float = 0.0
    deviation_from_centroid_max: float = 0.0
    deviation_from_centroid_std: float = 0.0
    
    # Deviation from truth (if applicable)
    deviation_from_truth_mean: Optional[float] = None
    deviation_from_truth_max: Optional[float] = None
    
    # Time coverage
    first_timestamp: Optional[datetime] = None
    last_timestamp: Optional[datetime] = None
    data_gaps: int = 0  # Number of disconnect events
    
    # Accuracy metrics
    cep_50: Optional[float] = None  # Circular Error Probable (50%)
    drms_2: Optional[float] = None  # 2DRMS (95%)


@dataclass 
class SessionEvent:
    """Session event (button press, note, etc.)"""
    timestamp: datetime
    event_type: str  # event_start, flight_start, run_start, music_on, etc.
    flight_number: Optional[int] = None
    run_number: Optional[int] = None
    notes: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'event_type': self.event_type,
            'flight_number': self.flight_number,
            'run_number': self.run_number,
            'notes': self.notes,
            'data': self.data
        }


@dataclass
class ReceiverEvent:
    """Receiver status change event"""
    timestamp: datetime
    receiver_id: str
    event_type: ReceiverStatus
    last_lat: Optional[float] = None
    last_lon: Optional[float] = None
    new_lat: Optional[float] = None
    new_lon: Optional[float] = None
    notes: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'receiver_id': self.receiver_id,
            'event_type': self.event_type.value,
            'last_lat': self.last_lat,
            'last_lon': self.last_lon,
            'new_lat': self.new_lat,
            'new_lon': self.new_lon,
            'notes': self.notes
        }


@dataclass
class MapMarker:
    """Persistent map marker"""
    id: int
    label: str
    lat: float
    lon: float
    icon: str = "marker"
    radius_100m: bool = False
    radius_500m: bool = False
    radius_1km: bool = False
    radius_1_5km: bool = False
    radius_2km: bool = False
    circle_opacity: float = 0.3
    created_by: Optional[str] = None
    created_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'label': self.label,
            'lat': self.lat,
            'lon': self.lon,
            'icon': self.icon,
            'radius_100m': self.radius_100m,
            'radius_500m': self.radius_500m,
            'radius_1km': self.radius_1km,
            'radius_1_5km': self.radius_1_5km,
            'radius_2km': self.radius_2km,
            'circle_opacity': self.circle_opacity,
            'created_by': self.created_by,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


@dataclass
class GNSSSession:
    """Complete GNSS Link session"""
    session_id: str
    username: str
    role: str  # admin, analyst, observer
    
    # Session metadata
    radio_number: Optional[str] = None
    battery_pair_id: Optional[str] = None
    general_notes: str = ""
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    # Data collections
    receivers: Dict[str, ReceiverConfig] = field(default_factory=dict)
    positions: List[GNSSPosition] = field(default_factory=list)
    events: List[SessionEvent] = field(default_factory=list)
    receiver_events: List[ReceiverEvent] = field(default_factory=list)
    markers: List[MapMarker] = field(default_factory=list)
    
    # Computed statistics (populated by analysis)
    receiver_stats: Dict[str, ReceiverStatistics] = field(default_factory=dict)
    
    # Session state
    is_live: bool = False
    source_files: List[str] = field(default_factory=list)
    
    def get_positions_for_receiver(self, receiver_id: str) -> List[GNSSPosition]:
        """Get all positions for a specific receiver"""
        return [p for p in self.positions if p.receiver_id == receiver_id]
    
    def get_positions_in_timerange(
        self, 
        start: datetime, 
        end: datetime
    ) -> List[GNSSPosition]:
        """Get positions within a time range"""
        return [p for p in self.positions if start <= p.timestamp <= end]
    
    def get_truth_receivers(self) -> List[str]:
        """Get list of truth receiver IDs"""
        return [r.id for r in self.receivers.values() if r.is_truth_receiver]
    
    def get_active_receivers(self) -> List[str]:
        """Get list of enabled receiver IDs"""
        return [r.id for r in self.receivers.values() if r.enabled]


# Utility functions

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great circle distance between two points in meters.
    
    Args:
        lat1, lon1: First point coordinates in degrees
        lat2, lon2: Second point coordinates in degrees
        
    Returns:
        Distance in meters
    """
    R = 6371000  # Earth's radius in meters
    
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    
    a = (math.sin(delta_phi / 2) ** 2 + 
         math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    return R * c


def meters_to_degrees_lat(meters: float, lat: float) -> float:
    """Convert meters to degrees latitude at a given latitude"""
    return meters / 111320.0


def meters_to_degrees_lon(meters: float, lat: float) -> float:
    """Convert meters to degrees longitude at a given latitude"""
    return meters / (111320.0 * math.cos(math.radians(lat)))
