"""
Polymorphic Data Source Interface for Wardriving Analyzer

This module provides a unified interface for ingesting wardriving data from
multiple sources: CSV files, MQTT streams, serial connections, databases, etc.

Usage:
    # All sources implement the same interface
    csv_source = CSVDataSource("scan_monday.csv")
    mqtt_source = MQTTDataSource(broker="192.168.1.100")
    folder_source = CSVFolderSource("/data/wardrive_runs/")
    
    # The analyzer doesn't care about source type
    for detection in source.get_detections():
        analyzer.process_detection(detection)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterator, Optional, List, Dict, Any
from datetime import datetime
from pathlib import Path
import pandas as pd
import logging
import os
import io

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """
    Unified detection record - the common currency across all data sources.
    All data sources must convert their native format to this structure.
    """
    mac: str
    name: str = ""
    signal_type: str = "WIFI"
    rssi: int = -100
    lat: float = 0.0
    lon: float = 0.0
    has_gps: bool = False
    worker: str = "unknown"
    channel: str = ""
    timestamp: str = ""
    run_id: str = ""
    security: str = ""
    
    # Additional metadata fields
    extra: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        # Normalize MAC address
        self.mac = self.mac.lower().strip().strip('"')
        
        # Normalize name
        if self.name == 'nan' or not self.name:
            self.name = ""
        self.name = self.name.strip().strip('"')
        
        # Normalize signal type
        self.signal_type = self.signal_type.upper()
        
        # Determine GPS validity
        if not self.has_gps:
            self.has_gps = (self.lat != 0.0 and self.lon != 0.0)


@dataclass
class RunMetadata:
    """Metadata about a data collection run/session"""
    run_id: str
    source_type: str  # "csv", "mqtt", "serial", etc.
    source_name: str  # Filename, broker address, etc.
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    detection_count: int = 0
    gps_detection_count: int = 0
    unique_aps: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class DataSource(ABC):
    """
    Abstract base class for all data sources.
    
    This is the polymorphic interface - any class that implements this
    interface can be used interchangeably by the analysis engine.
    """
    
    @abstractmethod
    def get_detections(self) -> Iterator[Detection]:
        """
        Yield Detection objects from this source.
        This is the core interface method.
        """
        pass
    
    @abstractmethod
    def get_metadata(self) -> RunMetadata:
        """Return metadata about this data source/run"""
        pass
    
    @property
    @abstractmethod
    def source_type(self) -> str:
        """Return the type of this source (csv, mqtt, serial, etc.)"""
        pass
    
    def __repr__(self):
        meta = self.get_metadata()
        return f"<{self.__class__.__name__} run_id={meta.run_id} source={meta.source_name}>"


class CSVDataSource(DataSource):
    """
    Data source for a single CSV file.
    Handles various CSV formats and normalizes to Detection objects.
    """
    
    def __init__(self, filepath: str, run_id: Optional[str] = None):
        self.filepath = Path(filepath)
        self.run_id = run_id or f"csv_{self.filepath.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self._metadata: Optional[RunMetadata] = None
        self._df: Optional[pd.DataFrame] = None
        
    @property
    def source_type(self) -> str:
        return "csv"
    
    def _load_csv(self) -> pd.DataFrame:
        """Load and normalize CSV file"""
        if self._df is not None:
            return self._df
            
        # Read file and normalize line endings
        with open(self.filepath, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read().replace('\r\n', '\n').replace('\r', '\n')
        
        self._df = pd.read_csv(
            io.StringIO(content), 
            on_bad_lines='skip', 
            quotechar='"', 
            index_col=False
        )
        logger.info(f"Loaded CSV: {self.filepath} ({len(self._df)} rows)")
        return self._df
    
    def get_detections(self) -> Iterator[Detection]:
        """Yield Detection objects from CSV rows"""
        df = self._load_csv()
        
        detection_count = 0
        gps_count = 0
        unique_macs = set()
        first_ts = None
        last_ts = None
        
        for _, row in df.iterrows():
            # Skip IMU/ACCEL data
            signal_type = str(row.get('signal_type', 'UNKNOWN')).upper()
            if signal_type in ['IMU', 'ACCEL']:
                continue
            
            # Extract MAC
            mac = str(row.get('identifier', '')).lower().strip().strip('"')
            if not mac:
                continue
            
            # Extract name
            name = str(row.get('name', '')).strip().strip('"')
            if name == 'nan' or pd.isna(row.get('name')):
                name = ""
            
            # Handle numeric fields with NaN
            rssi_val = row.get('rssi_dbm', -100)
            rssi = int(rssi_val) if pd.notna(rssi_val) else -100
            
            lat_val = row.get('lat', 0.0)
            lat = float(lat_val) if pd.notna(lat_val) else 0.0
            
            lon_val = row.get('lon', 0.0)
            lon = float(lon_val) if pd.notna(lon_val) else 0.0
            
            fix_ok_val = row.get('fix_ok', 0)
            fix_ok = int(fix_ok_val) if pd.notna(fix_ok_val) else 0
            
            has_gps = (fix_ok == 1 and lat != 0.0 and lon != 0.0)
            
            # Worker
            worker = str(row.get('worker', 'unknown'))
            if worker == 'nan' or pd.isna(row.get('worker')):
                worker = 'unknown'
            
            # Channel
            channel_val = row.get('channel', '')
            channel = str(channel_val) if pd.notna(channel_val) else ''
            
            # Timestamp
            timestamp_val = row.get('iso8601_utc', '')
            timestamp = str(timestamp_val) if pd.notna(timestamp_val) else ''
            
            # Track first/last timestamp
            if timestamp:
                if first_ts is None or timestamp < first_ts:
                    first_ts = timestamp
                if last_ts is None or timestamp > last_ts:
                    last_ts = timestamp
            
            # Security (try multiple column names)
            security = ''
            for key in ['security', 'encryption', 'auth', 'privacy', 'capabilities', 'wifi_security', 'sec']:
                if key in row and pd.notna(row.get(key)):
                    security = str(row.get(key))
                    break
            
            # Track stats
            detection_count += 1
            if has_gps:
                gps_count += 1
            unique_macs.add(mac)
            
            yield Detection(
                mac=mac,
                name=name,
                signal_type=signal_type,
                rssi=rssi,
                lat=lat,
                lon=lon,
                has_gps=has_gps,
                worker=worker,
                channel=channel,
                timestamp=timestamp,
                run_id=self.run_id,
                security=security
            )
        
        # Update metadata after iteration
        self._metadata = RunMetadata(
            run_id=self.run_id,
            source_type=self.source_type,
            source_name=str(self.filepath),
            start_time=datetime.fromisoformat(first_ts.replace('Z', '+00:00')) if first_ts else None,
            end_time=datetime.fromisoformat(last_ts.replace('Z', '+00:00')) if last_ts else None,
            detection_count=detection_count,
            gps_detection_count=gps_count,
            unique_aps=len(unique_macs)
        )
    
    def get_metadata(self) -> RunMetadata:
        if self._metadata is None:
            # Force iteration to populate metadata
            list(self.get_detections())
        return self._metadata or RunMetadata(
            run_id=self.run_id,
            source_type=self.source_type,
            source_name=str(self.filepath)
        )


class CSVFolderSource(DataSource):
    """
    Data source for a folder of CSV files.
    Treats each CSV as a separate run for multi-day analysis.
    """
    
    def __init__(self, folder_path: str, pattern: str = "*.csv"):
        self.folder_path = Path(folder_path)
        self.pattern = pattern
        self._sources: List[CSVDataSource] = []
        self._metadata: Optional[RunMetadata] = None
        self._discover_files()
    
    @property
    def source_type(self) -> str:
        return "csv_folder"
    
    def _discover_files(self):
        """Discover CSV files in folder"""
        if not self.folder_path.exists():
            logger.warning(f"Folder not found: {self.folder_path}")
            return
        
        csv_files = sorted(self.folder_path.glob(self.pattern))
        for csv_file in csv_files:
            self._sources.append(CSVDataSource(str(csv_file)))
        
        logger.info(f"Discovered {len(self._sources)} CSV files in {self.folder_path}")
    
    def get_detections(self) -> Iterator[Detection]:
        """Yield detections from all CSV files"""
        total_detections = 0
        total_gps = 0
        all_macs = set()
        
        for source in self._sources:
            for detection in source.get_detections():
                total_detections += 1
                if detection.has_gps:
                    total_gps += 1
                all_macs.add(detection.mac)
                yield detection
        
        self._metadata = RunMetadata(
            run_id=f"folder_{self.folder_path.name}",
            source_type=self.source_type,
            source_name=str(self.folder_path),
            detection_count=total_detections,
            gps_detection_count=total_gps,
            unique_aps=len(all_macs),
            metadata={'file_count': len(self._sources)}
        )
    
    def get_metadata(self) -> RunMetadata:
        if self._metadata is None:
            list(self.get_detections())
        return self._metadata or RunMetadata(
            run_id=f"folder_{self.folder_path.name}",
            source_type=self.source_type,
            source_name=str(self.folder_path)
        )
    
    @property
    def sources(self) -> List[CSVDataSource]:
        """Access individual CSV sources"""
        return self._sources


class MultiSource(DataSource):
    """
    Composite data source that combines multiple sources.
    Useful for analyzing data from different sources together.
    """
    
    def __init__(self, sources: Optional[List[DataSource]] = None):
        self._sources: List[DataSource] = sources or []
        self._metadata: Optional[RunMetadata] = None
    
    @property
    def source_type(self) -> str:
        return "multi"
    
    def add_source(self, source: DataSource):
        """Add a data source"""
        self._sources.append(source)
        self._metadata = None  # Invalidate cached metadata
    
    def add_csv(self, filepath: str, run_id: Optional[str] = None) -> CSVDataSource:
        """Convenience method to add a CSV file"""
        source = CSVDataSource(filepath, run_id)
        self.add_source(source)
        return source
    
    def add_folder(self, folder_path: str, pattern: str = "*.csv") -> CSVFolderSource:
        """Convenience method to add a folder of CSVs"""
        source = CSVFolderSource(folder_path, pattern)
        self.add_source(source)
        return source
    
    def get_detections(self) -> Iterator[Detection]:
        """Yield detections from all sources"""
        total_detections = 0
        total_gps = 0
        all_macs = set()
        
        for source in self._sources:
            for detection in source.get_detections():
                total_detections += 1
                if detection.has_gps:
                    total_gps += 1
                all_macs.add(detection.mac)
                yield detection
        
        self._metadata = RunMetadata(
            run_id=f"multi_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            source_type=self.source_type,
            source_name=f"{len(self._sources)} sources",
            detection_count=total_detections,
            gps_detection_count=total_gps,
            unique_aps=len(all_macs),
            metadata={'source_count': len(self._sources)}
        )
    
    def get_metadata(self) -> RunMetadata:
        if self._metadata is None:
            list(self.get_detections())
        return self._metadata or RunMetadata(
            run_id="multi",
            source_type=self.source_type,
            source_name=f"{len(self._sources)} sources"
        )
    
    @property
    def sources(self) -> List[DataSource]:
        return self._sources
    
    def clear(self):
        """Clear all sources"""
        self._sources.clear()
        self._metadata = None


class MQTTDataSource(DataSource):
    """
    Data source for live MQTT streams.
    Collects detections in real-time from MQTT broker.
    """
    
    def __init__(self, broker_host: str = "localhost", broker_port: int = 1883,
                 topics: Optional[List[str]] = None, run_id: Optional[str] = None):
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.topics = topics or ["wardriving/+/detection", "wardriving/+/batch"]
        self.run_id = run_id or f"mqtt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self._detections: List[Detection] = []
        self._connected = False
        self._metadata: Optional[RunMetadata] = None
    
    @property
    def source_type(self) -> str:
        return "mqtt"
    
    def add_detection(self, data: Dict[str, Any]):
        """Add a detection from MQTT message (called by MQTT handler)"""
        try:
            mac = str(data.get('identifier', '')).lower()
            if not mac:
                return
            
            signal_type = str(data.get('signal_type', 'WIFI')).upper()
            if signal_type in ['IMU', 'ACCEL']:
                return
            
            detection = Detection(
                mac=mac,
                name=str(data.get('name', '')),
                signal_type=signal_type,
                rssi=int(data.get('rssi_dbm', -100)),
                lat=float(data.get('lat', 0.0)),
                lon=float(data.get('lon', 0.0)),
                has_gps=int(data.get('fix_ok', 0)) == 1,
                worker=str(data.get('worker', 'unknown')),
                channel=str(data.get('channel', '')),
                timestamp=str(data.get('iso8601_utc', datetime.now().isoformat())),
                run_id=self.run_id,
                security=str(data.get('security', ''))
            )
            self._detections.append(detection)
            
        except Exception as e:
            logger.error(f"Error adding MQTT detection: {e}")
    
    def get_detections(self) -> Iterator[Detection]:
        """Yield collected detections"""
        for detection in self._detections:
            yield detection
    
    def get_metadata(self) -> RunMetadata:
        unique_macs = set(d.mac for d in self._detections)
        gps_count = sum(1 for d in self._detections if d.has_gps)
        
        return RunMetadata(
            run_id=self.run_id,
            source_type=self.source_type,
            source_name=f"{self.broker_host}:{self.broker_port}",
            detection_count=len(self._detections),
            gps_detection_count=gps_count,
            unique_aps=len(unique_macs),
            metadata={'connected': self._connected, 'topics': self.topics}
        )
    
    def clear(self):
        """Clear collected detections"""
        self._detections.clear()


# Factory function for creating sources
def create_source(source_type: str, **kwargs) -> DataSource:
    """
    Factory function to create data sources.
    
    Args:
        source_type: Type of source ("csv", "csv_folder", "mqtt", "multi")
        **kwargs: Arguments for the specific source type
    
    Returns:
        DataSource instance
    """
    sources = {
        'csv': CSVDataSource,
        'csv_folder': CSVFolderSource,
        'mqtt': MQTTDataSource,
        'multi': MultiSource
    }
    
    if source_type not in sources:
        raise ValueError(f"Unknown source type: {source_type}. Available: {list(sources.keys())}")
    
    return sources[source_type](**kwargs)
