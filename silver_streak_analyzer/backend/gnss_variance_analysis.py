#!/usr/bin/env python3
"""
GNSS Link - Variance Analysis Engine
Version: 4.8.0

Computes variance metrics for multi-receiver GNSS data including:
- Standard deviation (horizontal/vertical)
- CEP (Circular Error Probable) - 50%
- 2DRMS - 95%
- Individual vs centroid deviation
- Truth receiver comparison
"""

import math
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple, Any
from dataclasses import dataclass, field
import statistics

from gnss_models import (
    GNSSSession, GNSSPosition, ReceiverStatistics, ReceiverConfig,
    FixType, haversine_distance
)

logger = logging.getLogger(__name__)


@dataclass
class CentroidPosition:
    """Computed centroid position from all receivers at a timestamp"""
    timestamp: datetime
    lat: float
    lon: float
    alt: Optional[float] = None
    receiver_count: int = 0
    receiver_ids: List[str] = field(default_factory=list)


@dataclass
class DeviationRecord:
    """Deviation of a single position from reference"""
    timestamp: datetime
    receiver_id: str
    lat: float
    lon: float
    deviation_h_m: float  # Horizontal deviation in meters
    deviation_v_m: Optional[float] = None  # Vertical deviation
    reference_type: str = "centroid"  # "centroid" or "truth"


@dataclass
class VarianceMetrics:
    """Computed variance metrics for the session"""
    # Per-receiver statistics
    receiver_stats: Dict[str, ReceiverStatistics] = field(default_factory=dict)
    
    # Overall metrics (all receivers)
    all_receivers_std_h: float = 0.0  # Horizontal std dev
    all_receivers_std_v: float = 0.0  # Vertical std dev
    all_receivers_cep_50: float = 0.0  # CEP 50%
    all_receivers_2drms: float = 0.0  # 2DRMS 95%
    all_receivers_max_deviation: float = 0.0
    
    # Non-truth vs truth metrics
    truth_comparison_std_h: Optional[float] = None
    truth_comparison_std_v: Optional[float] = None
    truth_comparison_cep_50: Optional[float] = None
    truth_comparison_2drms: Optional[float] = None
    truth_comparison_max_deviation: Optional[float] = None
    truth_receiver_ids: List[str] = field(default_factory=list)
    
    # Time series data for charts
    deviation_records: List[DeviationRecord] = field(default_factory=list)
    centroid_positions: List[CentroidPosition] = field(default_factory=list)


class GNSSVarianceAnalyzer:
    """
    Analyzes GNSS position variance across multiple receivers.
    
    Features:
    - Compute per-receiver statistics
    - Calculate centroid (average) position over time
    - Measure deviation from centroid
    - Compare against truth receivers
    - Compute accuracy metrics (CEP, 2DRMS)
    """
    
    def __init__(
        self,
        time_window_ms: int = 500,  # Window for aligning positions across receivers
        min_receivers_for_centroid: int = 2
    ):
        """
        Initialize analyzer.
        
        Args:
            time_window_ms: Time window (ms) for grouping positions across receivers
            min_receivers_for_centroid: Minimum receivers needed to compute centroid
        """
        self.time_window_ms = time_window_ms
        self.min_receivers_for_centroid = min_receivers_for_centroid
    
    def analyze(self, session: GNSSSession) -> VarianceMetrics:
        """
        Perform full variance analysis on a session.
        
        Args:
            session: GNSSSession with position data
            
        Returns:
            VarianceMetrics with all computed statistics
        """
        metrics = VarianceMetrics()
        
        if not session.positions:
            logger.warning("No positions in session to analyze")
            return metrics
        
        # Get truth receivers
        truth_ids = session.get_truth_receivers()
        metrics.truth_receiver_ids = truth_ids
        
        # Group positions by time window
        time_groups = self._group_by_time(session.positions)
        logger.info(f"Grouped {len(session.positions)} positions into {len(time_groups)} time windows")
        
        # Compute centroids
        metrics.centroid_positions = self._compute_centroids(time_groups)
        logger.info(f"Computed {len(metrics.centroid_positions)} centroid positions")
        
        # Compute truth centroid if truth receivers exist
        truth_centroids = None
        if truth_ids:
            truth_positions = [p for p in session.positions if p.receiver_id in truth_ids]
            truth_groups = self._group_by_time(truth_positions)
            truth_centroids = self._compute_centroids(truth_groups, required_ids=truth_ids)
            logger.info(f"Computed {len(truth_centroids)} truth centroid positions")
        
        # Compute deviations
        all_deviations = []
        truth_deviations = []
        
        for pos in session.positions:
            # Find matching centroid
            centroid = self._find_nearest_centroid(pos.timestamp, metrics.centroid_positions)
            if centroid and centroid.receiver_count >= self.min_receivers_for_centroid:
                dev_h = haversine_distance(pos.lat, pos.lon, centroid.lat, centroid.lon)
                dev_v = None
                if pos.alt_msl is not None and centroid.alt is not None:
                    dev_v = abs(pos.alt_msl - centroid.alt)
                
                record = DeviationRecord(
                    timestamp=pos.timestamp,
                    receiver_id=pos.receiver_id,
                    lat=pos.lat,
                    lon=pos.lon,
                    deviation_h_m=dev_h,
                    deviation_v_m=dev_v,
                    reference_type="centroid"
                )
                all_deviations.append(record)
                metrics.deviation_records.append(record)
            
            # Compute deviation from truth if applicable
            if truth_centroids and pos.receiver_id not in truth_ids:
                truth_centroid = self._find_nearest_centroid(pos.timestamp, truth_centroids)
                if truth_centroid and truth_centroid.receiver_count > 0:
                    dev_h = haversine_distance(pos.lat, pos.lon, truth_centroid.lat, truth_centroid.lon)
                    dev_v = None
                    if pos.alt_msl is not None and truth_centroid.alt is not None:
                        dev_v = abs(pos.alt_msl - truth_centroid.alt)
                    
                    truth_deviations.append({
                        'h': dev_h,
                        'v': dev_v,
                        'receiver_id': pos.receiver_id
                    })
        
        # Compute overall metrics from all receivers
        if all_deviations:
            h_devs = [d.deviation_h_m for d in all_deviations]
            v_devs = [d.deviation_v_m for d in all_deviations if d.deviation_v_m is not None]
            
            metrics.all_receivers_std_h = statistics.stdev(h_devs) if len(h_devs) > 1 else 0.0
            metrics.all_receivers_std_v = statistics.stdev(v_devs) if len(v_devs) > 1 else 0.0
            metrics.all_receivers_max_deviation = max(h_devs)
            metrics.all_receivers_cep_50 = self._compute_cep(h_devs, 0.50)
            metrics.all_receivers_2drms = self._compute_2drms(h_devs)
        
        # Compute truth comparison metrics
        if truth_deviations:
            h_devs = [d['h'] for d in truth_deviations]
            v_devs = [d['v'] for d in truth_deviations if d['v'] is not None]
            
            metrics.truth_comparison_std_h = statistics.stdev(h_devs) if len(h_devs) > 1 else 0.0
            metrics.truth_comparison_std_v = statistics.stdev(v_devs) if len(v_devs) > 1 else 0.0
            metrics.truth_comparison_max_deviation = max(h_devs)
            metrics.truth_comparison_cep_50 = self._compute_cep(h_devs, 0.50)
            metrics.truth_comparison_2drms = self._compute_2drms(h_devs)
        
        # Compute per-receiver statistics
        for receiver_id in session.receivers.keys():
            stats = self._compute_receiver_stats(session, receiver_id, all_deviations, truth_deviations)
            metrics.receiver_stats[receiver_id] = stats
            session.receiver_stats[receiver_id] = stats
        
        logger.info(f"Analysis complete: {len(metrics.receiver_stats)} receivers analyzed")
        return metrics
    
    def _group_by_time(
        self, 
        positions: List[GNSSPosition]
    ) -> Dict[int, List[GNSSPosition]]:
        """
        Group positions into time windows.
        
        Args:
            positions: List of positions
            
        Returns:
            Dict mapping time bucket to list of positions
        """
        groups: Dict[int, List[GNSSPosition]] = {}
        
        for pos in positions:
            # Compute bucket (milliseconds since epoch, rounded to window)
            ts_ms = int(pos.timestamp.timestamp() * 1000)
            bucket = ts_ms // self.time_window_ms * self.time_window_ms
            
            if bucket not in groups:
                groups[bucket] = []
            groups[bucket].append(pos)
        
        return groups
    
    def _compute_centroids(
        self,
        time_groups: Dict[int, List[GNSSPosition]],
        required_ids: Optional[List[str]] = None
    ) -> List[CentroidPosition]:
        """
        Compute centroid positions for each time bucket.
        
        Args:
            time_groups: Positions grouped by time
            required_ids: If specified, only use positions from these receivers
            
        Returns:
            List of CentroidPosition objects
        """
        centroids = []
        
        for bucket, positions in sorted(time_groups.items()):
            # Filter to required receivers if specified
            if required_ids:
                positions = [p for p in positions if p.receiver_id in required_ids]
            
            if len(positions) < self.min_receivers_for_centroid:
                continue
            
            # Compute mean position
            lat_mean = statistics.mean(p.lat for p in positions)
            lon_mean = statistics.mean(p.lon for p in positions)
            
            # Compute mean altitude if available
            alts = [p.alt_msl for p in positions if p.alt_msl is not None]
            alt_mean = statistics.mean(alts) if alts else None
            
            # Use timestamp from first position in bucket
            ts = datetime.fromtimestamp(bucket / 1000.0)
            
            centroid = CentroidPosition(
                timestamp=ts,
                lat=lat_mean,
                lon=lon_mean,
                alt=alt_mean,
                receiver_count=len(positions),
                receiver_ids=[p.receiver_id for p in positions]
            )
            centroids.append(centroid)
        
        return centroids
    
    def _find_nearest_centroid(
        self,
        timestamp: datetime,
        centroids: List[CentroidPosition]
    ) -> Optional[CentroidPosition]:
        """Find the centroid nearest to a given timestamp"""
        if not centroids:
            return None
        
        ts_ms = timestamp.timestamp() * 1000
        best = None
        best_diff = float('inf')
        
        for c in centroids:
            c_ms = c.timestamp.timestamp() * 1000
            diff = abs(ts_ms - c_ms)
            if diff < best_diff and diff <= self.time_window_ms:
                best = c
                best_diff = diff
        
        return best
    
    def _compute_cep(self, deviations: List[float], percentile: float = 0.50) -> float:
        """
        Compute Circular Error Probable (CEP).
        
        Args:
            deviations: List of horizontal deviations in meters
            percentile: Percentile (0.50 for CEP50, 0.95 for CEP95)
            
        Returns:
            CEP value in meters
        """
        if not deviations:
            return 0.0
        
        sorted_devs = sorted(deviations)
        idx = int(len(sorted_devs) * percentile)
        idx = min(idx, len(sorted_devs) - 1)
        return sorted_devs[idx]
    
    def _compute_2drms(self, deviations: List[float]) -> float:
        """
        Compute 2DRMS (2x Distance Root Mean Square).
        Contains approximately 95% of positions.
        
        Args:
            deviations: List of horizontal deviations in meters
            
        Returns:
            2DRMS value in meters
        """
        if not deviations:
            return 0.0
        
        # RMS = sqrt(mean of squared deviations)
        rms = math.sqrt(statistics.mean(d**2 for d in deviations))
        return 2 * rms
    
    def _compute_receiver_stats(
        self,
        session: GNSSSession,
        receiver_id: str,
        all_deviations: List[DeviationRecord],
        truth_deviations: List[Dict]
    ) -> ReceiverStatistics:
        """Compute statistics for a single receiver"""
        
        positions = session.get_positions_for_receiver(receiver_id)
        stats = ReceiverStatistics(receiver_id=receiver_id)
        
        if not positions:
            return stats
        
        stats.position_count = len(positions)
        
        # Position statistics
        lats = [p.lat for p in positions]
        lons = [p.lon for p in positions]
        alts = [p.alt_msl for p in positions if p.alt_msl is not None]
        
        stats.lat_mean = statistics.mean(lats)
        stats.lat_std = statistics.stdev(lats) if len(lats) > 1 else 0.0
        stats.lon_mean = statistics.mean(lons)
        stats.lon_std = statistics.stdev(lons) if len(lons) > 1 else 0.0
        
        if alts:
            stats.alt_mean = statistics.mean(alts)
            stats.alt_std = statistics.stdev(alts) if len(alts) > 1 else 0.0
        
        # Fix type distribution
        fix_dist: Dict[int, int] = {}
        for p in positions:
            ft = p.fix_type.value
            fix_dist[ft] = fix_dist.get(ft, 0) + 1
        stats.fix_type_distribution = fix_dist
        
        # Satellite count
        sat_counts = [p.sat_count for p in positions]
        stats.sat_count_min = min(sat_counts)
        stats.sat_count_max = max(sat_counts)
        stats.sat_count_mean = statistics.mean(sat_counts)
        
        # DOP statistics
        hdops = [p.hdop for p in positions if p.hdop is not None]
        if hdops:
            stats.hdop_min = min(hdops)
            stats.hdop_max = max(hdops)
            stats.hdop_mean = statistics.mean(hdops)
        
        vdops = [p.vdop for p in positions if p.vdop is not None]
        if vdops:
            stats.vdop_min = min(vdops)
            stats.vdop_max = max(vdops)
            stats.vdop_mean = statistics.mean(vdops)
        
        pdops = [p.pdop for p in positions if p.pdop is not None]
        if pdops:
            stats.pdop_min = min(pdops)
            stats.pdop_max = max(pdops)
            stats.pdop_mean = statistics.mean(pdops)
        
        # Time coverage
        stats.first_timestamp = min(p.timestamp for p in positions)
        stats.last_timestamp = max(p.timestamp for p in positions)
        
        # Deviation from centroid
        rcvr_devs = [d for d in all_deviations if d.receiver_id == receiver_id]
        if rcvr_devs:
            h_devs = [d.deviation_h_m for d in rcvr_devs]
            stats.deviation_from_centroid_mean = statistics.mean(h_devs)
            stats.deviation_from_centroid_max = max(h_devs)
            stats.deviation_from_centroid_std = statistics.stdev(h_devs) if len(h_devs) > 1 else 0.0
            stats.cep_50 = self._compute_cep(h_devs, 0.50)
            stats.drms_2 = self._compute_2drms(h_devs)
        
        # Deviation from truth
        rcvr_truth_devs = [d for d in truth_deviations if d['receiver_id'] == receiver_id]
        if rcvr_truth_devs:
            h_devs = [d['h'] for d in rcvr_truth_devs]
            stats.deviation_from_truth_mean = statistics.mean(h_devs)
            stats.deviation_from_truth_max = max(h_devs)
        
        return stats
    
    def get_positions_at_time(
        self,
        session: GNSSSession,
        target_time: datetime
    ) -> Dict[str, GNSSPosition]:
        """
        Get the position for each receiver at a specific time.
        Uses nearest position within time window.
        
        Args:
            session: Session data
            target_time: Target timestamp
            
        Returns:
            Dict mapping receiver_id to position
        """
        result = {}
        target_ms = target_time.timestamp() * 1000
        
        for receiver_id in session.receivers.keys():
            positions = session.get_positions_for_receiver(receiver_id)
            
            best_pos = None
            best_diff = float('inf')
            
            for pos in positions:
                pos_ms = pos.timestamp.timestamp() * 1000
                diff = abs(target_ms - pos_ms)
                if diff < best_diff and diff <= self.time_window_ms:
                    best_pos = pos
                    best_diff = diff
            
            if best_pos:
                result[receiver_id] = best_pos
        
        return result
    
    def generate_deviation_time_series(
        self,
        session: GNSSSession,
        metrics: VarianceMetrics,
        receiver_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate time series data for deviation charts.
        
        Args:
            session: Session data
            metrics: Computed metrics
            receiver_id: Filter to specific receiver (None for all)
            
        Returns:
            List of dicts with timestamp, receiver_id, deviation_h, deviation_v
        """
        records = metrics.deviation_records
        
        if receiver_id:
            records = [r for r in records if r.receiver_id == receiver_id]
        
        return [
            {
                'timestamp': r.timestamp.isoformat(),
                'receiver_id': r.receiver_id,
                'deviation_h_m': r.deviation_h_m,
                'deviation_v_m': r.deviation_v_m,
                'lat': r.lat,
                'lon': r.lon
            }
            for r in sorted(records, key=lambda x: x.timestamp)
        ]


# Utility function for quick analysis
def analyze_session(session: GNSSSession) -> VarianceMetrics:
    """
    Convenience function to analyze a session.
    
    Args:
        session: GNSSSession to analyze
        
    Returns:
        VarianceMetrics
    """
    analyzer = GNSSVarianceAnalyzer()
    return analyzer.analyze(session)
