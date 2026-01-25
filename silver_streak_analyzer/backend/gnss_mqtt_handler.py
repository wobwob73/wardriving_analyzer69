"""
GNSS MQTT Handler for Silver Streak Analyzer
Subscribes to GNSS recorder MQTT topics and integrates with GNSS Link

Add to SSA backend folder alongside mqtt_handler.py
"""

import json
import logging
from typing import Callable, Optional, Dict, List
from datetime import datetime, timezone
import threading

logger = logging.getLogger(__name__)

# Try to import paho-mqtt
try:
    import paho.mqtt.client as mqtt
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False
    logger.warning("paho-mqtt not installed - GNSS MQTT features disabled")

# Try to import GNSS models
try:
    from gnss_models import GNSSPosition, GNSSSession, ReceiverConfig, ProtocolType, FixType
    GNSS_MODELS_AVAILABLE = True
except ImportError:
    GNSS_MODELS_AVAILABLE = False
    logger.warning("GNSS models not available")


class GNSSMQTTHandler:
    """
    Handles MQTT connections for live GNSS data from Raspberry Pi recorder.
    
    Subscribes to:
        gnss/+/fix      - Position data (lat, lon, alt, sats, etc.)
        gnss/+/status   - Device status (has_fix, protocol, etc.)
        gnss/system/status - System-wide status
    """
    
    def __init__(self,
                 broker_host: str = "localhost",
                 broker_port: int = 1883,
                 client_id: str = "ssa-gnss-client",
                 topic_prefix: str = "gnss",
                 username: Optional[str] = None,
                 password: Optional[str] = None):
        
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.client_id = client_id
        self.topic_prefix = topic_prefix
        self.username = username
        self.password = password
        
        self.client = None
        self.connected = False
        self.lock = threading.Lock()
        
        # Callbacks
        self.on_position_callback: Optional[Callable] = None
        self.on_status_callback: Optional[Callable] = None
        self.on_system_status_callback: Optional[Callable] = None
        
        # Live data storage
        self.live_positions: List[Dict] = []
        self.device_status: Dict[str, Dict] = {}
        self.system_status: Dict = {}
        self.max_stored_positions = 10000  # Limit memory usage
        
        # Statistics
        self.positions_received = 0
        self.last_position_time: Optional[datetime] = None
    
    def set_position_callback(self, callback: Callable):
        """Set callback for position updates: callback(device_id, position_dict)"""
        self.on_position_callback = callback
    
    def set_status_callback(self, callback: Callable):
        """Set callback for status updates: callback(device_id, status_dict)"""
        self.on_status_callback = callback
    
    def set_system_status_callback(self, callback: Callable):
        """Set callback for system status: callback(status_dict)"""
        self.on_system_status_callback = callback
    
    def connect(self) -> bool:
        """Connect to MQTT broker and subscribe to GNSS topics"""
        if not MQTT_AVAILABLE:
            logger.warning("MQTT not available - paho-mqtt not installed")
            return False
        
        try:
            # Handle different paho-mqtt versions
            try:
                self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1, client_id=self.client_id)
            except (AttributeError, TypeError):
                self.client = mqtt.Client(client_id=self.client_id)
            
            self.client.on_connect = self._on_connect
            self.client.on_disconnect = self._on_disconnect
            self.client.on_message = self._on_message
            
            if self.username and self.password:
                self.client.username_pw_set(self.username, self.password)
            
            logger.info(f"GNSS MQTT: Connecting to {self.broker_host}:{self.broker_port}...")
            self.client.connect(self.broker_host, self.broker_port, keepalive=60)
            self.client.loop_start()
            
            # Wait for connection
            for _ in range(50):
                if self.connected:
                    return True
                import time
                time.sleep(0.1)
            
            logger.warning(f"GNSS MQTT: Connection timeout")
            return False
            
        except Exception as e:
            logger.error(f"GNSS MQTT connection error: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from broker"""
        if self.client:
            try:
                self.client.loop_stop()
                self.client.disconnect()
                self.connected = False
                logger.info("GNSS MQTT: Disconnected")
            except Exception as e:
                logger.error(f"GNSS MQTT disconnect error: {e}")
    
    def _on_connect(self, client, userdata, flags, rc, *args):
        """Connection callback"""
        if rc == 0:
            self.connected = True
            logger.info(f"GNSS MQTT: Connected to {self.broker_host}:{self.broker_port}")
            
            # Subscribe to GNSS topics
            topics = [
                (f"{self.topic_prefix}/+/fix", 1),
                (f"{self.topic_prefix}/+/status", 1),
                (f"{self.topic_prefix}/system/status", 1),
            ]
            self.client.subscribe(topics)
            logger.info(f"GNSS MQTT: Subscribed to {self.topic_prefix}/# topics")
        else:
            logger.error(f"GNSS MQTT: Connection failed (code {rc})")
    
    def _on_disconnect(self, client, userdata, rc, *args):
        """Disconnection callback"""
        self.connected = False
        if rc != 0:
            logger.warning(f"GNSS MQTT: Unexpected disconnect (code {rc})")
    
    def _on_message(self, client, userdata, msg):
        """Message received callback"""
        try:
            topic = msg.topic
            payload = json.loads(msg.payload.decode('utf-8'))
            
            # Parse topic: gnss/{device_id}/{type} or gnss/system/status
            parts = topic.split('/')
            if len(parts) < 3:
                return
            
            msg_type = parts[-1]
            
            if parts[1] == "system":
                # System status
                self._handle_system_status(payload)
            else:
                device_id = parts[1]
                
                if msg_type == "fix":
                    self._handle_position(device_id, payload)
                elif msg_type == "status":
                    self._handle_device_status(device_id, payload)
        
        except json.JSONDecodeError:
            logger.debug(f"GNSS MQTT: Invalid JSON from {msg.topic}")
        except Exception as e:
            logger.error(f"GNSS MQTT message error: {e}")
    
    def _handle_position(self, device_id: str, data: Dict):
        """Handle incoming position data"""
        with self.lock:
            self.positions_received += 1
            self.last_position_time = datetime.now(timezone.utc)
            
            # Add device_id to data
            data['device_id'] = device_id
            data['received_at'] = self.last_position_time.isoformat()
            
            # Store position
            self.live_positions.append(data)
            
            # Trim if too many
            if len(self.live_positions) > self.max_stored_positions:
                self.live_positions = self.live_positions[-self.max_stored_positions:]
        
        # Call callback
        if self.on_position_callback:
            try:
                self.on_position_callback(device_id, data)
            except Exception as e:
                logger.error(f"Position callback error: {e}")
    
    def _handle_device_status(self, device_id: str, data: Dict):
        """Handle device status update"""
        with self.lock:
            self.device_status[device_id] = data
        
        if self.on_status_callback:
            try:
                self.on_status_callback(device_id, data)
            except Exception as e:
                logger.error(f"Status callback error: {e}")
    
    def _handle_system_status(self, data: Dict):
        """Handle system status update"""
        with self.lock:
            self.system_status = data
        
        if self.on_system_status_callback:
            try:
                self.on_system_status_callback(data)
            except Exception as e:
                logger.error(f"System status callback error: {e}")
    
    def get_latest_positions(self, count: int = 100) -> List[Dict]:
        """Get the latest N positions"""
        with self.lock:
            return self.live_positions[-count:]
    
    def get_positions_since(self, timestamp: datetime) -> List[Dict]:
        """Get positions since a given timestamp"""
        with self.lock:
            return [
                p for p in self.live_positions
                if datetime.fromisoformat(p.get('timestamp', '1970-01-01')) > timestamp
            ]
    
    def get_device_status(self, device_id: Optional[str] = None) -> Dict:
        """Get device status (all or specific)"""
        with self.lock:
            if device_id:
                return self.device_status.get(device_id, {})
            return self.device_status.copy()
    
    def get_system_status(self) -> Dict:
        """Get system status"""
        with self.lock:
            return self.system_status.copy()
    
    def get_statistics(self) -> Dict:
        """Get connection statistics"""
        return {
            'connected': self.connected,
            'broker': f"{self.broker_host}:{self.broker_port}",
            'positions_received': self.positions_received,
            'positions_stored': len(self.live_positions),
            'devices_tracked': len(self.device_status),
            'last_position_time': self.last_position_time.isoformat() if self.last_position_time else None
        }
    
    def clear_positions(self):
        """Clear stored positions"""
        with self.lock:
            self.live_positions.clear()
            self.positions_received = 0
    
    def to_gnss_position(self, data: Dict) -> Optional['GNSSPosition']:
        """Convert MQTT data to GNSSPosition object"""
        if not GNSS_MODELS_AVAILABLE:
            return None
        
        try:
            # Parse timestamp
            ts_str = data.get('timestamp', '')
            if ts_str:
                if ts_str.endswith('Z'):
                    ts_str = ts_str[:-1] + '+00:00'
                timestamp = datetime.fromisoformat(ts_str)
            else:
                timestamp = datetime.now(timezone.utc)
            
            # Determine protocol
            protocol_str = data.get('protocol', 'UNKNOWN')
            try:
                protocol = ProtocolType(protocol_str)
            except ValueError:
                protocol = ProtocolType.UNKNOWN
            
            # Determine fix type
            fix_quality = data.get('fix_quality', data.get('fix_type', 0))
            if isinstance(fix_quality, int):
                fix_type = FixType.from_value(fix_quality)
            else:
                fix_type = FixType.FIX_3D if data.get('has_fix', False) else FixType.NO_FIX
            
            # Convert accuracy from mm to m if needed
            acc_h = data.get('accuracy_h_m') or data.get('h_acc_mm')
            if acc_h and acc_h > 100:  # Likely in mm
                acc_h = acc_h / 1000.0
            
            acc_v = data.get('accuracy_v_m') or data.get('v_acc_mm')
            if acc_v and acc_v > 100:  # Likely in mm
                acc_v = acc_v / 1000.0
            
            return GNSSPosition(
                timestamp=timestamp,
                receiver_id=data.get('device_id', 'unknown'),
                lat=float(data.get('latitude', 0)),
                lon=float(data.get('longitude', 0)),
                alt_msl=data.get('altitude') or data.get('height_msl_m'),
                alt_hae=data.get('height_m'),
                fix_type=fix_type,
                carrier_solution=data.get('carrier_soln', 0),
                sat_count=data.get('num_sats') or data.get('num_sv', 0),
                hdop=data.get('hdop'),
                vdop=data.get('vdop'),
                pdop=data.get('pdop'),
                accuracy_h_m=acc_h,
                accuracy_v_m=acc_v,
                speed_mps=data.get('speed_mps') or (data.get('g_speed', 0) / 1000.0 if data.get('g_speed') else None),
                heading_deg=data.get('heading') or data.get('course_deg'),
                protocol=protocol,
                is_valid=data.get('has_fix', True)
            )
        except Exception as e:
            logger.error(f"Error converting to GNSSPosition: {e}")
            return None


# Convenience function to create and connect handler
def create_gnss_mqtt_handler(
    host: str,
    port: int = 1883,
    username: Optional[str] = None,
    password: Optional[str] = None
) -> Optional[GNSSMQTTHandler]:
    """
    Create and connect a GNSS MQTT handler.
    
    Args:
        host: MQTT broker hostname/IP
        port: MQTT broker port
        username: Optional username
        password: Optional password
    
    Returns:
        Connected GNSSMQTTHandler or None if connection failed
    """
    handler = GNSSMQTTHandler(
        broker_host=host,
        broker_port=port,
        username=username,
        password=password
    )
    
    if handler.connect():
        return handler
    return None
