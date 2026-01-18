"""
MQTT Handler for live wardriving data ingestion
Handles real-time data from wardriving rigs via MQTT broker
"""

import json
import logging
from typing import Callable, Optional, Dict
from datetime import datetime
import threading

logger = logging.getLogger(__name__)

# Try to import paho-mqtt, but don't fail if not available
try:
    import paho.mqtt.client as mqtt
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False
    logger.warning("paho-mqtt not installed - MQTT features disabled")


class MQTTHandler:
    """Handles MQTT connections and live data ingestion"""
    
    def __init__(self, 
                 broker_host: str = "localhost",
                 broker_port: int = 1883,
                 client_id: str = "wardriving-analyzer",
                 username: Optional[str] = None,
                 password: Optional[str] = None):
        
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.client_id = client_id
        self.username = username
        self.password = password
        
        self.client = None
        self.connected = False
        self.data_callback: Optional[Callable] = None
        self.topic_subscriptions = []
        
    def set_data_callback(self, callback: Callable):
        """Set callback function for received data"""
        self.data_callback = callback
    
    def on_connect(self, client, userdata, flags, rc):
        """MQTT connection callback"""
        if rc == 0:
            self.connected = True
            logger.info(f"Connected to MQTT broker at {self.broker_host}:{self.broker_port}")
            
            for topic in self.topic_subscriptions:
                client.subscribe(topic)
                logger.info(f"Subscribed to topic: {topic}")
        else:
            logger.error(f"Failed to connect to MQTT broker. Return code: {rc}")
            self.connected = False
    
    def on_disconnect(self, client, userdata, rc):
        """MQTT disconnection callback"""
        self.connected = False
        if rc != 0:
            logger.warning(f"Unexpected disconnection. Return code: {rc}")
    
    def on_message(self, client, userdata, msg):
        """MQTT message received callback"""
        try:
            payload = msg.payload.decode('utf-8')
            
            try:
                data = json.loads(payload)
            except json.JSONDecodeError:
                logger.warning(f"Could not parse JSON from {msg.topic}: {payload}")
                return
            
            if self.data_callback:
                self.data_callback(msg.topic, data)
            
        except Exception as e:
            logger.error(f"Error processing MQTT message: {e}")
    
    def connect(self) -> bool:
        """Connect to MQTT broker"""
        if not MQTT_AVAILABLE:
            logger.warning("MQTT not available - paho-mqtt not installed")
            return False
            
        try:
            self.client = mqtt.Client(client_id=self.client_id)
            
            self.client.on_connect = self.on_connect
            self.client.on_disconnect = self.on_disconnect
            self.client.on_message = self.on_message
            
            if self.username and self.password:
                self.client.username_pw_set(self.username, self.password)
            
            self.client.connect(self.broker_host, self.broker_port, keepalive=60)
            self.client.loop_start()
            
            logger.info("MQTT connection initiated")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to connect to MQTT broker: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from MQTT broker"""
        if self.client:
            self.client.loop_stop()
            self.client.disconnect()
            self.connected = False
            logger.info("Disconnected from MQTT broker")
    
    def subscribe(self, topic: str):
        """Subscribe to a topic"""
        self.topic_subscriptions.append(topic)
        if self.client and self.connected:
            self.client.subscribe(topic)
            logger.info(f"Subscribed to topic: {topic}")
    
    def publish(self, topic: str, payload: Dict, qos: int = 1) -> bool:
        """Publish a message"""
        if self.client and self.connected:
            try:
                message = json.dumps(payload)
                self.client.publish(topic, message, qos=qos)
                logger.debug(f"Published to {topic}")
                return True
            except Exception as e:
                logger.error(f"Failed to publish to {topic}: {e}")
                return False
        else:
            logger.warning("MQTT client not connected")
            return False


class WardrivingDataProcessor:
    """Processes wardriving data from MQTT"""
    
    def __init__(self, analysis_engine):
        self.analysis_engine = analysis_engine
        self.current_run_data = {}
        self.run_counter = 0
    
    def process_detection(self, topic: str, data: Dict):
        """Process a single AP detection from MQTT"""
        try:
            mac = data.get('identifier', '').lower()
            name = data.get('name', '')
            signal_type = data.get('signal_type', 'WIFI')
            lat = float(data.get('lat', 0.0))
            lon = float(data.get('lon', 0.0))
            rssi = int(data.get('rssi_dbm', -100))
            worker = data.get('worker', 'unknown')
            channel = data.get('channel', '')
            # WiFi security/encryption (optional; best-effort).
            security = ''
            for k in ['security', 'encryption', 'auth', 'privacy', 'capabilities', 'wifi_security', 'sec']:
                if k in data and data.get(k) not in (None, '', 'nan'):
                    security = str(data.get(k))
                    break
            timestamp = data.get('iso8601_utc', datetime.now().isoformat())
            fix_ok = int(data.get('fix_ok', 0))
            
            if signal_type.upper() in ['IMU', 'ACCEL']:
                return
            
            has_gps = (fix_ok == 1 and lat != 0.0 and lon != 0.0)
            
            # Import here to avoid circular dependency
            from analysis_engine import AccessPoint
            
            if mac not in self.analysis_engine.access_points:
                self.analysis_engine.access_points[mac] = AccessPoint(mac, name, signal_type)
            
            self.analysis_engine.access_points[mac].add_detection(
                lat, lon, rssi, worker, channel, timestamp, 
                run_id=f"live_{self.run_counter}",
                has_gps=has_gps,
                security=security
            )
            
            logger.debug(f"Processed detection: {name} ({mac}) Worker {worker} RSSI: {rssi}")
            
        except Exception as e:
            logger.error(f"Error processing detection: {e}")
    
    def process_batch(self, topic: str, data: Dict):
        """Process a batch of detections"""
        if 'detections' in data:
            for detection in data['detections']:
                self.process_detection(topic, detection)
    
    def mark_run_complete(self):
        """Mark current run as complete"""
        self.run_counter += 1
        logger.info(f"Wardriving run #{self.run_counter} complete")


class LiveDataServer:
    """Server for broadcasting live analysis updates via MQTT"""
    
    def __init__(self, mqtt_handler: MQTTHandler, analysis_engine):
        self.mqtt_handler = mqtt_handler
        self.analysis_engine = analysis_engine
        self.update_interval = 5
        self.running = False
        self.topic_prefix = "wardriving/analysis"
    
    def start_broadcasting(self, topic_prefix: str = "wardriving/analysis"):
        """Start broadcasting analysis updates"""
        self.running = True
        self.topic_prefix = topic_prefix
        logger.info(f"Starting live data broadcast to {topic_prefix}")
        
        thread = threading.Thread(target=self._broadcast_loop, daemon=True)
        thread.start()
    
    def stop_broadcasting(self):
        """Stop broadcasting"""
        self.running = False
        logger.info("Stopped live data broadcast")
    
    def _broadcast_loop(self):
        """Main broadcast loop"""
        import time
        
        while self.running:
            try:
                stats = self.analysis_engine.get_summary_stats()
                
                if stats:
                    self.mqtt_handler.publish(
                        f"{self.topic_prefix}/summary",
                        {
                            'timestamp': datetime.now().isoformat(),
                            **stats
                        }
                    )
                
                classifications = self.analysis_engine.get_filtered_results()
                self.mqtt_handler.publish(
                    f"{self.topic_prefix}/classifications",
                    {
                        'timestamp': datetime.now().isoformat(),
                        'aps': classifications[:100]
                    }
                )
                
                time.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"Error in broadcast loop: {e}")
                time.sleep(1)
