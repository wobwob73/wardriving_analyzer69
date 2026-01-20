#!/usr/bin/env python3
"""
GNSS Link - PDF Report Generator
Version: 4.6.1

Generates PDF reports with:
- Session summary
- Receiver statistics
- Variance metrics
- Map overlays with track visualization
- Charts (deviation over time, box plots, etc.)
"""

import io
import math
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass

# ReportLab imports
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.colors import HexColor, black, white, gray, Color
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, Image, KeepTogether
)
from reportlab.graphics.shapes import Drawing, Line, Rect, Circle, String
from reportlab.graphics.charts.linecharts import HorizontalLineChart
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.widgets.markers import makeMarker

# PIL for map image generation
try:
    from PIL import Image as PILImage, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

from gnss_models import (
    GNSSSession, GNSSPosition, ReceiverStatistics, ReceiverConfig,
    FixType, haversine_distance, MapMarker
)
from gnss_variance_analysis import VarianceMetrics, CentroidPosition

logger = logging.getLogger(__name__)


# Dark mode color scheme
DARK_BG = HexColor('#1a1a2e')
DARK_SURFACE = HexColor('#16213e')
DARK_SURFACE_LIGHT = HexColor('#1f2940')
DARK_BORDER = HexColor('#2d3a5a')
ACCENT_BLUE = HexColor('#0984e3')
ACCENT_CYAN = HexColor('#00cec9')
ACCENT_GREEN = HexColor('#00b894')
ACCENT_ORANGE = HexColor('#e17055')
ACCENT_PURPLE = HexColor('#6c5ce7')
TEXT_PRIMARY = HexColor('#e8e8e8')
TEXT_SECONDARY = HexColor('#a0a0b0')
TEXT_MUTED = HexColor('#6c7a8a')


@dataclass
class ReportConfig:
    """Configuration for report generation"""
    title: str = "GNSS Link Analysis Report"
    dark_mode: bool = True
    page_size: Tuple[float, float] = letter
    
    # Content toggles
    include_summary: bool = True
    include_receiver_table: bool = True
    include_variance_metrics: bool = True
    include_map_overview: bool = True
    include_track_overlay: bool = True
    include_deviation_chart: bool = True
    include_box_plot: bool = True
    include_event_log: bool = True
    
    # Metric selection
    metric_std_h: bool = True
    metric_std_v: bool = True
    metric_individual_vs_avg: bool = True
    metric_max_deviation: bool = True
    metric_cep_50: bool = True
    metric_2drms: bool = True
    
    # Map settings
    map_width_px: int = 800
    map_height_px: int = 600
    map_padding_percent: float = 0.1
    
    # Tile cache path (for offline maps)
    tile_cache_path: Optional[Path] = None


class GNSSReportGenerator:
    """
    Generates PDF reports for GNSS Link sessions.
    """
    
    def __init__(
        self,
        session: GNSSSession,
        metrics: VarianceMetrics,
        config: Optional[ReportConfig] = None
    ):
        """
        Initialize report generator.
        
        Args:
            session: GNSSSession with data
            metrics: Computed VarianceMetrics
            config: Report configuration
        """
        self.session = session
        self.metrics = metrics
        self.config = config or ReportConfig()
        
        # Set up styles
        self.styles = getSampleStyleSheet()
        self._setup_styles()
        
        # Receiver colors for charts
        self.receiver_colors = self._generate_receiver_colors()
    
    def _setup_styles(self):
        """Set up paragraph and table styles"""
        if self.config.dark_mode:
            text_color = TEXT_PRIMARY
            heading_color = ACCENT_BLUE
            subheading_color = ACCENT_CYAN
        else:
            text_color = black
            heading_color = HexColor('#2d3436')
            subheading_color = HexColor('#0984e3')
        
        self.styles.add(ParagraphStyle(
            name='ReportTitle',
            parent=self.styles['Title'],
            fontSize=24,
            textColor=text_color,
            spaceAfter=6
        ))
        
        self.styles.add(ParagraphStyle(
            name='ReportSubtitle',
            parent=self.styles['Normal'],
            fontSize=12,
            textColor=TEXT_SECONDARY if self.config.dark_mode else gray,
            alignment=TA_CENTER,
            spaceAfter=20
        ))
        
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading1'],
            fontSize=14,
            textColor=heading_color,
            spaceBefore=16,
            spaceAfter=8
        ))
        
        self.styles.add(ParagraphStyle(
            name='SubsectionHeader',
            parent=self.styles['Heading2'],
            fontSize=11,
            textColor=subheading_color,
            spaceBefore=10,
            spaceAfter=6
        ))
        
        self.styles.add(ParagraphStyle(
            name='ReportBody',
            parent=self.styles['Normal'],
            fontSize=9,
            textColor=text_color,
            spaceAfter=6
        ))
        
        self.styles.add(ParagraphStyle(
            name='SmallText',
            parent=self.styles['Normal'],
            fontSize=8,
            textColor=TEXT_SECONDARY if self.config.dark_mode else gray
        ))
    
    def _generate_receiver_colors(self) -> Dict[str, str]:
        """Generate distinct colors for each receiver"""
        colors = [
            '#00FF00', '#FF0000', '#0088FF', '#FFAA00', '#FF00FF',
            '#00FFFF', '#FF8800', '#8800FF', '#88FF00', '#FF0088',
            '#0088FF', '#AAFF00', '#FF00AA', '#00FFAA', '#AA00FF',
            '#FFFF00'
        ]
        
        result = {}
        for i, receiver_id in enumerate(self.session.receivers.keys()):
            result[receiver_id] = colors[i % len(colors)]
            # Override with configured color if available
            config = self.session.receivers.get(receiver_id)
            if config and config.icon_color:
                result[receiver_id] = config.icon_color
        
        return result
    
    def generate(self, output_path: Optional[Path] = None) -> bytes:
        """
        Generate the PDF report.
        
        Args:
            output_path: Optional path to save PDF file
            
        Returns:
            PDF bytes
        """
        buffer = io.BytesIO()
        
        doc = SimpleDocTemplate(
            buffer,
            pagesize=self.config.page_size,
            rightMargin=0.5*inch,
            leftMargin=0.5*inch,
            topMargin=0.5*inch,
            bottomMargin=0.5*inch
        )
        
        story = []
        
        # Title page
        story.extend(self._build_title_section())
        
        # Summary section
        if self.config.include_summary:
            story.extend(self._build_summary_section())
        
        # Receiver table
        if self.config.include_receiver_table:
            story.extend(self._build_receiver_table())
        
        # Variance metrics
        if self.config.include_variance_metrics:
            story.extend(self._build_variance_section())
        
        # Map overview with tracks
        if self.config.include_map_overview or self.config.include_track_overlay:
            story.append(PageBreak())
            story.extend(self._build_map_section())
        
        # Charts
        if self.config.include_deviation_chart:
            story.extend(self._build_deviation_chart())
        
        # Event log
        if self.config.include_event_log and self.session.events:
            story.extend(self._build_event_log())
        
        # Build PDF
        doc.build(story)
        
        pdf_bytes = buffer.getvalue()
        buffer.close()
        
        # Save to file if path provided
        if output_path:
            with open(output_path, 'wb') as f:
                f.write(pdf_bytes)
            logger.info(f"Report saved to {output_path}")
        
        return pdf_bytes
    
    def _build_title_section(self) -> List:
        """Build title page elements"""
        story = []
        
        story.append(Spacer(1, 0.5*inch))
        story.append(Paragraph(self.config.title, self.styles['ReportTitle']))
        
        # Subtitle with session info
        if self.session.start_time:
            date_str = self.session.start_time.strftime('%Y-%m-%d %H:%M')
            story.append(Paragraph(f"Session: {date_str}", self.styles['ReportSubtitle']))
        
        story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
                               self.styles['SmallText']))
        story.append(Spacer(1, 0.3*inch))
        
        return story
    
    def _build_summary_section(self) -> List:
        """Build session summary section"""
        story = []
        
        story.append(Paragraph("Session Summary", self.styles['SectionHeader']))
        
        # Summary data
        duration = "N/A"
        if self.session.start_time and self.session.end_time:
            delta = self.session.end_time - self.session.start_time
            hours, remainder = divmod(int(delta.total_seconds()), 3600)
            minutes, seconds = divmod(remainder, 60)
            duration = f"{hours}h {minutes}m {seconds}s"
        
        summary_data = [
            ['Parameter', 'Value'],
            ['User', self.session.username],
            ['Role', self.session.role],
            ['Start Time', self.session.start_time.strftime('%Y-%m-%d %H:%M:%S') if self.session.start_time else 'N/A'],
            ['End Time', self.session.end_time.strftime('%Y-%m-%d %H:%M:%S') if self.session.end_time else 'N/A'],
            ['Duration', duration],
            ['Total Receivers', str(len(self.session.receivers))],
            ['Truth Receivers', str(len(self.session.get_truth_receivers()))],
            ['Total Positions', str(len(self.session.positions))],
            ['Total Events', str(len(self.session.events))],
        ]
        
        if self.session.radio_number:
            summary_data.append(['Radio Number', self.session.radio_number])
        if self.session.battery_pair_id:
            summary_data.append(['Battery Pair', self.session.battery_pair_id])
        
        table = self._create_table(summary_data, [1.8*inch, 4*inch])
        story.append(table)
        story.append(Spacer(1, 0.2*inch))
        
        return story
    
    def _build_receiver_table(self) -> List:
        """Build receiver statistics table"""
        story = []
        
        story.append(Paragraph("Receiver Statistics", self.styles['SectionHeader']))
        
        # Header row
        header = ['Receiver', 'Positions', 'Sats', 'HDOP', 'Dev (m)', 'CEP50', '2DRMS']
        data = [header]
        
        for receiver_id, stats in self.metrics.receiver_stats.items():
            config = self.session.receivers.get(receiver_id)
            name = config.nickname if config and config.nickname else receiver_id
            
            # Mark truth receivers
            if config and config.is_truth_receiver:
                name = f"★ {name}"
            
            row = [
                name,
                str(stats.position_count),
                f"{stats.sat_count_mean:.1f}",
                f"{stats.hdop_mean:.2f}" if stats.hdop_mean else "N/A",
                f"{stats.deviation_from_centroid_mean:.2f}" if stats.deviation_from_centroid_mean else "N/A",
                f"{stats.cep_50:.2f}" if stats.cep_50 else "N/A",
                f"{stats.drms_2:.2f}" if stats.drms_2 else "N/A"
            ]
            data.append(row)
        
        table = self._create_table(data, [1.5*inch, 0.8*inch, 0.6*inch, 0.7*inch, 0.8*inch, 0.7*inch, 0.7*inch])
        story.append(table)
        story.append(Spacer(1, 0.2*inch))
        
        return story
    
    def _build_variance_section(self) -> List:
        """Build variance metrics section"""
        story = []
        
        story.append(Paragraph("Variance Analysis", self.styles['SectionHeader']))
        
        # All receivers metrics
        story.append(Paragraph("All Receivers", self.styles['SubsectionHeader']))
        
        all_data = [['Metric', 'Value', 'Description']]
        
        if self.config.metric_std_h:
            all_data.append([
                'Std Dev (H)',
                f"{self.metrics.all_receivers_std_h:.3f} m",
                'Horizontal position spread'
            ])
        
        if self.config.metric_std_v:
            all_data.append([
                'Std Dev (V)',
                f"{self.metrics.all_receivers_std_v:.3f} m",
                'Vertical position spread'
            ])
        
        if self.config.metric_max_deviation:
            all_data.append([
                'Max Deviation',
                f"{self.metrics.all_receivers_max_deviation:.3f} m",
                'Maximum deviation from centroid'
            ])
        
        if self.config.metric_cep_50:
            all_data.append([
                'CEP (50%)',
                f"{self.metrics.all_receivers_cep_50:.3f} m",
                'Radius containing 50% of fixes'
            ])
        
        if self.config.metric_2drms:
            all_data.append([
                '2DRMS (95%)',
                f"{self.metrics.all_receivers_2drms:.3f} m",
                'Radius containing 95% of fixes'
            ])
        
        table = self._create_table(all_data, [1.2*inch, 1.2*inch, 3.5*inch])
        story.append(table)
        story.append(Spacer(1, 0.15*inch))
        
        # Truth comparison metrics (if available)
        if self.metrics.truth_comparison_std_h is not None:
            story.append(Paragraph("Non-Truth vs Truth Receivers", self.styles['SubsectionHeader']))
            story.append(Paragraph(
                f"Truth receivers: {', '.join(self.metrics.truth_receiver_ids)}",
                self.styles['SmallText']
            ))
            
            truth_data = [['Metric', 'Value', 'Description']]
            
            if self.config.metric_std_h:
                truth_data.append([
                    'Std Dev (H)',
                    f"{self.metrics.truth_comparison_std_h:.3f} m",
                    'Horizontal deviation from truth'
                ])
            
            if self.config.metric_max_deviation:
                truth_data.append([
                    'Max Deviation',
                    f"{self.metrics.truth_comparison_max_deviation:.3f} m",
                    'Maximum deviation from truth'
                ])
            
            if self.config.metric_cep_50:
                truth_data.append([
                    'CEP (50%)',
                    f"{self.metrics.truth_comparison_cep_50:.3f} m",
                    'Radius containing 50% (vs truth)'
                ])
            
            if self.config.metric_2drms:
                truth_data.append([
                    '2DRMS (95%)',
                    f"{self.metrics.truth_comparison_2drms:.3f} m",
                    'Radius containing 95% (vs truth)'
                ])
            
            table = self._create_table(truth_data, [1.2*inch, 1.2*inch, 3.5*inch])
            story.append(table)
        
        story.append(Spacer(1, 0.2*inch))
        return story
    
    def _build_map_section(self) -> List:
        """Build map overview with receiver tracks"""
        story = []
        
        story.append(Paragraph("Track Overview", self.styles['SectionHeader']))
        
        if not HAS_PIL:
            story.append(Paragraph(
                "Map generation requires PIL/Pillow library",
                self.styles['ReportBody']
            ))
            return story
        
        if not self.session.positions:
            story.append(Paragraph("No position data available", self.styles['ReportBody']))
            return story
        
        # Generate map image
        try:
            map_image = self._generate_track_map()
            if map_image:
                # Convert PIL image to ReportLab image
                img_buffer = io.BytesIO()
                map_image.save(img_buffer, format='PNG')
                img_buffer.seek(0)
                
                # Scale to fit page
                img_width = 7 * inch
                img_height = img_width * (self.config.map_height_px / self.config.map_width_px)
                
                rl_image = Image(img_buffer, width=img_width, height=img_height)
                story.append(rl_image)
                
                # Legend
                story.append(Spacer(1, 0.1*inch))
                story.extend(self._build_map_legend())
        except Exception as e:
            logger.error(f"Failed to generate map: {e}")
            story.append(Paragraph(f"Map generation failed: {str(e)}", self.styles['ReportBody']))
        
        story.append(Spacer(1, 0.2*inch))
        return story
    
    def _generate_track_map(self) -> Optional['PILImage.Image']:
        """
        Generate a track map image with all receiver paths.
        
        Returns:
            PIL Image or None if generation fails
        """
        if not HAS_PIL:
            return None
        
        width = self.config.map_width_px
        height = self.config.map_height_px
        padding = self.config.map_padding_percent
        
        # Calculate bounds
        lats = [p.lat for p in self.session.positions]
        lons = [p.lon for p in self.session.positions]
        
        lat_min, lat_max = min(lats), max(lats)
        lon_min, lon_max = min(lons), max(lons)
        
        # Add padding
        lat_range = lat_max - lat_min
        lon_range = lon_max - lon_min
        
        # Ensure minimum range to avoid division by zero
        if lat_range < 0.0001:
            lat_range = 0.0001
        if lon_range < 0.0001:
            lon_range = 0.0001
        
        lat_min -= lat_range * padding
        lat_max += lat_range * padding
        lon_min -= lon_range * padding
        lon_max += lon_range * padding
        
        lat_range = lat_max - lat_min
        lon_range = lon_max - lon_min
        
        # Create image
        if self.config.dark_mode:
            bg_color = (26, 26, 46)  # DARK_BG
            grid_color = (45, 58, 90)  # DARK_BORDER
            text_color = (232, 232, 232)  # TEXT_PRIMARY
        else:
            bg_color = (255, 255, 255)
            grid_color = (200, 200, 200)
            text_color = (0, 0, 0)
        
        img = PILImage.new('RGB', (width, height), bg_color)
        draw = ImageDraw.Draw(img)
        
        # Draw grid
        self._draw_grid(draw, width, height, lat_min, lat_max, lon_min, lon_max,
                       grid_color, text_color)
        
        # Draw tracks for each receiver
        for receiver_id in self.session.receivers.keys():
            positions = self.session.get_positions_for_receiver(receiver_id)
            if not positions:
                continue
            
            # Sort by timestamp
            positions = sorted(positions, key=lambda p: p.timestamp)
            
            # Get color
            color_hex = self.receiver_colors.get(receiver_id, '#FFFFFF')
            color_rgb = self._hex_to_rgb(color_hex)
            
            # Draw trail
            points = []
            for pos in positions:
                x = int((pos.lon - lon_min) / lon_range * (width - 40) + 20)
                y = int((lat_max - pos.lat) / lat_range * (height - 40) + 20)
                points.append((x, y))
            
            # Draw lines
            if len(points) > 1:
                for i in range(len(points) - 1):
                    draw.line([points[i], points[i+1]], fill=color_rgb, width=2)
            
            # Draw start and end markers
            if points:
                # Start - circle
                draw.ellipse([points[0][0]-4, points[0][1]-4, 
                             points[0][0]+4, points[0][1]+4], 
                            fill=color_rgb, outline=text_color)
                # End - square
                draw.rectangle([points[-1][0]-4, points[-1][1]-4,
                               points[-1][0]+4, points[-1][1]+4],
                              fill=color_rgb, outline=text_color)
        
        # Draw centroid path
        if self.metrics.centroid_positions:
            centroid_color = (255, 255, 255) if self.config.dark_mode else (0, 0, 0)
            centroid_points = []
            for c in sorted(self.metrics.centroid_positions, key=lambda x: x.timestamp):
                x = int((c.lon - lon_min) / lon_range * (width - 40) + 20)
                y = int((lat_max - c.lat) / lat_range * (height - 40) + 20)
                centroid_points.append((x, y))
            
            if len(centroid_points) > 1:
                for i in range(len(centroid_points) - 1):
                    draw.line([centroid_points[i], centroid_points[i+1]], 
                             fill=centroid_color, width=3)
        
        # Draw markers
        for marker in self.session.markers:
            x = int((marker.lon - lon_min) / lon_range * (width - 40) + 20)
            y = int((lat_max - marker.lat) / lat_range * (height - 40) + 20)
            
            # Draw marker icon
            draw.polygon([(x, y-10), (x-7, y+5), (x+7, y+5)], 
                        fill=(255, 165, 0), outline=text_color)
            
            # Draw radius circles if enabled
            for radius_m, enabled in [
                (100, marker.radius_100m),
                (500, marker.radius_500m),
                (1000, marker.radius_1km),
                (1500, marker.radius_1_5km),
                (2000, marker.radius_2km)
            ]:
                if enabled:
                    # Convert meters to pixels (approximate)
                    meters_per_deg = 111320 * math.cos(math.radians(marker.lat))
                    radius_deg = radius_m / meters_per_deg
                    radius_px = int(radius_deg / lon_range * (width - 40))
                    
                    draw.ellipse([x - radius_px, y - radius_px,
                                 x + radius_px, y + radius_px],
                                outline=(255, 165, 0, 100), width=1)
        
        # Draw scale bar
        self._draw_scale_bar(draw, width, height, lat_min, lon_min, lon_range, text_color)
        
        return img
    
    def _draw_grid(
        self, 
        draw: 'ImageDraw.Draw',
        width: int, 
        height: int,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
        grid_color: Tuple[int, int, int],
        text_color: Tuple[int, int, int]
    ):
        """Draw coordinate grid on map"""
        lat_range = lat_max - lat_min
        lon_range = lon_max - lon_min
        
        # Determine grid spacing
        lat_step = self._nice_grid_step(lat_range)
        lon_step = self._nice_grid_step(lon_range)
        
        # Draw vertical lines (longitude)
        lon = math.ceil(lon_min / lon_step) * lon_step
        while lon < lon_max:
            x = int((lon - lon_min) / lon_range * (width - 40) + 20)
            draw.line([(x, 20), (x, height - 20)], fill=grid_color, width=1)
            
            # Label
            label = f"{lon:.4f}°"
            draw.text((x, height - 15), label, fill=text_color, anchor='mt')
            lon += lon_step
        
        # Draw horizontal lines (latitude)
        lat = math.ceil(lat_min / lat_step) * lat_step
        while lat < lat_max:
            y = int((lat_max - lat) / lat_range * (height - 40) + 20)
            draw.line([(20, y), (width - 20, y)], fill=grid_color, width=1)
            
            # Label
            label = f"{lat:.4f}°"
            draw.text((5, y), label, fill=text_color, anchor='lm')
            lat += lat_step
    
    def _nice_grid_step(self, range_val: float) -> float:
        """Calculate a nice grid step size"""
        # Target about 5 grid lines
        raw_step = range_val / 5
        
        # Round to a nice value
        magnitude = 10 ** math.floor(math.log10(raw_step))
        normalized = raw_step / magnitude
        
        if normalized < 1.5:
            nice = 1
        elif normalized < 3:
            nice = 2
        elif normalized < 7:
            nice = 5
        else:
            nice = 10
        
        return nice * magnitude
    
    def _draw_scale_bar(
        self,
        draw: 'ImageDraw.Draw',
        width: int,
        height: int,
        lat_min: float,
        lon_min: float,
        lon_range: float,
        text_color: Tuple[int, int, int]
    ):
        """Draw scale bar on map"""
        # Calculate meters per pixel
        meters_per_deg = 111320 * math.cos(math.radians(lat_min))
        meters_per_px = (lon_range * meters_per_deg) / (width - 40)
        
        # Target scale bar length
        target_px = 100
        target_m = target_px * meters_per_px
        
        # Round to nice value
        nice_m = self._nice_scale_value(target_m)
        bar_px = int(nice_m / meters_per_px)
        
        # Draw bar
        x1 = width - 20 - bar_px
        y1 = height - 30
        x2 = width - 20
        y2 = height - 30
        
        draw.line([(x1, y1), (x2, y2)], fill=text_color, width=2)
        draw.line([(x1, y1-5), (x1, y1+5)], fill=text_color, width=2)
        draw.line([(x2, y2-5), (x2, y2+5)], fill=text_color, width=2)
        
        # Label
        if nice_m >= 1000:
            label = f"{nice_m/1000:.1f} km"
        else:
            label = f"{nice_m:.0f} m"
        draw.text(((x1+x2)//2, y1-10), label, fill=text_color, anchor='mb')
    
    def _nice_scale_value(self, value: float) -> float:
        """Round to a nice scale bar value"""
        magnitude = 10 ** math.floor(math.log10(value))
        normalized = value / magnitude
        
        if normalized < 2:
            return magnitude
        elif normalized < 5:
            return 2 * magnitude
        else:
            return 5 * magnitude
    
    def _hex_to_rgb(self, hex_color: str) -> Tuple[int, int, int]:
        """Convert hex color to RGB tuple"""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    def _build_map_legend(self) -> List:
        """Build legend for the map"""
        story = []
        
        legend_data = [['Symbol', 'Receiver', 'Positions', 'Truth']]
        
        for receiver_id in self.session.receivers.keys():
            config = self.session.receivers.get(receiver_id)
            stats = self.metrics.receiver_stats.get(receiver_id)
            
            name = config.nickname if config and config.nickname else receiver_id
            color = self.receiver_colors.get(receiver_id, '#FFFFFF')
            
            is_truth = "★" if config and config.is_truth_receiver else ""
            pos_count = stats.position_count if stats else 0
            
            # Color swatch as text (would be better as actual colored box)
            legend_data.append([
                f"[{color}]",  # Could be replaced with actual color swatch
                name,
                str(pos_count),
                is_truth
            ])
        
        table = self._create_table(legend_data, [0.8*inch, 2*inch, 0.8*inch, 0.6*inch])
        story.append(table)
        
        return story
    
    def _build_deviation_chart(self) -> List:
        """Build deviation over time chart"""
        story = []
        
        story.append(Paragraph("Deviation Over Time", self.styles['SectionHeader']))
        
        if not self.metrics.deviation_records:
            story.append(Paragraph("No deviation data available", self.styles['ReportBody']))
            return story
        
        # Create simple text-based summary since ReportLab charts are complex
        # In a full implementation, this would use ReportLab's charting
        
        story.append(Paragraph(
            "Deviation time series data available for export. "
            "Use the analyzer's interactive charts for detailed visualization.",
            self.styles['ReportBody']
        ))
        
        # Show summary stats per receiver
        summary_data = [['Receiver', 'Mean Dev', 'Max Dev', 'Std Dev']]
        
        for receiver_id, stats in self.metrics.receiver_stats.items():
            config = self.session.receivers.get(receiver_id)
            name = config.nickname if config and config.nickname else receiver_id
            
            summary_data.append([
                name,
                f"{stats.deviation_from_centroid_mean:.2f} m" if stats.deviation_from_centroid_mean else "N/A",
                f"{stats.deviation_from_centroid_max:.2f} m" if stats.deviation_from_centroid_max else "N/A",
                f"{stats.deviation_from_centroid_std:.2f} m" if stats.deviation_from_centroid_std else "N/A"
            ])
        
        table = self._create_table(summary_data, [2*inch, 1.2*inch, 1.2*inch, 1.2*inch])
        story.append(table)
        
        story.append(Spacer(1, 0.2*inch))
        return story
    
    def _build_event_log(self) -> List:
        """Build session event log"""
        story = []
        
        story.append(Paragraph("Event Log", self.styles['SectionHeader']))
        
        event_data = [['Time', 'Event', 'Flight', 'Run', 'Notes']]
        
        for event in sorted(self.session.events, key=lambda e: e.timestamp):
            event_data.append([
                event.timestamp.strftime('%H:%M:%S'),
                event.event_type.replace('_', ' ').title(),
                str(event.flight_number) if event.flight_number else '',
                str(event.run_number) if event.run_number else '',
                (event.notes or '')[:30]  # Truncate long notes
            ])
        
        table = self._create_table(event_data, [1*inch, 1.5*inch, 0.6*inch, 0.6*inch, 2*inch])
        story.append(table)
        
        return story
    
    def _create_table(
        self,
        data: List[List[str]],
        col_widths: List[float]
    ) -> Table:
        """Create a styled table"""
        table = Table(data, colWidths=col_widths)
        
        if self.config.dark_mode:
            header_bg = ACCENT_BLUE
            row_bg1 = DARK_SURFACE
            row_bg2 = DARK_SURFACE_LIGHT
            border_color = DARK_BORDER
            header_text = white
            body_text = TEXT_PRIMARY
        else:
            header_bg = HexColor('#0984e3')
            row_bg1 = HexColor('#ffffff')
            row_bg2 = HexColor('#f5f5f5')
            border_color = HexColor('#cccccc')
            header_text = white
            body_text = black
        
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), header_bg),
            ('TEXTCOLOR', (0, 0), (-1, 0), header_text),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('TEXTCOLOR', (0, 1), (-1, -1), body_text),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('GRID', (0, 0), (-1, -1), 0.5, border_color),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [row_bg1, row_bg2]),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
            ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ]))
        
        return table


def generate_gnss_report(
    session: GNSSSession,
    metrics: VarianceMetrics,
    output_path: Optional[Path] = None,
    config: Optional[ReportConfig] = None
) -> bytes:
    """
    Convenience function to generate a GNSS Link report.
    
    Args:
        session: GNSSSession with data
        metrics: Computed variance metrics
        output_path: Optional path to save PDF
        config: Report configuration
        
    Returns:
        PDF bytes
    """
    generator = GNSSReportGenerator(session, metrics, config)
    return generator.generate(output_path)
