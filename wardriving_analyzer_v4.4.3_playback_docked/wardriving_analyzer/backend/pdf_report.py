"""
PDF Report Generator for Wardriving Analyzer

Creates professional PDF reports with:
- Executive summary
- Statistics and metrics
- AP classification breakdown
- Security analysis
- Maps (if GPS data available)
- Detailed AP listings
"""

import io
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch, mm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        PageBreak, Image, HRFlowable
    )
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

# Alias for convenience
PDF_AVAILABLE = REPORTLAB_AVAILABLE

logger = logging.getLogger(__name__)


class PDFReportGenerator:
    """
    Generates comprehensive PDF reports from wardriving analysis data.
    """
    
    # Color scheme
    COLORS = {
        'primary': colors.HexColor('#1a1a2e'),
        'secondary': colors.HexColor('#16213e'),
        'accent': colors.HexColor('#0f3460'),
        'static': colors.HexColor('#10b981'),
        'mobile': colors.HexColor('#f59e0b'),
        'uncertain': colors.HexColor('#ef4444'),
        'text': colors.HexColor('#e6edf3'),
        'text_secondary': colors.HexColor('#8b949e'),
        'border': colors.HexColor('#30363d'),
        'white': colors.white,
        'black': colors.black,
    }
    
    def __init__(self, analysis_engine, report_options: dict = None):
        """
        Initialize report generator.
        
        Args:
            analysis_engine: AnalysisEngine instance with classified data
        """
        if not REPORTLAB_AVAILABLE:
            raise ImportError("reportlab is required for PDF generation. Install with: pip install reportlab")
        
        self.engine = analysis_engine
        self.report_options = report_options or {}
        self.filter_ids = None  # Will be set in generate() if filtering
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()

    @staticmethod
    def _bold_font_name(name: str) -> str:
        """Return a built-in reportlab bold font name for a given base font."""
        base = (name or 'Helvetica').strip()
        if base.endswith('-Bold'):
            return base
        mapping = {
            'Helvetica': 'Helvetica-Bold',
            'Times-Roman': 'Times-Bold',
            'Times': 'Times-Bold',
            'Courier': 'Courier-Bold',
        }
        return mapping.get(base, base + '-Bold')
    
    def _setup_custom_styles(self):
        """Create custom paragraph styles"""

        base_font = str(self.report_options.get('pdf_font_family') or 'Helvetica').strip()
        self.base_font = base_font
        self.bold_font = self._bold_font_name(base_font)
        body_size = int(self.report_options.get('pdf_body_font_size') or 10)
        title_size = int(self.report_options.get('pdf_title_font_size') or 24)
        section_size = max(12, body_size + 6)
        sub_size = max(11, body_size + 2)

        self.styles.add(ParagraphStyle(
            'ReportTitle',
            parent=self.styles['Title'],
            fontSize=title_size,
            fontName=self.bold_font,
            spaceAfter=30,
            textColor=self.COLORS['primary'],
            alignment=TA_CENTER
        ))

        self.styles.add(ParagraphStyle(
            'CompanyLine',
            parent=self.styles['Normal'],
            fontSize=int(self.report_options.get('company_font_size') or 12),
            fontName=(self.bold_font if bool(self.report_options.get('company_bold', False)) else self.base_font),
            textColor=self.COLORS['accent'],
            alignment=TA_CENTER,
            spaceAfter=12
        ))

        self.styles.add(ParagraphStyle(
            'SectionHeader',
            parent=self.styles['Heading1'],
            fontSize=section_size,
            fontName=self._bold_font_name(base_font),
            spaceBefore=20,
            spaceAfter=12,
            textColor=self.COLORS['primary'],
            borderWidth=1,
            borderColor=self.COLORS['border'],
            borderPadding=5
        ))
        
        self.styles.add(ParagraphStyle(
            'SubHeader',
            parent=self.styles['Heading2'],
            fontSize=sub_size,
            fontName=self._bold_font_name(base_font),
            spaceBefore=15,
            spaceAfter=8,
            textColor=self.COLORS['accent']
        ))
        
        self.styles.add(ParagraphStyle(
            'ReportBody',
            parent=self.styles['Normal'],
            fontName=base_font,
            fontSize=body_size,
            spaceAfter=8,
            textColor=self.COLORS['black']
        ))
        
        self.styles.add(ParagraphStyle(
            'SmallText',
            parent=self.styles['Normal'],
            fontName=base_font,
            fontSize=max(8, body_size-2),
            textColor=self.COLORS['text_secondary']
        ))
        
        self.styles.add(ParagraphStyle(
            'StatValue',
            parent=self.styles['Normal'],
            fontSize=18,
            alignment=TA_CENTER,
            textColor=self.COLORS['primary'],
            fontName=self.bold_font
        ))
        
        self.styles.add(ParagraphStyle(
            'StatLabel',
            parent=self.styles['Normal'],
            fontSize=9,
            alignment=TA_CENTER,
            textColor=self.COLORS['text_secondary']
        ))
    
    def generate(self, output_path: Optional[str] = None, 
                 title: str = "RF Site Survey Analysis Report",
                 include_details: bool = True,
                 filter_ids: Optional[List[str]] = None) -> bytes:
        """
        Generate PDF report.
        
        Args:
            output_path: Optional file path to save PDF
            title: Report title
            include_details: Whether to include detailed AP listings
            filter_ids: Optional list of AP IDs to include (if None, include all)
        
        Returns:
            PDF bytes
        """
        # Store filter_ids for use in building sections
        if filter_ids:
            self.filter_ids = set(str(x).lower() for x in filter_ids)
        else:
            self.filter_ids = None
        
        buffer = io.BytesIO()
        
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=0.75*inch,
            leftMargin=0.75*inch,
            topMargin=0.75*inch,
            bottomMargin=0.75*inch
        )
        
        story = []
        
        # Title page
        story.extend(self._build_title_page(title))
        story.append(PageBreak())
        
        # Executive summary
        story.extend(self._build_executive_summary())
        story.append(Spacer(1, 20))
        
        # Statistics
        story.extend(self._build_statistics_section())
        story.append(PageBreak())
        
        # Security analysis
        story.extend(self._build_security_section())
        story.append(Spacer(1, 20))
        
        # Classification breakdown
        story.extend(self._build_classification_section())
        
        # Detailed AP listings (optional)
        if include_details:
            story.append(PageBreak())
            story.extend(self._build_details_section())
        
        # Build PDF
        doc.build(story, onFirstPage=self._add_header_footer, 
                  onLaterPages=self._add_header_footer)
        
        pdf_bytes = buffer.getvalue()
        buffer.close()
        
        # Save to file if path provided
        if output_path:
            with open(output_path, 'wb') as f:
                f.write(pdf_bytes)
            logger.info(f"PDF report saved to: {output_path}")
        
        return pdf_bytes
    
    def _add_header_footer(self, canvas, doc):
        """Add header and footer to each page"""
        canvas.saveState()

        # Optional watermark
        try:
            rep = self.report_options or {}
            wm_enabled = bool(rep.get('watermark_enabled', False))
            wm_path = rep.get('watermark_path')
            wm_opacity = float(rep.get('watermark_opacity', 0.08) or 0.08)
            if wm_enabled and wm_path and os.path.exists(wm_path):
                # Best-effort transparency
                try:
                    if hasattr(canvas, 'setFillAlpha'):
                        canvas.setFillAlpha(wm_opacity)
                    if hasattr(canvas, 'setStrokeAlpha'):
                        canvas.setStrokeAlpha(wm_opacity)
                except Exception:
                    pass
                pw, ph = doc.pagesize
                # Scale watermark to ~60% of page width
                target_w = pw * 0.6
                # Keep aspect ratio if possible
                from reportlab.lib.utils import ImageReader
                img = ImageReader(wm_path)
                iw, ih = img.getSize()
                scale = target_w / float(iw) if iw else 1.0
                target_h = ih * scale
                x = (pw - target_w) / 2.0
                y = (ph - target_h) / 2.0
                canvas.drawImage(img, x, y, width=target_w, height=target_h, mask='auto')
        except Exception:
            pass

        
        # Footer
        canvas.setFont(str(self.report_options.get('pdf_font_family') or self.base_font), 8)
        canvas.setFillColor(self.COLORS['text_secondary'])
        canvas.drawString(
            doc.leftMargin,
            0.5*inch,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        canvas.drawRightString(
            doc.pagesize[0] - doc.rightMargin,
            0.5*inch,
            f"Page {doc.page}"
        )
        
        # Header line
        canvas.setStrokeColor(self.COLORS['border'])
        canvas.line(
            doc.leftMargin,
            doc.pagesize[1] - 0.5*inch,
            doc.pagesize[0] - doc.rightMargin,
            doc.pagesize[1] - 0.5*inch
        )
        
        canvas.restoreState()
    
    def _get_filtered_aps(self) -> Dict[str, Any]:
        """Get access points, filtered by filter_ids if set"""
        if self.filter_ids is None:
            return self.engine.access_points
        return {
            mac: ap for mac, ap in self.engine.access_points.items()
            if mac.lower() in self.filter_ids
        }
    
    def _get_filtered_classifications(self) -> Dict[str, Any]:
        """Get classifications, filtered by filter_ids if set"""
        if self.filter_ids is None:
            return self.engine.classifications
        return {
            mac: c for mac, c in self.engine.classifications.items()
            if mac.lower() in self.filter_ids
        }
    
    def _get_filtered_stats(self) -> Dict[str, Any]:
        """Get summary stats for filtered APs"""
        filtered_aps = self._get_filtered_aps()
        filtered_classes = self._get_filtered_classifications()
        
        if not filtered_aps:
            return {}
        
        # Calculate stats for filtered set
        total_detections = 0
        with_gps = 0
        static_count = 0
        mobile_count = 0
        uncertain_count = 0
        
        for mac, ap in filtered_aps.items():
            total_detections += len(ap.detections)
            if ap.locations:
                with_gps += 1
            
            c = filtered_classes.get(mac, {})
            cls = c.get('classification', 'uncertain')
            if cls == 'static':
                static_count += 1
            elif cls == 'mobile':
                mobile_count += 1
            else:
                uncertain_count += 1
        
        return {
            'total_aps': len(filtered_aps),
            'total_detections': total_detections,
            'with_gps': with_gps,
            'static': static_count,
            'mobile': mobile_count,
            'uncertain': uncertain_count
        }
    
    def _build_title_page(self, title: str) -> List:
        """Build title page elements"""
        elements = []
        
        elements.append(Spacer(1, 2*inch))
        elements.append(Paragraph(title, self.styles['ReportTitle']))
        # Optional company line
        rep = self.report_options or {}
        company = str(rep.get('company_name') or '').strip()
        if company:
            cf = str(rep.get('company_font_family') or rep.get('pdf_font_family') or self.base_font).strip()
            cs = int(rep.get('company_font_size') or max(10, int(rep.get('pdf_body_font_size') or 10) + 2))
            cb = bool(rep.get('company_bold', False))

            fname = self._bold_font_name(cf) if cb else cf
            company_style = ParagraphStyle(
                'CompanyLineRuntime',
                parent=self.styles['Normal'],
                fontName=fname,
                fontSize=cs,
                alignment=TA_CENTER,
                textColor=self.COLORS['accent']
            )
            elements.append(Spacer(1, 0.15*inch))
            elements.append(Paragraph(company, company_style))

        elements.append(Spacer(1, 0.5*inch))
        
        # Subtitle with date
        subtitle = "Multi-Spectrum RF Site Survey Analysis"
        if self.filter_ids:
            subtitle += f" ({len(self.filter_ids)} Selected SOIs)"
        
        elements.append(Paragraph(subtitle, self.styles['SubHeader']))
        elements.append(Paragraph(
            f"Report Generated: {datetime.now().strftime('%B %d, %Y at %H:%M')}",
            self.styles['SmallText']
        ))
        
        elements.append(Spacer(1, 1*inch))
        
        # Quick stats box - use filtered stats
        stats = self._get_filtered_stats()
        if stats:
            quick_stats = [
                ['Total APs', 'Detections', 'GPS Coverage'],
                [
                    str(stats.get('total_aps', 0)),
                    f"{stats.get('total_detections', 0):,}",
                    f"{stats.get('with_gps', 0)} APs"
                ]
            ]
            
            table = Table(quick_stats, colWidths=[2*inch, 2*inch, 2*inch])
            table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), self.base_font),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('TEXTCOLOR', (0, 0), (-1, 0), self.COLORS['text_secondary']),
                ('FONTNAME', (0, 1), (-1, 1), self._bold_font_name(self.base_font)),
                ('FONTSIZE', (0, 1), (-1, 1), 20),
                ('TEXTCOLOR', (0, 1), (-1, 1), self.COLORS['primary']),
                ('TOPPADDING', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
            ]))
            elements.append(table)
        
        return elements
    
    def _build_executive_summary(self) -> List:
        """Build executive summary section"""
        elements = []
        
        elements.append(Paragraph("Executive Summary", self.styles['SectionHeader']))
        
        stats = self._get_filtered_stats()
        if not stats:
            elements.append(Paragraph("No data available for analysis.", self.styles['ReportBody']))
            return elements
        
        total_aps = stats.get('total_aps', 0)
        static_count = stats.get('static', 0)
        mobile_count = stats.get('mobile', 0)
        uncertain_count = stats.get('uncertain', 0)
        with_gps = stats.get('with_gps', 0)
        
        static_pct = (static_count / total_aps * 100) if total_aps > 0 else 0
        mobile_pct = (mobile_count / total_aps * 100) if total_aps > 0 else 0
        gps_pct = (with_gps / total_aps * 100) if total_aps > 0 else 0
        
        filter_note = ""
        if self.filter_ids:
            filter_note = f"<i>(Report filtered to {len(self.filter_ids)} selected SOIs)</i><br/><br/>"
        
        summary_text = f"""
        {filter_note}This RF site survey analysis processed <b>{stats.get('total_detections', 0):,}</b> signal 
        detections, identifying <b>{total_aps}</b> unique access points across WiFi, BLE, 
        and other RF spectrums.
        <br/><br/>
        <b>Classification Results:</b><br/>
        • <font color="#10b981"><b>{static_count}</b></font> access points ({static_pct:.1f}%) classified as <b>static</b> (fixed infrastructure)<br/>
        • <font color="#f59e0b"><b>{mobile_count}</b></font> access points ({mobile_pct:.1f}%) classified as <b>mobile</b> (vehicles, portable devices)<br/>
        • <font color="#ef4444"><b>{uncertain_count}</b></font> access points classified as <b>uncertain</b> (insufficient data)<br/>
        <br/>
        <b>GPS Coverage:</b> {with_gps} APs ({gps_pct:.1f}%) have geographic location data.
        """
        
        elements.append(Paragraph(summary_text, self.styles['ReportBody']))
        
        return elements
    
    def _build_statistics_section(self) -> List:
        """Build statistics section with detailed metrics"""
        elements = []
        
        elements.append(Paragraph("Analysis Statistics", self.styles['SectionHeader']))
        
        stats = self._get_filtered_stats()
        if not stats:
            return elements
        
        # Main stats table
        stats_data = [
            ['Metric', 'Value', 'Metric', 'Value'],
            ['Total Access Points', str(stats.get('total_aps', 0)),
             'Total Detections', f"{stats.get('total_detections', 0):,}"],
            ['Static APs', str(stats.get('static', 0)),
             'Mobile APs', str(stats.get('mobile', 0))],
            ['Uncertain APs', str(stats.get('uncertain', 0)),
             'GPS-Located APs', str(stats.get('with_gps', 0))],
        ]
        
        table = Table(stats_data, colWidths=[1.8*inch, 1.3*inch, 1.8*inch, 1.3*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), self.COLORS['accent']),
            ('TEXTCOLOR', (0, 0), (-1, 0), self.COLORS['white']),
            ('FONTNAME', (0, 0), (-1, 0), self.bold_font),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 1), (-1, -1), self.base_font),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 0.5, self.COLORS['border']),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8f9fa')),
        ]))
        elements.append(table)
        
        # Device type breakdown
        elements.append(Spacer(1, 20))
        elements.append(Paragraph("Device Type Distribution", self.styles['SubHeader']))
        
        device_counts = self._get_device_type_counts()
        if device_counts:
            device_data = [['Device Type', 'Count', 'Percentage']]
            total = sum(device_counts.values())
            for dtype, count in sorted(device_counts.items(), key=lambda x: -x[1]):
                pct = (count / total * 100) if total > 0 else 0
                device_data.append([dtype.upper(), str(count), f"{pct:.1f}%"])
            
            device_table = Table(device_data, colWidths=[2.5*inch, 1.5*inch, 1.5*inch])
            device_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), self.COLORS['secondary']),
                ('TEXTCOLOR', (0, 0), (-1, 0), self.COLORS['white']),
                ('FONTNAME', (0, 0), (-1, 0), self.bold_font),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('GRID', (0, 0), (-1, -1), 0.5, self.COLORS['border']),
                ('TOPPADDING', (0, 0), (-1, -1), 6),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ]))
            elements.append(device_table)
        
        return elements
    
    def _build_security_section(self) -> List:
        """Build security analysis section"""
        elements = []
        
        elements.append(Paragraph("Security Analysis", self.styles['SectionHeader']))
        
        security_counts = self._get_security_counts()
        if not security_counts:
            elements.append(Paragraph(
                "No security information available (may be BLE-only data or security data not captured).",
                self.styles['ReportBody']
            ))
            return elements
        
        # Security breakdown table
        sec_data = [['Security Type', 'Count', 'Percentage', 'Risk Level']]
        total = sum(security_counts.values())
        
        risk_levels = {
            'OPEN': ('HIGH', self.COLORS['uncertain']),
            'WEP': ('CRITICAL', self.COLORS['uncertain']),
            'WPA1': ('MEDIUM', self.COLORS['mobile']),
            'OWE': ('LOW', self.COLORS['static']),
            'WPA2': ('LOW', self.COLORS['static']),
            'WPA3': ('MINIMAL', self.COLORS['static']),
            'WPA2/WPA3': ('LOW', self.COLORS['static']),
            'UNKNOWN': ('UNKNOWN', self.COLORS['text_secondary']),
        }
        
        for sec_type, count in sorted(security_counts.items(), key=lambda x: -x[1]):
            pct = (count / total * 100) if total > 0 else 0
            risk, _ = risk_levels.get(sec_type, ('UNKNOWN', self.COLORS['text_secondary']))
            sec_data.append([sec_type, str(count), f"{pct:.1f}%", risk])
        
        sec_table = Table(sec_data, colWidths=[2*inch, 1.2*inch, 1.3*inch, 1.5*inch])
        sec_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), self.COLORS['accent']),
            ('TEXTCOLOR', (0, 0), (-1, 0), self.COLORS['white']),
            ('FONTNAME', (0, 0), (-1, 0), self.bold_font),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 0.5, self.COLORS['border']),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        elements.append(sec_table)
        
        # Security findings
        elements.append(Spacer(1, 15))
        
        open_count = security_counts.get('OPEN', 0) + security_counts.get('WEP', 0)
        if open_count > 0:
            elements.append(Paragraph(
                f"<font color='#ef4444'><b>⚠ Security Alert:</b></font> "
                f"{open_count} access points have weak or no encryption (OPEN/WEP).",
                self.styles['ReportBody']
            ))
        
        wpa3_count = security_counts.get('WPA3', 0) + security_counts.get('WPA2/WPA3', 0)
        wpa3_pct = (wpa3_count / total * 100) if total > 0 else 0
        elements.append(Paragraph(
            f"<b>WPA3 Adoption:</b> {wpa3_count} networks ({wpa3_pct:.1f}%) support WPA3 encryption.",
            self.styles['ReportBody']
        ))
        
        return elements
    
    def _build_classification_section(self) -> List:
        """Build classification breakdown section"""
        elements = []
        
        elements.append(Paragraph("Classification Breakdown", self.styles['SectionHeader']))
        
        # Get top APs by category
        results = self.engine.get_filtered_results(include_no_gps=True)
        
        # Filter by IDs if set
        if self.filter_ids:
            results = [r for r in results if r.get('id', '').lower() in self.filter_ids or r.get('mac', '').lower() in self.filter_ids]
        
        for classification in ['static', 'mobile', 'uncertain']:
            filtered = [r for r in results if r.get('classification') == classification]
            if not filtered:
                continue
            
            color = {'static': '#10b981', 'mobile': '#f59e0b', 'uncertain': '#ef4444'}[classification]
            
            elements.append(Paragraph(
                f"<font color='{color}'><b>{classification.upper()}</b></font> "
                f"({len(filtered)} APs)",
                self.styles['SubHeader']
            ))
            
            # Top 10 by detection count
            top_aps = sorted(filtered, key=lambda x: -x.get('detections', 0))[:10]
            
            table_data = [['Name/SSID', 'MAC', 'Type', 'Detections', 'Confidence']]
            for ap in top_aps:
                name = ap.get('name') or '(Hidden)'
                if len(name) > 25:
                    name = name[:22] + '...'
                table_data.append([
                    name,
                    ap.get('mac', '')[:17],
                    ap.get('type', 'UNKNOWN'),
                    str(ap.get('detections', 0)),
                    f"{ap.get('confidence', 0)*100:.1f}%"
                ])
            
            table = Table(table_data, colWidths=[2*inch, 1.5*inch, 0.8*inch, 0.9*inch, 0.9*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor(color)),
                ('TEXTCOLOR', (0, 0), (-1, 0), self.COLORS['white']),
                ('FONTNAME', (0, 0), (-1, 0), self.bold_font),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('ALIGN', (0, 1), (0, -1), 'LEFT'),
                ('GRID', (0, 0), (-1, -1), 0.5, self.COLORS['border']),
                ('TOPPADDING', (0, 0), (-1, -1), 4),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ]))
            elements.append(table)
            elements.append(Spacer(1, 15))
        
        return elements
    
    def _build_details_section(self) -> List:
        """Build detailed AP listing section"""
        elements = []
        
        elements.append(Paragraph("Detailed Access Point Listing", self.styles['SectionHeader']))
        
        results = self.engine.get_filtered_results(include_no_gps=True)
        
        # Filter by IDs if set
        if self.filter_ids:
            results = [r for r in results if r.get('id', '').lower() in self.filter_ids or r.get('mac', '').lower() in self.filter_ids]
        
        results = sorted(results, key=lambda x: (-x.get('detections', 0), x.get('mac', '')))
        
        # Split into chunks for pagination
        chunk_size = 25
        for i in range(0, len(results), chunk_size):
            chunk = results[i:i+chunk_size]
            
            table_data = [['Name', 'MAC', 'Type', 'Class', 'RSSI', 'Det', 'GPS']]
            for ap in chunk:
                name = ap.get('name') or '(Hidden)'
                if len(name) > 20:
                    name = name[:17] + '...'
                
                has_gps = '✓' if ap.get('has_gps') else '—'
                classification = ap.get('classification', 'unk')[:3].upper()
                
                table_data.append([
                    name,
                    ap.get('mac', '')[:17],
                    ap.get('type', '?')[:4],
                    classification,
                    f"{ap.get('rssi_mean', -100):.0f}",
                    str(ap.get('detections', 0)),
                    has_gps
                ])
            
            table = Table(table_data, 
                         colWidths=[1.8*inch, 1.4*inch, 0.5*inch, 0.5*inch, 0.6*inch, 0.5*inch, 0.4*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), self.COLORS['secondary']),
                ('TEXTCOLOR', (0, 0), (-1, 0), self.COLORS['white']),
                ('FONTNAME', (0, 0), (-1, 0), self.bold_font),
                ('FONTSIZE', (0, 0), (-1, -1), 7),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('ALIGN', (0, 1), (0, -1), 'LEFT'),
                ('GRID', (0, 0), (-1, -1), 0.5, self.COLORS['border']),
                ('TOPPADDING', (0, 0), (-1, -1), 3),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8f9fa')]),
            ]))
            elements.append(table)
            
            if i + chunk_size < len(results):
                elements.append(PageBreak())
        
        return elements
    
    def _get_device_type_counts(self) -> Dict[str, int]:
        """Get count of APs by device type (filtered if filter_ids is set)"""
        counts = {}
        filtered_aps = self._get_filtered_aps()
        for mac, ap in filtered_aps.items():
            dtype = self.engine._categorize_device(ap.signal_type)
            counts[dtype] = counts.get(dtype, 0) + 1
        return counts
    
    def _get_security_counts(self) -> Dict[str, int]:
        """Get count of APs by security type (filtered if filter_ids is set)"""
        counts = {}
        filtered_aps = self._get_filtered_aps()
        for mac, ap in filtered_aps.items():
            sec = ap._primary_security() if hasattr(ap, '_primary_security') else 'UNKNOWN'
            if sec:
                counts[sec] = counts.get(sec, 0) + 1
        return counts


def generate_pdf_report(analysis_engine, output_path: Optional[str] = None,
                       title: str = "RF Site Survey Analysis Report",
                       include_details: bool = True,
                       filter_ids: Optional[List[str]] = None,
                       report_options: Optional[dict] = None) -> bytes:
    """
    Convenience function to generate PDF report.
    
    Args:
        analysis_engine: AnalysisEngine with classified data
        output_path: Optional path to save PDF
        title: Report title
        include_details: Include detailed AP listings
    
    Returns:
        PDF bytes
    """
    generator = PDFReportGenerator(analysis_engine, report_options=report_options)
    return generator.generate(output_path, title, include_details, filter_ids=filter_ids)
