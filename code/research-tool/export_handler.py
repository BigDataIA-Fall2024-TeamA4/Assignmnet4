from fpdf import FPDF
from datetime import datetime
from typing import Dict, List
import jinja2
import os
from pathlib import Path

class ResearchReportExporter:
    def __init__(self):
        self.pdf_generator = PDFGenerator()
        self.codelab_generator = CodelabGenerator()
        
        # Setup Jinja2 environment for templates
        template_dir = Path(__file__).parent.parent / 'templates'
        self.jinja_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(template_dir))
        )

class PDFGenerator:
    """Handles PDF report generation"""
    
    class ResearchPDF(FPDF):
        def header(self):
            # Logo
            # self.image('logo.png', 10, 8, 33)  # Uncomment and add your logo
            self.set_font('Arial', 'B', 20)
            self.cell(0, 10, 'Research Analysis Report', 0, 1, 'C')
            self.ln(20)

        def footer(self):
            self.set_y(-15)
            self.set_font('Arial', 'I', 8)
            self.cell(0, 10, f'Page {self.page_no()}/{{nb}}', 0, 0, 'C')

    def generate_pdf(self, research_data: Dict) -> bytes:
        pdf = self.ResearchPDF()
        pdf.alias_nb_pages()
        pdf.add_page()

        # Add metadata
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, f"Document: {research_data['document_title']}", ln=True)
        pdf.cell(0, 10, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
        pdf.ln(10)

        # Add agent findings
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, "Research Findings", ln=True)

        for agent_name, findings in research_data['agents'].items():
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 10, f"{agent_name.replace('_', ' ').title()}", ln=True)
            
            pdf.set_font('Arial', '', 12)
            for finding in findings:
                pdf.multi_cell(0, 10, f"• {finding['content']}")
            pdf.ln(5)

        # Add Q&A section
        pdf.add_page()
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, "Questions and Answers", ln=True)

        for qa in research_data['qa_interactions']:
            pdf.set_font('Arial', 'B', 12)
            pdf.multi_cell(0, 10, f"Q: {qa['question']}")
            pdf.set_font('Arial', '', 12)
            pdf.multi_cell(0, 10, f"A: {qa['answer']}")
            pdf.ln(5)

        # Add summary if available
        if research_data.get('summary'):
            pdf.add_page()
            pdf.set_font('Arial', 'B', 14)
            pdf.cell(0, 10, "Research Summary", ln=True)
            pdf.set_font('Arial', '', 12)
            pdf.multi_cell(0, 10, research_data['summary'])

        return pdf.output(dest='S')

class CodelabGenerator:
    """Handles Codelab format generation"""
    
    def __init__(self):
        self.template_dir = Path(__file__).parent.parent / 'templates'
        self.jinja_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(self.template_dir))
        )

    def generate_codelab(self, research_data: Dict) -> str:
        template = self.jinja_env.get_template('codelab_template.html')
        
        # Prepare data for template
        template_data = {
            'title': f"Research Analysis: {research_data['document_title']}",
            'date': datetime.now().strftime('%Y-%m-%d %H:%M'),
            'duration': '30 mins',
            'agent_findings': [
                {
                    'agent_name': agent_name.replace('_', ' ').title(),
                    'findings': findings
                }
                for agent_name, findings in research_data['agents'].items()
            ],
            'qa_pairs': research_data['qa_interactions'],
            'summary': research_data.get('summary', '')
        }
        
        return template.render(**template_data)

def create_export_handler() -> ResearchReportExporter:
    """Factory function to create export handler instance"""
    return ResearchReportExporter() 