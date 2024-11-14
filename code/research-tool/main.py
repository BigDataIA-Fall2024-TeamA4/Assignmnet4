from fastapi import FastAPI, Depends, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from typing import List, Dict, Optional
import jwt
from datetime import datetime
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import tempfile
import os
import subprocess
from pathlib import Path
import shutil
from md2pdf.core import md2pdf

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SECRET_KEY = "test_secret_key"
JWT_ALGORITHM = "HS256"

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# More realistic mock data
MOCK_PDFS = [
    "AI_Ethics_2024.pdf",
    "Machine_Learning_Fundamentals.pdf",
    "Neural_Networks_Deep_Learning.pdf",
    "Data_Science_Research.pdf"
]

MOCK_FINDINGS = {
    "arxiv_agent": [
        {
            "content": "Recent research shows that transformer models have achieved state-of-the-art results in natural language processing tasks.",
            "source": "arxiv.org/abs/2401.12345"
        },
        {
            "content": "A 2023 study demonstrated that smaller, efficiently trained models can match the performance of larger models.",
            "source": "arxiv.org/abs/2312.67890"
        }
    ],
    "web_search_agent": [
        {
            "content": "According to industry reports, the adoption of AI in healthcare is expected to grow by 48% annually through 2027.",
            "source": "techreports.com/ai-healthcare-2024"
        },
        {
            "content": "Recent developments in quantum computing suggest potential breakthroughs in optimization problems.",
            "source": "quantumtech.org/trends"
        }
    ],
    "rag_agent": [
        {
            "content": "The document emphasizes the importance of ethical considerations in AI development.",
            "source": "document_section_3.2"
        },
        {
            "content": "Key findings indicate that hybrid approaches combining rule-based and neural systems often provide more robust solutions.",
            "source": "document_section_4.1"
        }
    ]
}

@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    token = jwt.encode({"sub": form_data.username}, SECRET_KEY, algorithm=JWT_ALGORITHM)
    return {"access_token": token, "token_type": "bearer"}

@app.get("/pdfs")
async def get_pdfs(_: str = Depends(oauth2_scheme)):
    return MOCK_PDFS

@app.post("/research/question")
async def research_question(question: dict, _: str = Depends(oauth2_scheme)):
    # Generate a more contextual mock answer based on the question
    question_text = question['question'].lower()
    
    if 'ethics' in question_text:
        answer = "Based on current research and industry practices, AI ethics frameworks emphasize transparency, fairness, and accountability. Organizations should implement robust governance structures and regular audits to ensure ethical AI deployment."
    elif 'performance' in question_text:
        answer = "Performance optimization in modern AI systems involves careful consideration of model architecture, training data quality, and computational resources. Recent benchmarks suggest that properly tuned smaller models can often match larger models in specific tasks."
    elif 'future' in question_text:
        answer = "Future trends in AI development point towards more efficient architectures, increased focus on interpretability, and better integration with domain expertise. Quantum computing may also play a significant role in next-generation AI systems."
    else:
        answer = f"Analysis of the question '{question['question']}' reveals several key insights from both academic research and industry applications. The findings suggest a multi-faceted approach would be most effective."

    return {
        "agent_findings": MOCK_FINDINGS,
        "answer": answer
    }

def generate_markdown(research_data):
    """Generate markdown content from research data"""
    markdown_content = f"""
# Research Analysis Report
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}

## Document Analysis
{generate_findings_section(research_data['agent_findings']['rag_agent'])}

## Academic Research
{generate_findings_section(research_data['agent_findings']['arxiv_agent'])}

## Industry Insights
{generate_findings_section(research_data['agent_findings']['web_search_agent'])}
    """
    return markdown_content

def generate_findings_section(findings):
    """Generate markdown for a findings section"""
    return '\n'.join([
        f"### Finding\nDuration: 1\n\n{finding['content']}\n\n*Source*: {finding['source']}\n"
        for finding in findings
    ])

@app.post("/export/pdf")
async def export_pdf(data: dict, _: str = Depends(oauth2_scheme)):
    try:
        # Create a BytesIO buffer to receive PDF data
        buffer = BytesIO()
        
        # Create the PDF document
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )
        
        # Create styles
        styles = getSampleStyleSheet()
        title_style = styles['Heading1']
        heading_style = styles['Heading2']
        normal_style = styles['Normal']
        
        # Build the PDF content
        story = []
        
        # Add title
        story.append(Paragraph(f"Research Report - {datetime.now().strftime('%Y-%m-%d')}", title_style))
        story.append(Spacer(1, 12))
        
        # Add content for each Q&A pair
        for question, result in data.get("qa_history", []):
            # Add question
            story.append(Paragraph("Question:", heading_style))
            story.append(Paragraph(question, normal_style))
            story.append(Spacer(1, 12))
            
            # Add findings
            story.append(Paragraph("Findings:", heading_style))
            for agent, findings in result["agent_findings"].items():
                story.append(Paragraph(f"{agent}:", styles['Heading3']))
                for finding in findings:
                    story.append(Paragraph(f"• {finding['content']}", normal_style))
                story.append(Spacer(1, 6))
            
            # Add answer
            story.append(Paragraph("Answer:", heading_style))
            story.append(Paragraph(result["answer"], normal_style))
            story.append(Spacer(1, 20))
        
        # Build the PDF
        doc.build(story)
        
        # Get the value from the buffer
        pdf_content = buffer.getvalue()
        buffer.close()
        
        return Response(
            content=pdf_content,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename=research_report_{datetime.now().strftime('%Y%m%d')}.pdf"
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {str(e)}")

@app.post("/export/codelab")
async def export_codelab(data: dict, _: str = Depends(oauth2_scheme)):
    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        codelabs_md = f"""
author: Research Tool
summary: Research Analysis Report
id: research-analysis-{timestamp}
categories: research
environments: web
status: Published
feedback link: https://github.com/yourusername/yourrepo
analytics account: UA-123456-1

# Research Analysis Report
Duration: 1

## Overview
Duration: 2

This report presents research findings from multiple sources.

## Document Analysis
Duration: 5

{generate_findings_section(data['agent_findings']['rag_agent'])}

## Academic Research
Duration: 5

{generate_findings_section(data['agent_findings']['arxiv_agent'])}

## Industry Insights
Duration: 5

{generate_findings_section(data['agent_findings']['web_search_agent'])}
"""
        
        # Convert markdown to HTML before returning
        html_content = f"""
        <html>
            <body>
                <h1>Research Analysis Report</h1>
                <h2>Document Analysis</h2>
                {generate_html_section(data['agent_findings']['rag_agent'])}
                
                <h2>Academic Research</h2>
                {generate_html_section(data['agent_findings']['arxiv_agent'])}
                
                <h2>Industry Insights</h2>
                {generate_html_section(data['agent_findings']['web_search_agent'])}
            </body>
        </html>
        """
        
        return Response(
            content=html_content,
            media_type="text/html",
            headers={
                "Content-Disposition": "inline",
                "Cache-Control": "no-cache"
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"HTML generation failed: {str(e)}")

def generate_html_section(findings):
    """Generate HTML for a findings section"""
    return ''.join([
        f"<div class='finding'><p>{finding['content']}</p><p><em>Source: {finding['source']}</em></p></div>"
        for finding in findings
    ])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 