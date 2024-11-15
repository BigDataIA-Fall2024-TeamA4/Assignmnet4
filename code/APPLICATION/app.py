import os
from fastapi import FastAPI, HTTPException, Depends, Body, Security
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
from snowflake.connector import connect
from pinecone import Pinecone
import openai
from dotenv import load_dotenv
from datetime import datetime, timedelta
import bcrypt
from jose import jwt
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from serpapi import GoogleSearch
from sentence_transformers import SentenceTransformer
import logging
import arxiv
import re
import nltk
from nltk.corpus import stopwords
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import textwrap
from fastapi.responses import Response
from io import BytesIO

# Load environment variables
load_dotenv()

# Initialize FastAPI
app = FastAPI()

logging.basicConfig(level=logging.INFO)

# Set up Pinecone and OpenAI
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY
SERPAPI_API_KEY=os.getenv("SERPAPI_API_KEY")

# JWT Secret Key and Expiry
JWT_SECRET = os.getenv("SECRET_KEY")
JWT_ALGORITHM = "HS256"
JWT_EXPIRY_MINUTES = 60  # Token expiry time in minutes

# Define security for JWT token dependency
security = HTTPBearer()

# Initialize Pinecone with API key and environment
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=PINECONE_API_KEY)

# Access the index
pinecone_index = pc.Index("document-embeddings")

# Snowflake connection function
def create_snowflake_connection() -> connect:
    """Create and return a Snowflake connection."""
    try:
        return connect(
            account=os.getenv("SNOWFLAKE_ACCOUNT"),
            user=os.getenv("SNOWFLAKE_USER"),
            password=os.getenv("SNOWFLAKE_PASSWORD"),
            database=os.getenv("SNOWFLAKE_DATABASE"),
            schema=os.getenv("SNOWFLAKE_SCHEMA"),
            warehouse=os.getenv("SNOWFLAKE_WAREHOUSE")
        )
    except Exception as e:
        logging.error(f"Error creating Snowflake connection: {e}")
        raise HTTPException(status_code=500, detail="Database connection error.")

# Define Pydantic models
class User(BaseModel):
    username: str
    password: str

class QueryRequest(BaseModel):
    query: str
    pdf_name: str
    task: str

class ResearchSession(BaseModel):
    username: str
    pdf_name: str
    question: str
    answer: str
    timestamp: datetime = datetime.utcnow()

class GenerateReportRequest(BaseModel):
    query: str
    task: str
    pdf_name: Optional[str] = None

class QuestionSession(BaseModel):
    question: str
    findings: Dict[str, List[str]]  # Dictionary with agent names as keys and list of findings as values
    answer: str

class GenerateReportRequest(BaseModel):
    sessions: List[QuestionSession]  # List of QuestionSession objects

class CodelabsRequest(BaseModel):
    sessions: List[Dict]

# Helper functions for password hashing
def hash_password(password: str) -> str:
    """Hash a password and return the hashed string."""
    salt = bcrypt.gensalt()
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed_password.decode('utf-8')

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify if a plain password matches its hashed version."""
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))

def create_jwt_token(data: dict) -> str:
    """Create and return a JWT token."""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=JWT_EXPIRY_MINUTES)
    to_encode.update({"exp": expire})
    token = jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return token

def verify_jwt_token(credentials: HTTPAuthorizationCredentials = Security(security)) -> dict:
    """Verify a JWT token and return the decoded payload."""
    try:
        payload = jwt.decode(credentials.credentials, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
    
@app.get("/")
async def root():
    return {"message": "Welcome to the FastAPI application!"}

# User signup with password hashing
@app.post("/signup")
def signup(user: User):
    conn = create_snowflake_connection()
    hashed_password = hash_password(user.password)  # Hash the password
    try:
        with conn.cursor() as cur:
            cur.execute("INSERT INTO PUBLICATION.PUBLIC.USERS (USERNAME, PASSWORD_HASH) VALUES (%s, %s)", 
                        (user.username, hashed_password))
            conn.commit()
            return {"message": "User registered successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail="Signup failed")
    finally:
        conn.close()

# User login with JWT token generation
@app.post("/login")
def login(user: User):
    conn = create_snowflake_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT PASSWORD FROM PUBLICATION.PUBLIC.USERS WHERE USERNAME = %s", (user.username,))
            result = cur.fetchone()
            if result and verify_password(user.password, result[0]):
                # Generate and return JWT token on successful login
                token = create_jwt_token({"sub": user.username})
                return {"message": "Login successful", "token": token}
            else:
                raise HTTPException(status_code=400, detail="Invalid credentials")
    finally:
        conn.close()

# Ensure stopwords are downloaded if using nltk

nltk.download("stopwords")
STOPWORDS = set(stopwords.words("english"))

def extract_keywords(question: str) -> set:
    """Extract and return keywords from a given question."""
    words = re.findall(r'\w+', question.lower())
    keywords = {word for word in words if word not in STOPWORDS and len(word) > 2}
    return keywords

# Web Search Agent
def search_web(query: str, max_results: int = 10) -> str:
    serpapi_params = {
        "engine": "google",
        "api_key": os.getenv("SERPAPI_API_KEY")
    }
    search = GoogleSearch({
        **serpapi_params,
        "q": query,
        "num": max_results
    })
    
    try:
        response = search.get_dict()
        if "organic_results" not in response:
            return "No search results found. Please check the query or API limits."
        
        results = response["organic_results"]
        contexts = "\n---\n".join(
            [f"{x['title']}\n{x.get('snippet', 'No snippet available')}\n{x['link']}"
             for x in results]
        )
        return contexts
    except Exception as e:
        print("Error in SerpAPI response:", e)
        return "Error retrieving search results."
    
hf_embed_model_id = "BAAI/bge-small-en-v1.5"
hf_embedding_model = SentenceTransformer(hf_embed_model_id)

def rag_agent(question: str, pdf_name: str) -> dict:
    try:
        query_embedding = hf_embedding_model.encode(question).tolist()
        results = pinecone_index.query(vector=query_embedding, top_k=10, include_metadata=True)
        context = "\n".join(
            match['metadata']['text'] for match in results['matches']
            if 'metadata' in match and 'text' in match['metadata']
        )
        if not context:
            return {
                "answer": "The answer to your question was not found in the document.",
                "source": "no_document_match"
            }
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Use the provided context to answer questions."},
                {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"}
            ],
            max_tokens=150
        )
        return {
            "answer": response['choices'][0]['message']['content'].strip(),
            "source": "pinecone"
        }
    except Exception as e:
        logging.error(f"Error in RAG agent: {e}")
        return {"answer": "Error processing the question.", "source": "error"}
            
def arxiv_agent(query: str, max_results: int = 10) -> str:
    try:
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )
        results = []
        for result in search.results():
            paper_info = f"Title: {result.title}\n" \
                         f"Authors: {', '.join([author.name for author in result.authors])}\n" \
                         f"Summary: {result.summary}\n" \
                         f"Link: {result.entry_id}\n---"
            results.append(paper_info)
        return "\n\n".join(results) if results else "No relevant papers found."
    except Exception as e:
        logging.error(f"Error in Arxiv agent: {e}")
        return "Error retrieving Arxiv results."

def extract_titles_and_links(sessions):
    titles_and_links = []
    for idx, session in enumerate(sessions, 1):
        for agent, findings in session.findings.items():
            if agent in ["Web Search Agent", "Arxiv Agent"]:
                for finding in findings:
                    parts = finding.split("\n")
                    if len(parts) >= 3:
                        title = parts[0]
                        link = parts[-1]
                        titles_and_links.append((idx, title, link))
    return titles_and_links

def generate_pdf(sessions: List[QuestionSession]) -> BytesIO:
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)

    # Define styling constants
    page_width, page_height = letter
    margin_top, margin_bottom, margin_left, margin_right = 70, 40, 50, 50
    title_font, body_font = "Helvetica-Bold", "Helvetica"
    title_size, heading_size, content_size = 16, 12, 10
    line_height = 15

    # Calculate usable area
    y = page_height - margin_top
    max_y = page_height - margin_top
    min_y = margin_bottom + 50

    def new_page():
        nonlocal y
        c.showPage()
        c.setFont(body_font, content_size)
        c.setStrokeColorRGB(0, 0, 0)
        c.setFillColor('black')
        c.setLineWidth(1)
        c.rect(margin_left - 10, margin_bottom - 10, page_width - (2 * margin_left) + 20, page_height - (2 * margin_bottom) + 20)
        page_num = str(c.getPageNumber())
        c.setFont(body_font, 10)
        c.drawString(page_width/2 - 20, margin_bottom - 20, f"Page {page_num}")
        y = max_y
        return y

    # Draw border on first page
    c.setStrokeColorRGB(0, 0, 0)
    c.setLineWidth(1)
    c.rect(margin_left - 10, margin_bottom - 10, page_width - (2 * margin_left) + 20, page_height - (2 * margin_bottom) + 20)
    c.setFont(body_font, 10)
    c.drawString(page_width/2 - 20, margin_bottom - 20, "Page 1")

    # Title
    c.setFont(title_font, title_size)
    title = "Dynamic Research Report"
    title_width = c.stringWidth(title, title_font, title_size)
    c.drawString((page_width - title_width) / 2, y - 20, title)
    y -= 50

    # Overview Section
    c.setFont(title_font, heading_size)
    c.drawString(margin_left, y, "Overview")
    y -= line_height * 2

    # Calculate column widths
    sno_width = 30
    link_width = 200
    title_width = page_width - margin_right - margin_left - sno_width - link_width - 20

    # Draw table headers
    c.setFont(title_font, content_size)
    c.drawString(margin_left, y, "S.No.")
    c.drawString(margin_left + sno_width, y, "Title")
    c.drawString(margin_left + sno_width + title_width, y, "Link")
    y -= line_height

    # Draw horizontal line under headers
    c.line(margin_left, y, page_width - margin_right, y)
    y -= line_height

    # Collect all titles and links
    all_entries = []
    for session in sessions:
        for agent, findings in session.findings.items():
            if agent in ["Web Search Agent", "Arxiv Agent"]:
                for finding in findings:
                    parts = finding.split("\n")
                    if len(parts) >= 3:
                        if agent == "Web Search Agent":
                            # For Web Search, remove "Title: " and "Link: " prefixes if they exist
                            title = parts[0].replace("Title: ", "") if parts[0].startswith("Title: ") else parts[0]
                            link = parts[2].replace("Link: ", "") if parts[2].startswith("Link: ") else parts[2]
                        else:  # Arxiv Agent
                            # For Arxiv, remove "Title: " and "Link: " prefixes
                            title = next((p.replace("Title: ", "") for p in parts if p.startswith("Title:")), "")
                            link = next((p.replace("Link: ", "") for p in parts if p.startswith("Link:")), "")
                        if title and link:
                            all_entries.append((title, link))

    # Draw table content
    c.setFont(body_font, content_size)
    for idx, (title, link) in enumerate(all_entries, 1):
        if y < min_y + 30:  # Add buffer for multi-line entries
            y = new_page()
            # Redraw headers on new page
            c.setFont(title_font, content_size)
            c.drawString(margin_left, y, "S.No.")
            c.drawString(margin_left + sno_width, y, "Title")
            c.drawString(margin_left + sno_width + title_width, y, "Link")
            y -= line_height
            c.line(margin_left, y, page_width - margin_right, y)
            y -= line_height
            c.setFont(body_font, content_size)

        # Draw serial number
        c.drawString(margin_left, y, str(idx))

        # Wrap and draw title
        wrapped_title = textwrap.wrap(title, width=int(title_width/6))
        title_height = len(wrapped_title) * line_height
        for i, line in enumerate(wrapped_title):
            if y - (i * line_height) < min_y:
                y = new_page()
                c.setFont(body_font, content_size)
            c.drawString(margin_left + sno_width, y - (i * line_height), line)

        # Draw link in blue
        c.setFillColor('blue')
        wrapped_link = textwrap.wrap(link, width=int(link_width/6))
        link_height = len(wrapped_link) * line_height
        for i, line in enumerate(wrapped_link):
            if y - (i * line_height) < min_y:
                y = new_page()
                c.setFont(body_font, content_size)
            c.drawString(margin_left + sno_width + title_width, y - (i * line_height), line)
        c.setFillColor('black')

        # Move to next row
        y -= max(title_height, link_height) + 5

    y -= line_height * 2

    # Rest of the report content
    for session_idx, session in enumerate(sessions, 1):
        if y < min_y:
            y = new_page()

        # Question
        c.setFont(title_font, heading_size)
        question_text = f"Question {session_idx}: {session.question}"
        wrapped_question = textwrap.wrap(question_text, width=65)
        for line in wrapped_question:
            if y < min_y:
                y = new_page()
            c.drawString(margin_left, y, line)
            y -= line_height
        y -= 20

        # Research Agents
        if y < min_y:
            y = new_page()
        c.setFont(title_font, heading_size)
        c.drawString(margin_left, y, "Research Agent Used:")
        y -= line_height
        for agent in session.findings.keys():
            if y < min_y:
                y = new_page()
            c.setFont(body_font, content_size)
            c.drawString(margin_left + 20, y, f"• {agent}")
            y -= line_height
        y -= 10

        # Findings
        if y < min_y:
            y = new_page()
        c.setFont(title_font, heading_size)
        c.drawString(margin_left, y, "Findings:")
        y -= line_height

        for agent, agent_findings in session.findings.items():
            if agent_findings:
                if y < min_y:
                    y = new_page()
                c.setFont(title_font, content_size)
                c.drawString(margin_left + 20, y, f"{agent}:")
                y -= line_height

                for finding_idx, finding in enumerate(agent_findings, 1):
                    if agent == "Web Search Agent":
                        parts = finding.split("\n")
                        if len(parts) >= 3:
                            if y < min_y:
                                y = new_page()
                            # Number
                            c.setFont(title_font, content_size)
                            c.drawString(margin_left + 40, y, f"{finding_idx}.")
                            y -= line_height
                            # Title
                            c.setFont(title_font, content_size)
                            wrapped_title = textwrap.wrap(parts[0], width=60)
                            for line in wrapped_title:
                                if y < min_y:
                                    y = new_page()
                                c.drawString(margin_left + 60, y, line)
                                y -= line_height
                            y -= 5
                            # Snippet
                            c.setFont(body_font, content_size)
                            wrapped_snippet = textwrap.wrap(parts[1], width=60)
                            for line in wrapped_snippet:
                                if y < min_y:
                                    y = new_page()
                                c.drawString(margin_left + 60, y, line)
                                y -= line_height
                            y -= 5
                            # Link
                            c.setFont(body_font, content_size)
                            c.setFillColor('blue')
                            wrapped_link = textwrap.wrap(parts[2], width=60)
                            for line in wrapped_link:
                                if y < min_y:
                                    y = new_page()
                                c.drawString(margin_left + 60, y, line)
                                y -= line_height
                            c.setFillColor('black')
                            y -= 15
                    elif agent == "Arxiv Agent":
                        if y < min_y:
                            y = new_page()
                        c.setFont(title_font, content_size)
                        c.drawString(margin_left + 40, y, f"{finding_idx}.")
                        y -= line_height
                        parts = finding.split("\n")
                        full_summary = ""
                        capturing_summary = False
                        for part in parts:
                            if part.startswith("Title:"):
                                if y < min_y:
                                    y = new_page()
                                c.setFont(title_font, content_size)
                                c.drawString(margin_left + 60, y, "Title:")
                                y -= line_height
                                c.setFont(body_font, content_size)
                                wrapped_lines = textwrap.wrap(part.replace("Title:", "").strip(), width=60)
                                for line in wrapped_lines:
                                    if y < min_y:
                                        y = new_page()
                                    c.drawString(margin_left + 60, y, line)
                                    y -= line_height
                                y -= 5
                            elif part.startswith("Authors:"):
                                if y < min_y:
                                    y = new_page()
                                c.setFont(title_font, content_size)
                                c.drawString(margin_left + 60, y, "Authors:")
                                y -= line_height
                                c.setFont(body_font, content_size)
                                wrapped_lines = textwrap.wrap(part.replace("Authors:", "").strip(), width=60)
                                for line in wrapped_lines:
                                    if y < min_y:
                                        y = new_page()
                                    c.drawString(margin_left + 60, y, line)
                                    y -= line_height
                                y -= 5
                            elif part.startswith("Summary:"):
                                capturing_summary = True
                                full_summary = part.replace("Summary:", "").strip()
                            elif capturing_summary and not part.startswith("Link:"):
                                full_summary += " " + part.strip()
                            elif part.startswith("Link:"):
                                if y < min_y:
                                    y = new_page()
                                c.setFont(title_font, content_size)
                                c.drawString(margin_left + 60, y, "Summary:")
                                y -= line_height
                                c.setFont(body_font, content_size)
                                wrapped_lines = textwrap.wrap(full_summary, width=60)
                                for line in wrapped_lines:
                                    if y < min_y:
                                        y = new_page()
                                    c.drawString(margin_left + 60, y, line)
                                    y -= line_height
                                y -= 5
                                if y < min_y:
                                    y = new_page()
                                c.setFont(title_font, content_size)
                                c.drawString(margin_left + 60, y, "Link:")
                                y -= line_height
                                c.setFont(body_font, content_size)
                                c.setFillColor('blue')
                                wrapped_lines = textwrap.wrap(part.replace("Link:", "").strip(), width=60)
                                for line in wrapped_lines:
                                    if y < min_y:
                                        y = new_page()
                                    c.drawString(margin_left + 60, y, line)
                                    y -= line_height
                                c.setFillColor('black')
                                y -= 15
                    elif agent == "RAG Agent":
                        if y < min_y:
                            y = new_page()
                        c.setFont(title_font, content_size)
                        c.drawString(margin_left + 40, y, f"{finding_idx}.")
                        y -= line_height
                        c.setFont(body_font, content_size)
                        wrapped_lines = textwrap.wrap(finding, width=60)
                        for line in wrapped_lines:
                            if y < min_y:
                                y = new_page()
                            c.drawString(margin_left + 60, y, line)
                            y -= line_height
                        y -= 10
                y -= 20
        y -= 30

    c.save()
    buffer.seek(0)
    return buffer

def generate_codelabs_markdown(sessions, output_path="codelabs_output.md"):
    try:
        with open(output_path, "w") as f:
            f.write("# Dynamic Research Report\n\n")
            for idx, session in enumerate(sessions, 1):
                question = session["question"]
                findings = session["findings"]
                answer = session["answer"]

                # Section title for each question
                f.write(f"## Question {idx}\n\n")

                # Question Section
                f.write("### Question\n")
                f.write(f"{question}\n\n")

                # Findings Section
                f.write("### Findings\n")
                for agent, agent_findings in findings.items():
                    f.write(f"#### {agent}\n")
                    if agent == "Web Search Agent":
                        for point in agent_findings:
                            parts = point.split("\n")
                            if len(parts) >= 3:
                                f.write(f"- **Title:** {parts[0]}\n")
                                f.write(f"  **Snippet:** {parts[1]}\n")
                                f.write(f"  **Link:** {parts[2]}\n")
                    elif agent == "Arxiv Agent":
                        for point in agent_findings:
                            parts = point.split("\n")
                            title = authors = summary = link = ""
                            for part in parts:
                                if part.startswith("Title:"):
                                    title = part.replace("Title:", "").strip()
                                elif part.startswith("Authors:"):
                                    authors = part.replace("Authors:", "").strip()
                                elif part.startswith("Summary:"):
                                    summary = part.replace("Summary:", "").strip()
                                elif part.startswith("Link:"):
                                    link = part.replace("Link:", "").strip()
                            f.write(f"- **Title:** {title}\n")
                            f.write(f"  **Authors:** {authors}\n")
                            f.write(f"  **Summary:** {summary}\n")
                            f.write(f"  **Link:** {link}\n")
                    elif agent == "RAG Agent":
                        for point in agent_findings:
                            f.write(f"- {point}\n")
                    f.write("\n")

                # Answer Section
                f.write("### Answer\n")
                f.write(f"{answer}\n\n")

        print(f"Codelabs Markdown generated at {output_path}")
    except Exception as e:
        print(f"Error generating Codelabs Markdown: {e}")

# Research session endpoint with error handling
@app.post("/research")
def run_research(request: QueryRequest, token_data: dict = Depends(verify_jwt_token)):
    try:
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="The question field is empty.")
        if request.task == "WEBSEARCH":
            return JSONResponse(content={"result": search_web(request.query)})
        elif request.task == "RAG":
            rag_response = rag_agent(request.query, request.pdf_name)
            return JSONResponse(content=rag_response) 
        elif request.task == "ARXIV":
            return JSONResponse(content={"result": arxiv_agent(request.query)})
        else:
            raise HTTPException(status_code=400, detail="Invalid task type.")
    except Exception as e:
        logging.error(f"Error in research endpoint: {e}")
        raise HTTPException(status_code=500, detail="Error processing the research request.")

# Save research session endpoint with error handling
@app.post("/save_session")
def save_session(session: ResearchSession, token_data: dict = Depends(verify_jwt_token)):
    try:
        conn = create_snowflake_connection()
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO PUBLICATION.PUBLIC.RESEARCH_SESSIONS (USERNAME, PDF_NAME, QUESTION, ANSWER, TIMESTAMP) VALUES (%s, %s, %s, %s, %s)",
                (session.username, session.pdf_name, session.question, session.answer, session.timestamp)
            )
            conn.commit()
            return {"message": "Session saved successfully."}
    except Exception as e:
        logging.error(f"Error saving session: {e}")
        raise HTTPException(status_code=500, detail="Error saving research session.")
    finally:
        if 'conn' in locals():
            conn.close()

@app.post("/generate_dynamic_reports")
def generate_dynamic_reports(request: GenerateReportRequest, 
                           token_data: dict = Depends(verify_jwt_token)):
    try:
        pdf_buffer = generate_pdf(request.sessions)
        
        return Response(
            content=pdf_buffer.getvalue(),
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename=research_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            }
        )
    except Exception as e:
        logging.error(f"Error generating reports: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate_codelabs")
async def generate_codelabs(request: CodelabsRequest):
    try:
        codelabs_content = """
        author: Research Assistant
        summary: Research Report
        id: dynamic-research-report
        categories: research
        environments: web
        status: Published
        """
        
        for idx, session in enumerate(request.sessions, 1):
            codelabs_content += f"\n## Question {idx}\n"
            codelabs_content += f"{session['question']}\n\n"
            codelabs_content += "### Findings\n"
            
            for agent, findings in session['findings'].items():
                codelabs_content += f"\n#### {agent}\n"
                for finding in findings:
                    codelabs_content += f"* {finding}\n"
            
            codelabs_content += f"\n### Answer\n{session.get('answer', '')}\n"

        return {
            "status": "success",
            "codelabs_content": codelabs_content
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))