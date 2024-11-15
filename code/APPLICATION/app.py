import os
from fastapi import FastAPI, HTTPException, Depends, Body, Security
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List
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
def create_snowflake_connection():
    return connect(
        account=os.getenv("SNOWFLAKE_ACCOUNT"),
        user=os.getenv("SNOWFLAKE_USER"),
        password=os.getenv("SNOWFLAKE_PASSWORD"),
        database=os.getenv("SNOWFLAKE_DATABASE"),
        schema=os.getenv("SNOWFLAKE_SCHEMA"),
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE")
    )

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

# Helper functions for password hashing
def hash_password(password: str) -> str:
    salt = bcrypt.gensalt()
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed_password.decode('utf-8')

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))

def create_jwt_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=JWT_EXPIRY_MINUTES)
    to_encode.update({"exp": expire})
    token = jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return token

# Function to verify JWT token
def verify_jwt_token(credentials: HTTPAuthorizationCredentials = Security(security)):
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
    # Remove punctuation and split the question into words
    words = re.findall(r'\w+', question.lower())
    # Filter out common stopwords and short words (e.g., "a", "an", "is")
    keywords = {word for word in words if word not in STOPWORDS and len(word) > 2}
    return keywords

# Web Search Agent
def search_web(query: str, max_results: int = 5) -> str:
    # Initialize GoogleSearch client with API key
    serpapi_params = {
        "engine": "google",
        "api_key": os.getenv("SERPAPI_API_KEY")
    }
    search = GoogleSearch({
        **serpapi_params,
        "q": query,
        "num": max_results
    })

    # Try to fetch results and handle potential errors
    try:
        response = search.get_dict()
        print("Raw response from SerpAPI:", response)  # Debug: Print the raw response

        # Check if "organic_results" is present
        if "organic_results" not in response:
            return "No search results found. Please check the query or API limits."

        # Format the results
        results = response["organic_results"]
        contexts = "\n---\n".join(
            ["\n".join([x["title"], x.get("snippet", "No snippet available"), x["link"]]) for x in results]
        )
        return contexts

    except Exception as e:
        print("Error in SerpAPI response:", e)
        return "Error retrieving search results."


hf_embed_model_id = "BAAI/bge-small-en-v1.5"
hf_embedding_model = SentenceTransformer(hf_embed_model_id)

def rag_agent(question: str, pdf_name: str) -> dict:
    logging.basicConfig(level=logging.INFO)
    
    # Generate query embedding for the question
    query_embedding = hf_embedding_model.encode(question).tolist()

    # Query Pinecone for relevant document chunks
    results = pinecone_index.query(vector=query_embedding, top_k=10, include_metadata=True)
    logging.info(f"Raw results from Pinecone: {results}")

    # Filter and extract context from the metadata of top matches
    context = "\n".join(
        match['metadata']['text'] for match in results['matches']
        if 'metadata' in match and 'text' in match['metadata']
    )

    # Extract keywords from the question
    question_keywords = extract_keywords(question)
    logging.info(f"Extracted keywords from question: {question_keywords}")

    # Check if any of the question's keywords are in the context
    if not context or not any(keyword in context.lower() for keyword in question_keywords):
        # No relevant context found in the document
        message = (
            "The answer to your question was not found in the document. "
            "Consider using Web Search or Arxiv Search for broader results."
        )
        return {
            "answer": message,
            "source": "no_document_match",
            "suggested_search": ["web_search", "arxiv_search"]
        }

    # Log the final context for debugging
    logging.info(f"Context passed to OpenAI: {context[:500]}")  # Log first 500 characters for brevity

    # Generate answer from OpenAI using the retrieved context
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
        "source": "pinecone",
        "fallback_to_web_search": False
    }

def arxiv_agent(query: str, max_results: int = 5) -> str:
    # Search for papers on arXiv based on the query
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )
    
    # Format the results for display
    results = []
    for result in search.results():
        paper_info = f"Title: {result.title}\n" \
                     f"Authors: {', '.join([author.name for author in result.authors])}\n" \
                     f"Summary: {result.summary}\n" \
                     f"Link: {result.entry_id}\n---"
        results.append(paper_info)
    
    # Join all results as a single formatted string
    return "\n\n".join(results) if results else "No relevant papers found."

# Research session endpoint with JWT protection

@app.post("/research")
def run_research(request: QueryRequest):
    logging.info(f"Received request payload: {request}")

    if request.task == "WEBSEARCH":
        web_search_results = search_web(request.query)
        return JSONResponse(content={"result": web_search_results})
    
    elif request.task == "RAG":
        rag_response = rag_agent(request.query, request.pdf_name)  # Assuming `rag_agent` is defined elsewhere in your code
        return JSONResponse(content=rag_response)

    elif request.task == "ARXIV":
        arxiv_results = arxiv_agent(request.query)
        return JSONResponse(content={"result": arxiv_results})
    
    else:
        raise HTTPException(status_code=400, detail="Invalid task type")

# Save research session with JWT protection
@app.post("/save_session")
def save_session(session: ResearchSession, token_data: dict = Depends(verify_jwt_token)):
    conn = create_snowflake_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("INSERT INTO PUBLICATION.PUBLIC.RESEARCH_SESSIONS (USERNAME, PDF_NAME, QUESTION, ANSWER, TIMESTAMP) VALUES (%s, %s, %s, %s, %s)",
                        (session.username, session.pdf_name, session.question, session.answer, session.timestamp))
            conn.commit()
            return {"message": "Session saved successfully"}
    finally:
        conn.close()
