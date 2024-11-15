from diagrams import Diagram, Cluster
from diagrams.programming.framework import FastAPI
from diagrams.onprem.workflow import Airflow
from diagrams.generic.database import SQL
from diagrams.custom import Custom
from diagrams.aws.storage import S3

# Create the architecture diagram
with Diagram("PDF Q&A System Architecture", show=True, direction="LR"):
    # Data Source Layer
    with Cluster("Document Source"):
        pdf_docs = Custom("PDF Documents", "./cfa.jpeg")
        s3_bucket = S3("S3 Bucket")

    # Data Processing Layer
    with Cluster("Data Processing"):
        airflow = Airflow("Airflow Pipeline")
        docling = Custom("Docling Processor", "./docling_icon.jpg")
        
        with Cluster("Vector Storage"):
            pinecone = Custom("Pinecone DB", "./pinecone.jpeg")

    # Application Layer
    with Cluster("Application Layer"):
        fastapi = FastAPI("FastAPI Backend")
        streamlit = Custom("Streamlit UI", "./streamlit.jpeg")
        
        # AI Agents within Application Layer
        with Cluster("AI Agents"):
            archive_agent = Custom("Arxiv Agent", "./agent_icon.png")
            web_agent = Custom("Web Search Agent", "./web.png")
            rag_agent = Custom("RAG Agent", "./rag.png")

    # Export Layer
    with Cluster("Export Options"):
        pdf_export = Custom("PDF Export", "./pdf_icon.png")
        codelabs = Custom("Codelabs Export", "./codelabs_icon.png")

    # Define the flows
    # Data Processing Flow
    pdf_docs >> s3_bucket >> airflow >> docling >> pinecone

    # Pinecone only connects to RAG agent
    pinecone >> rag_agent
    
    # AI Agents all connect to FastAPI
    [archive_agent, web_agent, rag_agent] >> fastapi
    fastapi >> streamlit
    
    # Agent Flow from Streamlit
    streamlit >> [archive_agent, web_agent, rag_agent]

    # Export Flow
    streamlit >> pdf_export
    streamlit >> codelabs