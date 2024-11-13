import os
import logging
from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator
from dotenv import load_dotenv
import boto3
from botocore.exceptions import NoCredentialsError
from botocore.config import Config
from pinecone import Pinecone
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.utils.export import generate_multimodal_pages
from langchain_core.documents import Document as LCDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from typing import List, Dict
import pandas as pd
from io import BytesIO
from PIL import Image

# Load environment variables from .env file
load_dotenv()
S3_BUCKET = os.getenv("AWS_BUCKET_NAME")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
HF_EMBED_MODEL_ID = "BAAI/bge-small-en-v1.5"
s3_client = boto3.client("s3")

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
pinecone_index = pc.Index("document-embeddings")

# Initialize the embedding model and text splitter
embeddings = HuggingFaceEmbeddings(model_name=HF_EMBED_MODEL_ID)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Airflow DAG definition
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 0
}
dag = DAG(
    "docling_embed_pipeline",
    default_args=default_args,
    description="Process a single PDF from S3, embed using docling, store in Pinecone, and export structured data",
    schedule_interval=None,
    start_date=datetime(2023, 1, 1),
    catchup=False,
)

def convert_image_to_bytes(image: Image.Image) -> bytes:
    """Convert a PIL image to bytes."""
    with BytesIO() as output:
        image.save(output, format="PNG")  
        return output.getvalue()
    
def select_pdfs(max_size_mb: float = 1.5) -> List[str]:
    """Select two PDF file paths from S3 that are below the specified size (1.5 MB default), sorted by name."""
    s3 = boto3.client("s3")
    response = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix="pdfs")
    
    max_size_bytes = max_size_mb * 1024 * 1024  # Convert MB to bytes
    
    # Sort PDFs by Key (name) to ensure consistency
    pdf_objects = sorted(
        response.get("Contents", []),
        key=lambda x: x["Key"]
    )
    
    selected_pdfs = [
        obj["Key"] for obj in pdf_objects if obj["Key"].endswith(".pdf") and obj["Size"] <= max_size_bytes
    ][:2]  # Select only the first two suitable PDFs
    
    if selected_pdfs:
        logger.info(f"Selected PDFs: {selected_pdfs}")
    else:
        logger.warning("No suitable PDFs found under the size limit.")
    return selected_pdfs

def generate_signed_url(pdf_key: str) -> str:
    """Generate a signed URL for a single PDF."""
    logger.info("Generating signed URL for the selected PDF.")
    try:
        s3_client = boto3.client(
            "s3",
            region_name="us-east-2",
            config=Config(signature_version='s3v4')
        )
        signed_url = s3_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": S3_BUCKET, "Key": pdf_key},
            ExpiresIn=3600
        )
        logger.info(f"Generated signed URL: {signed_url}")
        return signed_url
    except Exception as e:
        logger.error("Error generating signed URL for the PDF: %s", e, exc_info=True)
        raise

def upload_dataframe_to_s3(df: pd.DataFrame, bucket_name: str, s3_key: str):
    """Upload DataFrame as a Parquet file to S3 without saving locally."""
    try:
        parquet_buffer = BytesIO()
        df.to_parquet(parquet_buffer, index=False)
        parquet_buffer.seek(0)
        
        s3_client.upload_fileobj(parquet_buffer, bucket_name, s3_key)
        logger.info(f"Uploaded DataFrame to s3://{bucket_name}/{s3_key}")
    except NoCredentialsError:
        logger.error("Credentials not available for S3 upload.")
        raise

def load_convert_and_export_pdf(url: str, doc_name: str) -> List[Dict[str, str]]:
    """Load a single PDF using Docling, convert to text, export structured data directly to S3, and split into chunks."""
    logger.info(f"Loading and converting PDF: {doc_name} from URL: {url}")
    try:
        rows = []
        options = PdfPipelineOptions(
            ocr=True,
            extract_tables=True,
            images_scale=2.0,
            generate_page_images=True
        )
        converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)})
        conv_res = converter.convert(url)

        # Process pages to export structured data and append to rows
        for content_text, content_md, content_dt, page_cells, page_segments, page in generate_multimodal_pages(conv_res):
            rows.append({
                "content_text": content_text,
                "content_md": content_md,
                "content_dt": content_dt,
                "page_cells": page_cells,
                "segments": page_segments,
                "page_image": convert_image_to_bytes(page.image) if page.image else None
            })

        # Convert rows to DataFrame and upload directly to S3 as Parquet
        df = pd.DataFrame(rows).drop(columns=["page_image"], errors="ignore")  # Drop page_image for S3 compatibility
        s3_key = f"structured_data/{doc_name}_structured_data.parquet"
        upload_dataframe_to_s3(df, S3_BUCKET, s3_key)

        # Convert to markdown and split into chunks
        text = conv_res.document.export_to_markdown()
        logger.info(f"Successfully converted PDF to text (first 100 chars): {text[:100]}")

        # Create dictionaries from chunks
        chunks = [{"page_content": chunk.page_content} for chunk in text_splitter.split_documents([LCDocument(page_content=text)])]
        logger.info(f"Document split into {len(chunks)} chunks.")
        
        return chunks
    except Exception as e:
        logger.error(f"Error loading and converting PDF {doc_name}: {e}", exc_info=True)
        raise

def check_existing_embeddings(docs: List[Dict[str, str]], doc_name: str) -> List[Dict[str, str]]:
    """Check for existing embeddings in Pinecone for each chunk of the document."""
    logger.info("Checking for existing embeddings in Pinecone for document: %s", doc_name)
    try:
        unprocessed_chunks = []
        
        # Create LCDocument objects directly from dictionary data
        doc_objects = [LCDocument(page_content=chunk["page_content"]) for chunk in docs]
        
        for index, doc in enumerate(doc_objects):
            chunk_id = f"{doc_name}-chunk-{index}"
            if not pinecone_index.fetch(ids=[chunk_id]).get("matches"):
                unprocessed_chunks.append({"id": chunk_id, "content": doc.page_content})
                logger.info("Chunk %s is new and will be processed.", chunk_id)
            else:
                logger.info("Chunk %s already exists in Pinecone. Skipping embedding.", chunk_id)
        
        return unprocessed_chunks
    except Exception as e:
        logger.error("Error checking existing embeddings: %s", e, exc_info=True)
        raise

def embed_and_store(unprocessed_chunks: List[Dict[str, str]]) -> None:
    """Embed text chunks and store in Pinecone."""
    logger.info("Starting embedding and storing process for unprocessed chunks.")
    try:
        for chunk in unprocessed_chunks:
            embedding_vector = embeddings.embed_query(chunk["content"])
            pinecone_index.upsert([(chunk["id"], embedding_vector)])
            logger.info(f"Stored embedding for chunk ID: {chunk['id']}")
    except Exception as e:
        logger.error("Error during embedding and storing process: %s", e, exc_info=True)
        raise

with dag:
    pdf_keys = select_pdfs()  # Retrieve two PDFs sorted by name
    if not pdf_keys:
        raise ValueError("No suitable PDFs found to process.")
    
    previous_task = None
    for i, pdf_key in enumerate(pdf_keys):
        doc_name = f"Document_{i+1}"
        
        generate_signed_url_task = PythonOperator(
            task_id=f"generate_signed_url_{i+1}",
            python_callable=generate_signed_url,
            op_kwargs={"pdf_key": pdf_key},
        )

        load_convert_and_export_pdf_task = PythonOperator(
            task_id=f"load_convert_and_export_pdf_{i+1}",
            python_callable=load_convert_and_export_pdf,
            op_kwargs={
                "url": generate_signed_url_task.output,
                "doc_name": doc_name
            },
        )

        check_existing_embeddings_task = PythonOperator(
            task_id=f"check_existing_embeddings_{i+1}",
            python_callable=check_existing_embeddings,
            op_kwargs={
                "docs": load_convert_and_export_pdf_task.output,
                "doc_name": doc_name
            },
        )

        embed_and_store_task = PythonOperator(
            task_id=f"embed_and_store_{i+1}",
            python_callable=embed_and_store,
            op_kwargs={
                "unprocessed_chunks": check_existing_embeddings_task.output
            },
        )

        # Sequentially link tasks for this PDF
        generate_signed_url_task >> load_convert_and_export_pdf_task >> check_existing_embeddings_task >> embed_and_store_task
        
        # Ensure all tasks for the previous PDF complete before starting the next PDF
        if previous_task:
            previous_task >> generate_signed_url_task
        
        # Update the previous_task to the last task of the current PDF
        previous_task = embed_and_store_task