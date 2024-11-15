# PDF Q&A System

A sophisticated document processing and question-answering system built with modern cloud architecture, containerized with Docker, and deployed on Google Cloud Platform (GCP).

WE ATTEST THAT WE HAVEN'T USED ANY OTHER STUDENTS' WORK IN OUR ASSIGNMENT AND ABIDE BY THE POLICIES LISTED IN THE STUDENT HANDBOOK

## Table of Contents

- [System Architecture](#system-architecture)
- [Prerequisites](#prerequisites)
- [Tech Stack](#tech-stack)
- [Local Development Setup](#local-development-setup)
- [GCP Deployment](#gcp-deployment)
- [Environment Variables](#environment-variables)
- [Contributing](#contributing)
- [License](#license)
- [Contributions](#contributions)

## System Architecture

The system consists of multiple layers:
- Document Source: PDF ingestion through S3-compatible storage
- Data Processing: Airflow-orchestrated pipeline with document processing
- Vector Storage: Pinecone for efficient vector embeddings
- Application Layer: FastAPI backend with AI agents and Streamlit frontend
- Export Options: PDF and Codelabs export capabilities

**Architecture Diagram**

[Insert Architecture Diagram]

## Prerequisites

- Google Cloud Platform Account
- Docker and Docker Compose
- Python 3.9+
- GCP CLI installed and configured

## Tech Stack

### Core Components
- **FastAPI**: Backend API framework
- **Streamlit**: Frontend UI
- **Apache Airflow**: Workflow orchestration
- **Pinecone**: Vector database
- **Docker**: Containerization
- **S3-compatible Storage**: Document storage
- **Poetry**: Dependency management

### AI Components
- RAG (Retrieval Augmented Generation) Agent
- Arxiv Research Agent
- Web Search Agent

## Local Development Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd pdf-qa-system
   ```

2. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configurations
   ```

3. **Build and run with Docker Compose**
   ```bash
   docker-compose up --build
   ```

## GCP Deployment

### Prerequisites
1. Enable required GCP APIs:
   ```bash
   gcloud services enable container.googleapis.com \
       cloudbuild.googleapis.com \
       artifactregistry.googleapis.com
   ```

2. Create a GCP project and set it as default:
   ```bash
   gcloud config set project [PROJECT_ID]
   ```

### Deployment Steps

1. **Create GCP Container Registry**
   ```bash
   gcloud artifacts repositories create pdf-qa-repo \
       --repository-format=docker \
       --location=us-central1
   ```

2. **Build and Push Docker Images**
   ```bash
   # Configure Docker to use GCP authentication
   gcloud auth configure-docker us-central1-docker.pkg.dev

   # Build and push images
   docker build -t us-central1-docker.pkg.dev/[PROJECT_ID]/pdf-qa-repo/backend:latest ./backend
   docker build -t us-central1-docker.pkg.dev/[PROJECT_ID]/pdf-qa-repo/frontend:latest ./frontend
   docker build -t us-central1-docker.pkg.dev/[PROJECT_ID]/pdf-qa-repo/airflow:latest ./airflow

   docker push us-central1-docker.pkg.dev/[PROJECT_ID]/pdf-qa-repo/backend:latest
   docker push us-central1-docker.pkg.dev/[PROJECT_ID]/pdf-qa-repo/frontend:latest
   docker push us-central1-docker.pkg.dev/[PROJECT_ID]/pdf-qa-repo/airflow:latest
   ```

3. **Deploy to GCP**
   ```bash
   # Apply Kubernetes configurations
   kubectl apply -f k8s/
   ```

## Environment Variables

Create a `.env` file with the following variables:
# API Keys
PINECONE_API_KEY=your_pinecone_key
OPENAI_API_KEY=your_openai_key

# GCP Configuration
GCP_PROJECT_ID=your_project_id
GCP_REGION=your_region

# Storage
BUCKET_NAME=your_bucket_name

# Database
PINECONE_ENVIRONMENT=your_pinecone_environment
PINECONE_INDEX_NAME=your_index_name

# Application
BACKEND_URL=http://localhost:8000
FRONTEND_URL=http://localhost:8501

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License

Copyright (c) 2024

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Contributions

This project is the result of collaborative efforts from the following contributors:

- [Contributor 1](https://github.com/contributor1)
- [Contributor 2](https://github.com/contributor2)
- [Contributor 3](https://github.com/contributor3)

