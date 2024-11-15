# DocuQ: Multi-Agent Research Application

This project focuses on building a comprehensive research tool that automates document processing and enables efficient search and exploration of information. It includes a system to parse documents, extract meaningful data, and store it for fast retrieval. The tool allows users to select documents, conduct research through specialized agents, and receive answers to their queries. Additionally, it provides features to save research sessions and export findings in professional formats, ensuring a seamless and efficient research experience.

**Codelabs Link** : [Link](https://codelabs-preview.appspot.com/?file_id=17O8djYyN_MTFYeqBNaczttgK4dczwN2ya2wCeghjpYI/edit?tab=t.0#4)

(The codelabs documentation also includes the link for project demo and deployment URLS)


**WE ATTEST THAT WE HAVEN'T USED ANY OTHER STUDENTS' WORK IN OUR ASSIGNMENT AND ABIDE BY THE POLICIES LISTED IN THE STUDENT HANDBOOK**



## Table of Contents

- [System Architecture](#system-architecture)
- [Architecture Diagram](#architecture-diagram)
- [Prerequisites](#prerequisites)
- [Tech Stack](#tech-stack)
- [Local Development Setup](#local-development-setup)
- [GCP Deployment](#gcp-deployment)
- [Environment Variables](#environment-variables)
- [Contributing](#contributing)
- [License](#license)
- [Contributions](#contributions)

## System Architecture

- **Data Parsing and Vectorization:** Airflow automates the pipeline to parse documents using Docling, generate embeddings, and store them in Pinecone for efficient similarity searches.
- **Metadata and Vector Retrieval:** Metadata from parsed documents is stored in Snowflake for structured access, while Pinecone retrieves vectors for context-aware querying.
- **User Interaction:** Streamlit offers a front-end interface where users can select documents, ask questions, and explore summaries generated in real-time by NVIDIA APIs.
- **Multi-Agent Research:** Langraph coordinates agents like the Arxiv Agent for related research papers, the Web Search Agent for broader context, and the RAG Agent for document-based queries, ensuring users have comprehensive insights.
- **Session Management and Feedback Storage:** User interactions, including queries and results, are stored in Snowflake, facilitating structured reporting and session reviews.
- **Deployment:** Docker containerizes all components, ensuring seamless deployment to the cloud and consistent functionality across environments.


## Architecture Diagram

![image](https://github.com/user-attachments/assets/59201ffc-bb8c-44e8-aded-46a928b0bb7e)


## Prerequisites

- Google Cloud Platform Account
- Docker and Docker Compose
- Python 3.9+
- GCP CLI installed and configured
- Airflow Setup
- Pincone Account
- API Keys : OpenAi API Key and SerpAPI Key

## Tech Stack

### Core Components
- **FastAPI**: Backend API framework
- **Streamlit**: Frontend UI
- **Apache Airflow**: Workflow orchestration
- **Pinecone**: Vector database
- **Docker**: Containerization
- **S3-compatible Storage**: Document storage
- **Poetry**: Dependency management
- **Snowflake** : Storing User data

### AI Components
- RAG(Retrieval Augmented Generation)Agent
- Arxiv Research Agent
- Web Search Agent

## Local Development Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd DocuQ: Multi-Agent Research Application
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

### API Keys

```bash
PINECONE_API_KEY=your_pinecone_key
OPENAI_API_KEY=your_openai_key
SERPAPI_KEY=your_serpapi_key
```

### GCP Configuration

```bash
GCP_PROJECT_ID=your_project_id
GCP_REGION=your_region
```

### Storage

```bash
AWS_BUCKET_NAME=your_bucket_name
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_scret_access_key
AWS_REGION=your_aws_region
```

### Database

```bash
PINECONE_ENVIRONMENT=your_pinecone_environment
PINECONE_INDEX_NAME=your_index_name
SNOWFLAKE_ACCOUNT=your_snowflake_account_name
SNOWFLAKE_USER=your_snowflake_username
SNOWFLAKE_PASSWORD=your_snowflake_password
SNOWFLAKE_DATABASE=your_snowflake_database
SNOWFLAKE_SCHEMA=your_snowflake_schema
SNOWFLAKE_WAREHOUSE=uour_snowflake_warehouse_name
```

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

- **Vaishnavi Veerkumar (40%)** :Configuring Docling to process the provided dataset, extract text, and export structured information(stored in s3 bucket),Storing the parsed document vectors in Pinecone,Airflow pipeline that integrates Docling and Pinecone, Implementation of research agents- RAG, Web Search and Arxiv, Integration report generation to main code
 
- **Siddharth Pawar (30%)** : Implementation of the Application using Streamlit and FastAPI, User registration and login implementation; storing user data in snowflake, Fetching pdfs for user choice and previewing them, Contributed to developing Web Search and Arxiv research agents, Codelabs documentation, Beautification of UI

- **Sriram Venkatesh (30%)** : Streamlit frontend for the application, Codelabs integration for export output, PDF integration for export Q&A research questions, Diagrams code, System Architecture, Readme for the application, Integration with RAG agent, web search agent and Arxiv agent, Dockerizing the application, Creating an image for the fastapi and streamlit application, Deployment of the application on google cloud, GCP setup

