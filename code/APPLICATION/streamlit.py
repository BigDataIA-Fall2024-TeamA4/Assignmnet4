import streamlit as st
import requests
import boto3
import os
from dotenv import load_dotenv

# Set up environment variables and client for S3 access

load_dotenv()
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION")
S3_BUCKET = os.getenv("AWS_BUCKET_NAME")
API_URL = os.getenv("API_URL")
s3_client = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION
)

# Streamlit session state keys for logged-in status and question limit
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "questions_left" not in st.session_state:
    st.session_state.questions_left = 5

# Function for login/signup
def login_signup(username, password, signup=False):
    endpoint = "/signup" if signup else "/login"
    response = requests.post(f"{API_URL}{endpoint}", json={"username": username, "password": password})
    if response.status_code == 200:
        st.session_state.logged_in = True
        st.session_state.token = response.json().get("token")
        st.session_state.username = username  # Store username in session_state
        st.session_state.questions_left = 5  # Reset question limit
        st.success("Logged in successfully!" if not signup else "Signed up and logged in successfully!")
    else:
        st.error("Login failed!" if not signup else "Signup failed!")

# Function for fetching PDF list from S3
def get_pdf_list():
    specific_pdfs = {"pdfs_new/becker-rf-lit-review-2018.pdf", "pdfs_new/cash-flow-focus-endowments-trusts.pdf"}
    pdf_list = []

    try:
        # List objects in the specified S3 bucket and folder
        response = s3_client.list_objects_v2(Bucket=S3_BUCKET, Prefix="pdfs_new/")
        for obj in response.get("Contents", []):
            # Check if the object key matches one of the specific PDFs
            if obj["Key"] in specific_pdfs:
                # Get the PDF filename
                file_name = obj["Key"].split("/")[-1]
                pdf_list.append(file_name)
    except Exception as e:
        st.error("Error fetching PDF list from S3.")
        print(f"Error: {e}")
    
    return pdf_list

# Function for RAG-based Q&A
def ask_question(pdf_name, question):
    if st.session_state.questions_left > 0:
        headers = {"Authorization": f"Bearer {st.session_state.token}"}
        
        # Define the payload to be sent
        payload = {
            "query": question,
            "pdf_name": pdf_name,
            "task": "RAG"
        }
        
        # Print the payload before sending the request
        print(f"Payload to /research: {payload}")
        
        # Send the request with the payload
        response = requests.post(f"{API_URL}/research", json=payload, headers=headers)
        
        if response.status_code == 200:
            result = response.json()
            answer = result.get("answer")
            source = result.get("source", "unknown")
            st.session_state.questions_left -= 1

            # Display answer and source
            if answer:
                st.write("**Answer:**")
                st.write(answer)
                if source == "web_search":
                    st.info("Answer obtained from web search.")
                else:
                    st.info("Answer obtained from document embeddings.")
        else:
            st.error("Error in fetching answer.")
    else:
        st.warning("Question limit reached.")


# Function for web search
def web_search(query):
    headers = {"Authorization": f"Bearer {st.session_state.token}"}
    response = requests.post(f"{API_URL}/research", json={
        "query": query,
        "pdf_name": "",  # Add an empty string for pdf_name
        "task": "WEBSEARCH"
    }, headers=headers)
    
    # Check if the response is JSON
    try:
        response_data = response.json()
        print(f"Web search response: {response_data}")
        
        if response.status_code == 200:
            return response_data.get("result")
        else:
            st.error(f"Error in web search: {response_data.get('detail', 'Unknown error')}")
    except requests.exceptions.JSONDecodeError:
        # Print raw response text if it's not JSON
        print("Non-JSON response received:", response.text)
        st.error("Received a non-JSON response from the server. Please check the server logs for more details.")


# Main app function
def main():
    st.title("PDF Q&A and Web Search")

    if not st.session_state.logged_in:
        # Login/Signup interface
        st.subheader("Login or Signup")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            login_signup(username, password, signup=False)
        if st.button("Signup"):
            login_signup(username, password, signup=True)
    else:
        # Main app after login
        st.sidebar.subheader(f"Welcome, {st.session_state.get('username', 'User')}!")  # Access username from session_state
        pdf_options = get_pdf_list()
        selected_pdf = st.selectbox("Choose a PDF", pdf_options)
        
        st.write(f"You have {st.session_state.questions_left} questions remaining for this session.")
        
        # Question Input for PDF
        question = st.text_input("Ask a question about the PDF")
        if st.button("Submit Question"):
            if question:
                answer = ask_question(selected_pdf, question)
                if answer:
                    st.write("**Answer:**")
                    st.write(answer)

        # Web Search Option
        st.subheader("Or perform a web search")
        web_query = st.text_input("Enter a query for web search")
        if st.button("Search"):
            if web_query:
                search_results = web_search(web_query)
                if search_results:
                    st.write("**Web Search Results:**")
                    st.write(search_results)

# Run the main app function
if __name__ == "__main__":
    main()
