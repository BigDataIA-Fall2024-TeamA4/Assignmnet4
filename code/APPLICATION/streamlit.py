import streamlit as st
import requests
import boto3
import os
import re
from dotenv import load_dotenv
from io import BytesIO
import base64
from typing import List, Optional, Union
from datetime import datetime

# Configure page layout and style
st.set_page_config(layout="wide")  # Set layout to wide

# Custom CSS for black background and text color
st.markdown(
    """
    <style>
    /* Set main background to black and text to white */
    body, .stApp {
        background-color: black;
        color: white;
    }

    /* Make labels big and bold */
    label {
        font-size: 18px !important;
        font-weight: bold !important;
        color: white !important;
    }

    /* Remove border and background color from select boxes and input fields, make text large and bold */
    .stSelectbox, .stTextInput {
        background-color: black !important;
        color: white !important;
        border: none !important;
        box-shadow: none !important;
        font-size: 18px !important;
        font-weight: bold !important;
    }

    /* Set button text to white and adjust button background */
    .stButton>button {
        color: white !important;
        background-color: #444444 !important;
        border: 1px solid white !important;
        font-size: 18px !important;
        font-weight: bold !important;
    }

    /* Make header text white */
    h1, h2, h3, h4, h5, h6 {
        color: white !important;
    }
    </style>

    <style>
    /* Center the title */
    .title-center {
        text-align: center;
        font-size: 32px;
        font-weight: bold;
        color: white;
    }
    </style>
    <h1 class="title-center">DocuQ: RAG, Web Search and Arxiv Research Agents</h1>
    
    """,
    unsafe_allow_html=True
)

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
if "session_data" not in st.session_state:
    st.session_state.session_data = []

def is_valid_password(password: str) -> Optional[str]:
    try:
        if len(password) < 8:
            return "Password should be at least 8 characters long"
        if not re.search(r"[A-Z]", password):
            return "Password should contain at least one uppercase letter"
        if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
            return "Password should contain at least one special character"
        return None
    except Exception as e:
        st.error("Error in password validation.")
        return str(e)

# Function for login/signup with password validation and username uniqueness check
def login_signup(username: str, password: str, signup: bool = False) -> None:
    try:
        if signup:
            check_response = requests.get(f"{API_URL}/check_username", json={"username": username})
            if check_response.status_code == 409:
                st.error("Username already exists. Please choose a different username.")
                return
            elif check_response.status_code != 200:
                st.error("Error checking username availability.")
                return

            password_error = is_valid_password(password)
            if password_error:
                st.error(password_error)
                return

        endpoint = "/signup" if signup else "/login"
        response = requests.post(f"{API_URL}{endpoint}", json={"username": username, "password": password})
        if response.status_code == 200:
            st.session_state.logged_in = True
            st.session_state.token = response.json().get("token")
            st.session_state.username = username
            st.session_state.questions_left = 5
            st.success("Logged in successfully!" if not signup else "Signed up and logged in successfully!")
        else:
            error_message = response.json().get("detail", "Signup/Login failed!")
            st.error(error_message)
    except Exception as e:
        st.error("Error in login/signup process.")
        print(f"Error: {e}")

def logout():
    """Function to handle logout"""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.session_state.logged_in = False


# Function for fetching PDF list from S3
def get_pdf_list() -> List[str]:
    try:
        specific_pdfs = {"pdfs_new/becker-rf-lit-review-2018.pdf", "pdfs_new/cash-flow-focus-endowments-trusts.pdf"}
        pdf_list = []
        response = s3_client.list_objects_v2(Bucket=S3_BUCKET, Prefix="pdfs_new/")
        for obj in response.get("Contents", []):
            if obj["Key"] in specific_pdfs:
                file_name = obj["Key"].split("/")[-1]
                pdf_list.append(file_name)
        return pdf_list
    except Exception as e:
        st.error("Error fetching PDF list from S3.")
        print(f"Error: {e}")
        return []


def preview_pdf(pdf_name: str) -> None:
    try:
        response = s3_client.get_object(Bucket=S3_BUCKET, Key=f"pdfs_new/{pdf_name}")
        pdf_bytes = response['Body'].read()
        base64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="900" height="800" type="application/pdf"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)
    except Exception as e:
        st.error("Error fetching PDF content from S3.")
        print(f"Error: {e}")

# Function for RAG-based Q&A
def ask_question(pdf_name: str, question: str) -> Optional[str]:
    try:
        if not question.strip():
            st.error("Please enter a query to search.")
            return None

        if st.session_state.questions_left > 0:
            headers = {"Authorization": f"Bearer {st.session_state.token}"}
            payload = {
                "query": question,
                "pdf_name": pdf_name,
                "task": "RAG"
            }
            response = requests.post(f"{API_URL}/research", json=payload, headers=headers)
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, dict) and "answer" in result:
                    st.session_state.questions_left -= 1
                    return result
                else:
                    st.error("Unexpected response format from server.")
                    return None
            else:
                st.error(f"Error: {response.json().get('detail', 'Failed to fetch answer.')}")
                return None
        else:
            st.warning("Question limit reached.")
            return None
    except Exception as e:
        st.error(f"Error processing question: {str(e)}")
        return None
    
# Function for web search
# Function for web search
def web_search(query):
    if not query.strip():
        st.error("Please enter a question to search.")
        return
    
    headers = {"Authorization": f"Bearer {st.session_state.token}"}
    response = requests.post(f"{API_URL}/research", json={
        "query": query,
        "pdf_name": "",
        "task": "WEBSEARCH"
    }, headers=headers)
    
    try:
        response_data = response.json()
        if response.status_code == 200:
            results = response_data.get("result")
            if results:
                st.markdown("### Web Search Results:")
                
                for result in results.split("\n---\n"):
                    # Separate title, snippet, and link
                    parts = result.split("\n")
                    if len(parts) >= 3:
                        title = parts[0]
                        snippet = parts[1]
                        link = parts[2]
                        
                        # Replace this line:
                        # st.markdown(f"**[{title}]({link})**")
                        # With this:
                        st.markdown(f"**[{title}]({link})**", unsafe_allow_html=True)
                        
                        # Display the snippet text
                        st.write(snippet)
                        # Add a horizontal line for separation between results
                        st.markdown("---")
            else:
                st.warning("No results found.")
        else:
            st.error(f"Error in web search: {response_data.get('detail', 'Unknown error')}")
    except requests.exceptions.JSONDecodeError:
        st.error("Received a non-JSON response from the server. Please check the server logs for more details.")

# Function for arxiv search
def arxiv_search(query):
    if not query.strip():
        st.error("Please enter a question for the Arxiv search.")
        return
    
    headers = {"Authorization": f"Bearer {st.session_state.token}"}
    response = requests.post(f"{API_URL}/research", json={
        "query": query,
        "pdf_name": "",
        "task": "ARXIV"
    }, headers=headers)
    
    try:
        response_data = response.json()
        if response.status_code == 200:
            results = response_data.get("result")
            if results:
                st.markdown("### Arxiv Search Results:")
                
                # Split results into individual entries if separated by a specific delimiter (e.g., "\n---\n")
                for result in results.split("\n---\n"):
                    title = authors = summary = link = ""
                    capturing_summary = False  # Flag to start capturing multi-line summary text
                    
                    parts = result.split("\n")
                    for part in parts:
                        if part.startswith("Title:"):
                            title = part.replace("Title: ", "").strip()
                            capturing_summary = False  # Stop capturing summary if it was active
                        elif part.startswith("Authors:"):
                            authors = part.replace("Authors: ", "").strip()
                            capturing_summary = False
                        elif part.startswith("Summary:"):
                            summary = part.replace("Summary: ", "").strip()
                            capturing_summary = True  # Start capturing summary from here
                        elif part.startswith("Link:"):
                            link = part.replace("Link: ", "").strip()
                            capturing_summary = False
                        elif capturing_summary:
                            # Append each line to summary while capturing it
                            summary += " " + part.strip()
                    
                    # Display each section if available
                    if title:
                        st.markdown(f"**Title:** {title}")
                    if authors:
                        st.markdown(f"**Authors:** {authors}")
                    if summary:
                        st.write(summary.strip())
                    if link:
                        st.markdown(f"[**Link to full article**]({link})")
                    
                    # Add a horizontal line for separation between entries
                    st.markdown("---")
            else:
                st.warning("No results found.")
        else:
            st.error(f"Error in arXiv search: {response_data.get('detail', 'Unknown error')}")
    except requests.exceptions.JSONDecodeError:
        st.error("Received a non-JSON response from the server. Please check the server logs for more details.")

def add_to_session_data(question, response_data, source):
    findings = {}
    
    if source == "WEBSEARCH":
        # Process web search results
        web_findings = []
        if isinstance(response_data, dict) and "result" in response_data:
            results = response_data["result"].split("\n---\n")
            for result in results:
                parts = result.split("\n")
                if len(parts) >= 3:
                    formatted_result = f"Title: {parts[0]}\nSnippet: {parts[1]}\nLink: {parts[2]}"
                    web_findings.append(formatted_result)
        findings["Web Search Agent"] = web_findings

    elif source == "ARXIV":
        # Process Arxiv results
        arxiv_findings = []
        if isinstance(response_data, dict) and "result" in response_data:
            results = response_data["result"].split("\n---\n")
            for result in results:
                arxiv_findings.append(result)
        findings["Arxiv Agent"] = arxiv_findings

    elif source == "RAG":
        # Process RAG results
        findings["RAG Agent"] = [response_data.get("answer", "")]

    st.session_state.session_data.append({
        "question": question,
        "findings": findings,
        "answer": response_data.get("answer", "")
    })

def generate_report_from_session():
    if not st.session_state.session_data:
        st.error("No research data available to generate report.")
        return

    headers = {"Authorization": f"Bearer {st.session_state.token}"}
    
    # Format sessions data
    formatted_sessions = [
        {
            "question": session["question"],
            "findings": session["findings"],
            "answer": session.get("answer", "")
        }
        for session in st.session_state.session_data
    ]
    
    payload = {"sessions": formatted_sessions}
    
    try:
        response = requests.post(
            f"{API_URL}/generate_dynamic_reports",
            json=payload,
            headers=headers
        )
        
        if response.status_code == 200:
            pdf_data = response.content
            
            # Create a unique key for the download button
            button_key = f"download_pdf_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Use st.download_button with custom CSS class
            st.markdown(
                f"""
                <style>
                    div.stDownloadButton > button {{
                        color: white !important;
                        background-color: #444444 !important;
                        border: 1px solid white !important;
                        font-size: 18px !important;
                        font-weight: bold !important;
                    }}
                    div.stDownloadButton > button:hover {{
                        color: white !important;
                        border-color: white !important;
                    }}
                </style>
                """,
                unsafe_allow_html=True
            )
            
            st.download_button(
                label="Download Research Report (PDF)",
                data=pdf_data,
                file_name=f"research_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf",
                key=button_key
            )
            
            st.success("PDF report generated successfully!")
        else:
            st.error(f"Error generating report: {response.json().get('detail', 'Unknown error')}")
    except Exception as e:
        st.error(f"Error generating report: {str(e)}")        
def open_in_codelabs():
    # Check if button already exists in session state
    if 'codelabs_button_clicked' not in st.session_state:
        st.session_state.codelabs_button_clicked = False

    if not st.session_state.session_data:
        st.error("No research data available to generate Codelabs format.")
        return
    
    try:
        if not st.session_state.codelabs_button_clicked:
            payload = {
                "sessions": st.session_state.session_data
            }
            
            response = requests.post(
                f"{API_URL}/generate_codelabs",
                json=payload,
                headers={"Authorization": f"Bearer {st.session_state.token}"}
            )
            
            if response.status_code == 200:
                data = response.json()
                codelabs_content = data['codelabs_content']
                
                html_content = f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Codelabs Preview</title>
                    <script>
                        window.location.href = 'https://codelabs-preview.appspot.com/?markdown=' + encodeURIComponent(`{codelabs_content}`);
                    </script>
                </head>
                <body>
                    <p>Redirecting to Codelabs...</p>
                </body>
                </html>
                """
                
                # Create download button with unique key
                st.download_button(
                    label="Open in Codelabs",
                    data=html_content,
                    file_name="codelabs_preview.html",
                    mime="text/html",
                    key="unique_codelabs_button",
                    on_click=lambda: setattr(st.session_state, 'codelabs_button_clicked', True)
                )
    except Exception as e:
        st.error(f"Error: {str(e)}")

def main() -> None:
    try:
        # Add logout button at the top if user is logged in
        if st.session_state.get('logged_in', False):
            col1, col2 = st.columns([0.9, 0.1])  # Adjust ratio as needed
            with col2:
                if st.button("Logout"):
                    logout()
                    st.rerun()
            with col1:
                st.write(f"Hello, **{st.session_state.get('username', 'User')}!**")
        
        if not st.session_state.logged_in:
            st.subheader("Login or Signup")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            if st.button("Login"):
                login_signup(username, password, signup=False)
            if st.button("Signup"):
                login_signup(username, password, signup=True)
        else:
            # Dropdown with user-friendly options
            task_display = st.selectbox("Choose a task", ["RAG", "Web Search", "Arxiv Agent"])
            question = st.text_input("Enter your query")
            task_mapping = {"Web Search": "WEBSEARCH", "RAG": "RAG", "Arxiv Agent": "ARXIV"}
            task = task_mapping.get(task_display)

            selected_pdf = None
            if task == "RAG":
                selected_pdf = st.selectbox("Choose a PDF", get_pdf_list())
                if st.button("Preview PDF"):
                    preview_pdf(selected_pdf)

            # Task-specific query submission
            if task == "RAG" and st.button("Submit Question", key="submit_rag_button"):
                response_data = ask_question(selected_pdf, question)
                if response_data:
                    st.write(f"**Answer:** {response_data.get('answer', '')}")
                    add_to_session_data(question, response_data, "RAG")
                else:
                    st.error("Unexpected response format from the server.")

            elif task == "WEBSEARCH" and st.button("Submit Query", key="submit_websearch_button"):
                response = requests.post(
                    f"{API_URL}/research",
                    json={"query": question, "pdf_name": "", "task": "WEBSEARCH"},
                    headers={"Authorization": f"Bearer {st.session_state.token}"}
                )
                
                if response.status_code == 200:
                    response_data = response.json()
                    web_search(question)
                    add_to_session_data(question, response_data, "WEBSEARCH")

            elif task == "ARXIV" and st.button("Submit Query", key="submit_arxiv_button"):
                response = requests.post(
                    f"{API_URL}/research",
                    json={"query": question, "pdf_name": "", "task": "ARXIV"},
                    headers={"Authorization": f"Bearer {st.session_state.token}"}
                )
                
                if response.status_code == 200:
                    response_data = response.json()
                    arxiv_search(question)
                    add_to_session_data(question, response_data, "ARXIV")
            
            # Report Generation Buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Generate PDF Report"):
                    if not st.session_state.session_data:
                        st.error("No session data to generate a report.")
                    else:
                        generate_report_from_session()
            with col2:
                if st.button("Open in Codelabs"):
                    if not st.session_state.session_data:
                        st.error("No session data to generate a report.")
                    else:
                        open_in_codelabs()

    except Exception as e:
        st.error("Error running the main application.")
        print(f"Error: {e}")

# Run the main app function
if __name__ == "__main__":
    main()
