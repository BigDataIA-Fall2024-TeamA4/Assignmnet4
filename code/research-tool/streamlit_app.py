import streamlit as st
import requests
from datetime import datetime
from base64 import b64encode

# Initialize session state
if 'access_token' not in st.session_state:
    st.session_state.access_token = None
if 'research_session' not in st.session_state:
    st.session_state.research_session = None
if 'qa_history' not in st.session_state:
    st.session_state.qa_history = []

BACKEND_URL = "http://localhost:8000"

def format_qa_for_codelab(qa_history):
    """Format Q&A history into Codelab-friendly structure"""
    codelab_content = {
        "title": f"Research Report {datetime.now().strftime('%Y-%m-%d')}",
        "steps": []
    }
    
    for i, (question, result) in enumerate(qa_history, 1):
        # Format findings as text
        findings_text = ""
        for agent, findings in result["agent_findings"].items():
            findings_text += f"\n### {agent}\n"
            for finding in findings:
                findings_text += f"- {finding['content']}\n"
        
        step = {
            "title": f"Research Question {i}",
            "content": f"""
## Question
{question}

## Findings
{findings_text}

## Answer
{result["answer"]}
"""
        }
        codelab_content["steps"].append(step)
    
    return codelab_content

def login(username, password):
    try:
        response = requests.post(
            f"{BACKEND_URL}/token",
            data={"username": username, "password": password}
        )
        if response.status_code == 200:
            return response.json().get("access_token")
        return None
    except Exception as e:
        st.error(f"Login failed: {str(e)}")
        return None

def get_pdf_list(token):
    try:
        response = requests.get(
            f"{BACKEND_URL}/pdfs",
            headers={"Authorization": f"Bearer {token}"}
        )
        return response.json()
    except Exception as e:
        st.error(f"Failed to get PDFs: {str(e)}")
        return []

def main():
    st.title("Research Tool Demo")

    # Login section
    if not st.session_state.access_token:
        st.subheader("Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            token = login(username, password)
            if token:
                st.session_state.access_token = token
                st.success("Logged in successfully!")
                st.rerun()

    # Main interface
    else:
        # PDF selection
        pdfs = get_pdf_list(st.session_state.access_token)
        selected_pdf = st.selectbox("Select a document", pdfs)

        # Question interface
        if selected_pdf:
            st.subheader("Ask a Question")
            question = st.text_input("Enter your research question:")
            
            if st.button("Submit Question"):
                try:
                    response = requests.post(
                        f"{BACKEND_URL}/research/question",
                        headers={"Authorization": f"Bearer {st.session_state.access_token}"},
                        json={"document_id": selected_pdf, "question": question}
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.session_state.qa_history.append((question, result))
                        
                        # Display latest result
                        st.subheader("Research Findings")
                        for agent, findings in result["agent_findings"].items():
                            st.write(f"**{agent}:**")
                            for finding in findings:
                                st.write(f"- {finding['content']}")
                        
                        st.subheader("Answer")
                        st.write(result["answer"])

                except Exception as e:
                    st.error(f"Error: {str(e)}")

            # Export options
            if st.session_state.qa_history:
                st.subheader("Export Options")
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("Export as PDF"):
                        try:
                            response = requests.post(
                                f"{BACKEND_URL}/export/pdf",
                                headers={"Authorization": f"Bearer {st.session_state.access_token}"},
                                json={"qa_history": st.session_state.qa_history}
                            )
                            st.download_button(
                                "Download PDF",
                                response.content,
                                file_name=f"research_report_{datetime.now().strftime('%Y%m%d')}.pdf",
                                mime="application/pdf"
                            )
                        except Exception as e:
                            st.error(f"Error exporting PDF: {str(e)}")
                
                with col2:
                    if st.button("View as Codelab"):
                        try:
                            formatted_content = format_qa_for_codelab(st.session_state.qa_history)
                            response = requests.post(
                                f"{BACKEND_URL}/export/codelab",
                                headers={"Authorization": f"Bearer {st.session_state.access_token}"},
                                json=formatted_content
                            )
                            
                            # Create a temporary HTML file and open it in a new tab
                            html_content = response.text
                            
                            # Using HTML to create a link that opens in a new tab
                            st.markdown(
                                f'<a href="data:text/html;base64,{b64encode(html_content.encode()).decode()}" target="_blank" '
                                f'style="display: inline-block; padding: 0.5em 1em; color: white; background-color: #FF4B4B; '
                                f'border-radius: 5px; text-decoration: none;">Open Codelab in New Tab</a>', 
                                unsafe_allow_html=True
                            )
                        except Exception as e:
                            st.error(f"Error generating Codelab: {str(e)}")

if __name__ == "__main__":
    main()