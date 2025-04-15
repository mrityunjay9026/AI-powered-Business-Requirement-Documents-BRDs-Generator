import streamlit as st
import requests
import json
import base64
from datetime import datetime
from backend import *

# Configure the Streamlit page
st.set_page_config(
    page_title="AI-Powered BRD Generator",
    page_icon="📄",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 20px;
    }
    .section-header {
        font-size: 1.5rem;
        color: #1E3A8A;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    .info-text {
        font-size: 1rem;
        color: #4B5563;
    }
    .stButton>button {
        background-color: #1E3A8A;
        color: white;
        font-weight: bold;
        padding: 10px 20px;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1 class='main-header'>AI-Powered Business Requirements Document Generator</h1>", unsafe_allow_html=True)
st.markdown("<p class='info-text'>Generate comprehensive Business Requirement Documents using Llama3 AI</p>", unsafe_allow_html=True)

# Sidebar for customization options
with st.sidebar:
    st.markdown("## BRD Configuration")
    
    document_type = st.selectbox(
        "Document Type",
        ["Standard BRD", "Technical BRD", "Product BRD", "Market BRD"]
    )
    
    template_style = st.selectbox(
        "Template Style",
        ["Detailed", "Concise", "Visual", "Executive"]
    )
    
    output_format = st.selectbox(
        "Output Format",
        ["PDF", "Word", "Markdown", "HTML"]
    )
    
    st.markdown("## Advanced Settings")
    
    temperature = st.slider(
        "Creativity Level",
        min_value=0.1,
        max_value=1.0,
        value=0.7,
        step=0.1,
        help="Higher values make output more creative, lower values more deterministic"
    )
    
    max_tokens = st.slider(
        "Maximum Length",
        min_value=1000,
        max_value=10000,
        value=4000,
        step=500,
        help="Maximum number of tokens in the generated document"
    )

# Main form
st.markdown("<h2 class='section-header'>Project Information</h2>", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    project_name = st.text_input("Project Name")
    project_owner = st.text_input("Project Owner/Sponsor")
    department = st.text_input("Department")

with col2:
    start_date = st.date_input("Project Start Date")
    end_date = st.date_input("Expected Completion Date")
    priority = st.selectbox("Priority", ["Low", "Medium", "High", "Critical"])

st.markdown("<h2 class='section-header'>Project Description</h2>", unsafe_allow_html=True)
project_description = st.text_area("Describe your project and its objectives", height=150)

st.markdown("<h2 class='section-header'>Requirements Details</h2>", unsafe_allow_html=True)

# Business requirements
with st.expander("Business Requirements", expanded=True):
    business_requirements = st.text_area(
        "Describe the business needs and goals this project should address",
        height=120,
        help="What business problems are you trying to solve?"
    )

# User requirements
with st.expander("User Requirements", expanded=True):
    user_requirements = st.text_area(
        "Describe what users need to be able to do",
        height=120,
        help="Focus on functionality from the user's perspective"
    )

# Functional requirements
with st.expander("Functional Requirements", expanded=False):
    functional_requirements = st.text_area(
        "Describe specific functions the system should perform",
        height=120,
        help="Technical capabilities the system needs to have"
    )

# Non-functional requirements
with st.expander("Non-Functional Requirements", expanded=False):
    non_functional_requirements = st.text_area(
        "Describe performance, security, usability requirements",
        height=120,
        help="Quality attributes like response time, security levels, etc."
    )

# Constraints
with st.expander("Constraints", expanded=False):
    constraints = st.text_area(
        "Any limitations or restrictions that need to be considered",
        height=100,
        help="Budget, technical, time, or resource constraints"
    )

# Stakeholders
with st.expander("Stakeholders", expanded=False):
    stakeholders = st.text_area(
        "List key stakeholders and their roles/interests",
        height=100
    )

# Generation button
if st.button("Generate BRD", type="primary"):
    if not project_name or not project_description:
        st.error("Project name and description are required!")
    else:
        with st.spinner("Generating your Business Requirements Document..."):
            # Prepare data to send to backend
            request_data = {
                "project_name": project_name,
                "project_owner": project_owner,
                "department": department,
                "start_date": start_date.strftime("%Y-%m-%d"),
                "end_date": end_date.strftime("%Y-%m-%d"),
                "priority": priority,
                "project_description": project_description,
                "business_requirements": business_requirements,
                "user_requirements": user_requirements,
                "functional_requirements": functional_requirements,
                "non_functional_requirements": non_functional_requirements,
                "constraints": constraints,
                "stakeholders": stakeholders,
                "document_type": document_type,
                "template_style": template_style,
                "output_format": output_format,
                "settings": {
                    "temperature": temperature,
                    "max_tokens": max_tokens
                }
            }
            
            try:
                # Send request to backend
                response = requests.post(
                    "http://localhost:8000/generate_brd",
                    json=request_data
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Display success message
                    st.success("BRD generated successfully!")
                    
                    # Display the document preview
                    st.markdown("<h2 class='section-header'>BRD Preview</h2>", unsafe_allow_html=True)
                    
                    # Show different preview based on format
                    if output_format == "Markdown":
                        st.markdown(result["content"])
                    elif output_format == "HTML":
                        st.components.v1.html(result["content"], height=500)
                    else:
                        st.markdown("Document ready for download")
                    
                    # Provide download option
                    download_filename = f"BRD_{project_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}"
                    
                    if output_format == "PDF":
                        download_filename += ".pdf"
                        mime_type = "application/pdf"
                    elif output_format == "Word":
                        download_filename += ".docx"
                        mime_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                    elif output_format == "Markdown":
                        download_filename += ".md"
                        mime_type = "text/markdown"
                    else:  # HTML
                        download_filename += ".html"
                        mime_type = "text/html"
                    
                    st.download_button(
                        label=f"Download {output_format} Document",
                        data=base64.b64decode(result["document_base64"]),
                        file_name=download_filename,
                        mime=mime_type
                    )
                else:
                    st.error(f"Error: {response.status_code} - {response.text}")
            except Exception as e:
                st.error(f"Failed to connect to backend service: {str(e)}")

# Footer
st.markdown("---")
st.markdown("<p class='info-text'>Powered by Llama3 and LangChain</p>", unsafe_allow_html=True)