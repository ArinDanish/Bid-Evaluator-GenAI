import streamlit as st
import os
import tempfile
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
import google.generativeai as genai
from dotenv import load_dotenv
import datetime

# Load environment variables
load_dotenv()

# Configure Google Generative AI
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel('gemini-pro')

def init_session_state():
    """Initialize session state variables"""
    if 'rfp_db' not in st.session_state:
        st.session_state.rfp_db = None
    if 'bid_db' not in st.session_state:
        st.session_state.bid_db = None
    if 'rfp_text' not in st.session_state:
        st.session_state.rfp_text = None
    if 'bid_text' not in st.session_state:
        st.session_state.bid_text = None
    if 'qa_history' not in st.session_state:
        st.session_state.qa_history = []
    if 'evaluation_done' not in st.session_state:
        st.session_state.evaluation_done = False

def reset_session():
    """Reset all session state variables"""
    st.session_state.rfp_db = None
    st.session_state.bid_db = None
    st.session_state.rfp_text = None
    st.session_state.bid_text = None
    st.session_state.evaluation_done = False

def parse_pdf(file):
    """Extract text from PDF"""
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def create_chunks(text):
    """Split text into chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=2000,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = text_splitter.split_text(text)
    return chunks

def create_vector_store(chunks):
    """Create vector store from text chunks"""
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    vector_store = FAISS.from_texts(chunks, embeddings)
    return vector_store

def get_relevant_context(vector_store, query):
    """Get relevant context from vector store"""
    results = vector_store.similarity_search(query, k=3)
    return "\n\n".join([doc.page_content for doc in results])

def get_llm_response(vector_store, query):
    """Get response from Gemini Pro"""
    context = get_relevant_context(vector_store, query)
    
    prompt = f"""Based on the following context from the document, please answer the question.
    
    Context:
    {context}
    
    Question: {query}
    
    Please provide a clear and detailed answer based on the information in the context."""
    
    response = model.generate_content(prompt)
    return response.text

def evaluate_bid(bid_text, rfp_db):
    """Evaluate bid against RFP requirements"""
    # Get relevant RFP context
    rfp_context = get_relevant_context(rfp_db, "technical requirements financial requirements")
    
    technical_prompt = f"""Using the following RFP requirements, evaluate the technical qualifications of this bid:

    RFP Context:
    {rfp_context}

    Bid Text:
    {bid_text}

    Please evaluate the following aspects:
    1. Technical compliance with specifications
    2. Experience and past performance
    3. Methodology and approach
    4. Team qualifications

    Provide a detailed analysis with specific references from both documents."""
    
    financial_prompt = f"""Using the following RFP requirements, evaluate the financial aspects of this bid:

    RFP Context:
    {rfp_context}

    Bid Text:
    {bid_text}

    Please evaluate the following aspects:
    1. Cost structure and breakdown
    2. Financial stability
    3. Price competitiveness
    4. Payment terms and conditions

    Provide a detailed analysis with specific references from both documents."""
    
    technical_response = model.generate_content(technical_prompt)
    financial_response = model.generate_content(financial_prompt)
    
    return technical_response.text, financial_response.text

def main():
    st.set_page_config(page_title="Bid Evaluator", layout="wide")
    
    # Add custom CSS for history pane styling
    st.markdown("""
        <style>
        .history-pane {
            background-color: #0747a6;
            padding: 1rem;
            border-radius: 10px;
            color: white;
        }
        .history-item {
            background-color: rgba(255, 255, 255, 0.1);
            padding: 0.5rem;
            border-radius: 5px;
            margin-bottom: 0.5rem;
        }
        .stExpander {
            background-color: transparent !important;
            border: 1px solid rgba(255, 255, 255, 0.2) !important;
        }
        .stExpander > div:first-child {
            color: white !important;
        }
        /* Make evaluation results full width */
        .evaluation-results {
            width: 100%;
        }
        .evaluation-results .stMarkdown {
            width: 100%;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    init_session_state()
    
    # Add Reset button at the top right
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        st.title("Bid Evaluator")
    with col3:
        if st.button("Start New Evaluation", type="primary"):
            reset_session()
            st.rerun()
    
    # Create two columns - left for history, right for main content
    left_column, right_column = st.columns([1, 3])
    
    # Left column - Question History with blue background
    with left_column:
        st.markdown('<div class="history-pane">', unsafe_allow_html=True)
        st.header("Question History")
        if st.session_state.qa_history:
            for i, (timestamp, q, a) in enumerate(reversed(st.session_state.qa_history)):
                st.markdown(f'<div class="history-item">', unsafe_allow_html=True)
                with st.expander(f"Q: {q[:50]}...", expanded=False):
                    st.markdown(f"<p style='color: #ffffff80;'>Time: {timestamp}</p>", unsafe_allow_html=True)
                    st.markdown(f"<p style='color: white;'><strong>Question:</strong> {q}</p>", unsafe_allow_html=True)
                    st.markdown(f"<p style='color: white;'><strong>Answer:</strong> {a}</p>", unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Right column - Main Content
    with right_column:
        # Status Indicator
        if st.session_state.rfp_db and st.session_state.bid_db:
            st.success("Both documents are loaded and ready for analysis")
        elif st.session_state.rfp_db:
            st.info("RFP loaded. Please upload bid document.")
        else:
            st.info("Please start by uploading an RFP document.")

        st.header("Document Upload")
        
        # RFP Upload
        rfp_file = st.file_uploader("Upload RFP Document", type=['pdf'], key="rfp_uploader")
        if rfp_file and not st.session_state.rfp_db:
            with st.spinner("Processing RFP..."):
                with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                    tmp_file.write(rfp_file.getvalue())
                    tmp_file_path = tmp_file.name
                
                st.session_state.rfp_text = parse_pdf(tmp_file_path)
                chunks = create_chunks(st.session_state.rfp_text)
                st.session_state.rfp_db = create_vector_store(chunks)
                os.unlink(tmp_file_path)
                st.success("RFP processed successfully!")

        # Bid Upload
        bid_file = st.file_uploader("Upload Bid Document", type=['pdf'], key="bid_uploader")
        if bid_file and not st.session_state.bid_db:
            with st.spinner("Processing Bid..."):
                with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                    tmp_file.write(bid_file.getvalue())
                    tmp_file_path = tmp_file.name
                
                st.session_state.bid_text = parse_pdf(tmp_file_path)
                chunks = create_chunks(st.session_state.bid_text)
                st.session_state.bid_db = create_vector_store(chunks)
                os.unlink(tmp_file_path)
                st.success("Bid processed successfully!")

        # Evaluation Button and Results
        if st.session_state.rfp_db and st.session_state.bid_db:
            if st.button("Evaluate Bid", type="primary"):
                with st.spinner("Evaluating bid..."):
                    technical_eval, financial_eval = evaluate_bid(st.session_state.bid_text, st.session_state.rfp_db)
                    st.session_state.evaluation_done = True
                    
                    # Display results in full width
                    st.markdown('<div class="evaluation-results">', unsafe_allow_html=True)
                    st.header("Evaluation Results")
                    
                    st.markdown("### Technical Evaluation")
                    st.markdown(f"<div style='background-color: white; padding: 20px; border-radius: 5px; margin-bottom: 20px;'>{technical_eval}</div>", unsafe_allow_html=True)
                    
                    st.markdown("### Financial Evaluation")
                    st.markdown(f"<div style='background-color: white; padding: 20px; border-radius: 5px; margin-bottom: 20px;'>{financial_eval}</div>", unsafe_allow_html=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    st.success("Evaluation completed!")

            # Question Asking Section
            if st.session_state.bid_db:
                st.header("Ask Questions About the Bid")
                user_question = st.text_input("Enter your question about the bid document:")
                if st.button("Ask Question"):
                    if user_question:
                        with st.spinner("Getting answer..."):
                            answer = get_llm_response(st.session_state.bid_db, user_question)
                            
                            # Add to history
                            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            st.session_state.qa_history.append((timestamp, user_question, answer))
                            
                            # Display current answer
                            st.subheader("Answer:")
                            st.markdown(f"<div style='background-color: white; padding: 20px; border-radius: 5px; margin-bottom: 20px;'>{answer}</div>", unsafe_allow_html=True)
                    else:
                        st.warning("Please enter a question.")

if __name__ == "__main__":
    main()