import streamlit as st
from backend import DocumentHandler
from rag import RetrievalAndGeneration
import os
import shutil

st.set_page_config(layout="wide")

# State initialization
if 'context' not in st.session_state:
    st.session_state.context = []
if 'is_rag' not in st.session_state:
    st.session_state.is_rag = False
if 'is_socratic' not in st.session_state:
    st.session_state.is_socratic = False  # Default to Socratic mode being off

# Sidebar for History and Settings
st.sidebar.title("Options")
option = st.sidebar.radio("Go to", ["History", "Settings"])

if option == "History":
    st.sidebar.subheader("Chat History")
    
    # Button to clear chat history at the top
    if st.sidebar.button("Clear Chat History"):
        st.session_state.context = []
        st.experimental_rerun()  # Automatically rerun the code to update the history
    
    if 'context' in st.session_state:
        for msg in st.session_state.context:
            st.sidebar.write(f"{msg['role']}: {msg['content']}")
    else:
        st.sidebar.write("No chat history available.")

if option == "Settings":
    st.sidebar.subheader("Settings")
    with st.sidebar.expander("Patch Notes"):
        st.markdown("""
            **Patch Notes:**
            - Initial release with basic chat functionality.
            - Added file upload feature for document-based Q&A.
            - Improved caching and session management.
        """)

    with st.sidebar.expander("Support"):
        st.markdown("""
            For support, please contact us at:
            **engenieai.query@gmail.com**
        """)
        
    st.sidebar.write("New exciting things are to come soon.")

# Include custom font CSS
st.markdown("""
    <style>
    @font-face {
        font-family: '04b_30';
        src: url('file:///C:/Users/Dipto/Downloads/04B_30__.TTF') format('truetype');
    }
    .custom-font {
        font-family: '04b_30';
        font-size: 48px;
    }
    .custom-header {
        font-family: '04b_30';
        font-size: 48px;
    }
    </style>
    """, unsafe_allow_html=True)

# Use the custom font for the bot name
st.markdown('<h1 class="custom-font">ENGenieAI</h1>', unsafe_allow_html=True)
st.write("""Alright, future engineers, here’s the lowdown on how to navigate our friendly AI tutor, ENGenieAI. Let’s make sure we’re getting those answers in the most efficient way possible!""")

# Display rules, how to use, etc.
expander = st.expander("How to use?")
expander.write("""
### RAG (Post-File Upload)
1. **Click 'Upload Files'**: Get started by clicking that upload button.
2. **Choose Your File**: Select the file that holds the secrets you seek.
3. **Click 'Document Embeddings'**: Hit this button to make the magic happen.
4. **Ask Away!**: Type your question in the text box and prepare for enlightenment.

### Chat Mode
1. **Type Your Prompt**: Enter your query in the text box.
2. **Toggle Socratic Mode**: Use the checkbox to turn Socratic mode on or off.
    - **Socratic Mode On**: Our tutor will engage you with thought-provoking questions based on your input.
    - **Socratic Mode Off**: The bot will give you direct answers without engaging in the Socratic method.
""")

# Toggle Socratic mode
st.session_state.is_socratic = st.toggle("Enable Socratic Mode", value=st.session_state.is_socratic)
if st.session_state.is_socratic:
    st.write("Socratic Chatbot Mode ON!")

uploaded_files = st.file_uploader("Upload Files", type=["pdf"], accept_multiple_files=True)

prompt1 = st.chat_input("Search Box")

# Load environment variables
groq_api_key = os.getenv("GROQ_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")

# Initialize DocumentHandler and RAG
doc_handler = DocumentHandler(groq_api_key, google_api_key)
llm = doc_handler.load_llm()
embeddings = doc_handler.create_embeddings()
rag = RetrievalAndGeneration(llm)

# Cache the document loading process using st.cache_data
@st.cache_data
def cached_load_documents(_uploaded_files):
    return doc_handler.load_documents(_uploaded_files)

# Cache the vector store creation process using st.cache_resource
@st.cache_resource
def cached_create_vector_store(_documents):
    return doc_handler.create_vector_store(_documents, embeddings)

def vector_embedding(uploaded_files):
    # Clear previous session state and caches
    st.session_state.clear()
    st.cache_data.clear()
    st.cache_resource.clear()
    
    # Process new uploaded files
    if uploaded_files:
        st.session_state.docs = cached_load_documents(uploaded_files)
        st.session_state.vectors = cached_create_vector_store(st.session_state.docs)
        st.session_state.is_rag = True
        st.write("Vector Store DB Is Ready")
        print("Vector store is ready")

def clear_all():
    # Clear all session state and caches
    st.session_state.clear()
    st.cache_data.clear()
    st.cache_resource.clear()
    st.session_state.context = []
    st.session_state.is_rag = False

# Function to handle folder deletion and cache clearing
def delete_uploaded_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
        st.cache_data.clear()
        st.cache_resource.clear()
        st.write(f"Deleted folder: {folder_path} and cleared cache.")
        print(f"Deleted folder: {folder_path} and cleared cache.")

if st.button("Documents Embedding") and uploaded_files:
    vector_embedding(uploaded_files)

if st.button("Delete Uploaded Files"):
    delete_uploaded_folder("temp")
    clear_all()

if prompt1:
    if st.session_state.is_rag and "vectors" in st.session_state:
        response, context = rag.create_retrieval_response(prompt1, st.session_state.vectors)
        st.session_state.context.append({'role': 'user', 'content': prompt1})
        st.session_state.context.append({'role': 'assistant', 'content': response})
        st.write(response)
    else:
        if st.session_state.is_socratic:
            response = doc_handler.socratic_chat(prompt1, st.session_state.context)
        else:
            response = doc_handler.normal_chat(prompt1, st.session_state.context)
        st.session_state.context.append({'role': 'user', 'content': prompt1})
        st.session_state.context.append({'role': 'assistant', 'content': response})
        st.write(response)


