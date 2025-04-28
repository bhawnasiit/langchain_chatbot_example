from dotenv import load_dotenv
import os
import math
import streamlit as st
import tempfile
from datetime import datetime
from langchain_deepseek.chat_models import ChatDeepSeek
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SimpleSequentialChain
from langchain_core.runnables import RunnableLambda
from langchain.schema import HumanMessage
#from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
#from langchain_community.vectorstores import Chroma
from langchain.agents import initialize_agent, AgentType
from langchain.tools import tool
from langchain.memory import ConversationBufferMemory
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

class LocalHuggingFaceEmbeddings(Embeddings):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return self.model.encode(texts, convert_to_tensor=False).tolist()

    def embed_query(self, text):
        return self.model.encode(text, convert_to_tensor=False).tolist()

# Load environment variables
#load_dotenv()
# Instead of dotenv:
api_key = st.secrets["DEEPSEEK_API_KEY"] #os.getenv()
print("API key loaded:", api_key)  # Debug line
#os.environ["DEEPSEEK_API_KEY"] = api_key
if api_key is None:
    raise ValueError("API key not found!")



llm = ChatDeepSeek(
    model="deepseek-chat",
    openai_api_key=api_key,             
    max_tokens=100,
    temperature=0,
    openai_api_base="https://api.deepseek.com"  
)
# Check if the LLM is initialized correctly
if llm is None:
    raise ValueError("Failed to initialize the ChatDeepSeek model. Please check your API key and configuration.")

print("API KEY:", api_key[:6] + "..." if api_key else "None")

# --- UI ---
st.set_page_config(page_title="🧠 DeepSeek File QA Chatbot")
st.title("🧠 DeepSeek + LangChain Agent Chatbot")
st.write("Upload a PDF and start chatting with it! 🚀")

# --- File Upload ---
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

# If a file is uploaded
if uploaded_file:
    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    # Load documents
    loader = PyPDFLoader(tmp_path)
    docs = loader.load()

    # Split
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    # Embeddings
    #embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    embeddings = LocalHuggingFaceEmbeddings()

    # Vector store
    from langchain_community.vectorstores import FAISS

    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings.embed_query 
    )
    retriever = vectorstore.as_retriever()

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=False
    )

    # Define the square root tool
    @tool
    def calculate_square_root(x: str) -> str:
        """Calculate the square root of a number given as a string."""
        try:
            number = float(x)
            return f"The square root of {number} is {math.sqrt(number)}"
        except ValueError:
            return "Please provide a valid number."

    @tool
    def get_current_time(input_text: str = "") -> str:
        """Returns the current date and time in human-readable format."""
        return datetime.now().strftime("%A, %d %B %Y %I:%M %p")

    @tool
    def search_docs(query: str) -> str:
        """Answer questions from uploaded documents using retrieval."""
        return qa_chain.run(query)

    # Register the tool (already a Tool object via the @tool decorator)
    tools = [calculate_square_root, get_current_time, search_docs]

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    # Initialize the agent
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION, #ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        memory=memory
    )


# --- Chat Interface ---
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.text_input("Ask a question about the document or use tools:", key="user_input")
    if user_input:
        try:
            with st.spinner("Thinking..."):
                response = agent.run(user_input)
                st.session_state.chat_history.append(("You", user_input))
                st.session_state.chat_history.append(("Agent", response))
        except Exception as e:
            st.error(f"⚠️ Error: {e}")

    # Display chat history
    for speaker, msg in st.session_state.chat_history:
        if speaker == "You":
            st.markdown(f"**🧑‍💻 {speaker}:** {msg}")
        else:
            st.markdown(f"**🤖 {speaker}:** {msg}")

else:
    st.info("📄 Please upload a PDF to begin.")




