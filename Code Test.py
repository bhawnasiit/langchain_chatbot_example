from dotenv import load_dotenv
import os
from langchain_deepseek.chat_models import ChatDeepSeek
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SimpleSequentialChain
from langchain.chains import ConversationChain
from langchain_core.runnables import RunnableLambda
from langchain.schema import HumanMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, load_tools, AgentType
from langchain.agents import Tool
import math
from langchain.tools import tool



# Load environment variables
load_dotenv()
api_key = os.getenv("DEEPSEEK_API_KEY")
print("API key loaded:", api_key)  # Debug line
os.environ["DEEPSEEK_API_KEY"] = api_key



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

# Math tool
from langchain.tools import tool
import math

@tool
def calculate_square_root(x: str) -> str:
    """Calculate the square root of a number given as a string."""
    import math
    try:
        number = float(x)
        return f"The square root of {number} is {math.sqrt(number)}"
    except ValueError:
        return "Please provide a valid number."

tools = [calculate_square_root]  # it's already a Tool


# Initialize the agent
agent = initialize_agent(
    tools=tool,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)
response = agent.run("What is the square root of 144?")
print("Agent:", response)