from dotenv import load_dotenv
import os
from langchain_deepseek.chat_models import ChatDeepSeek
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SimpleSequentialChain
from langchain_core.runnables import RunnableLambda
from langchain.schema import HumanMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS


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


# Define a prompt template
prompt = PromptTemplate(
    input_variables=["topic"],
    template="Tell me a fun fact about {topic}."
)

# Define a runnable chain manually (instead of LLMChain)
def chain_fn(input_data):
    # Format prompt
    final_prompt = prompt.format(topic=input_data)
    # Wrap it in a HumanMessage and send to DeepSeek
    return llm.invoke([HumanMessage(content=final_prompt)])

# Wrap in a RunnableLambda chain
chain = RunnableLambda(chain_fn)

# Run the chain
result = chain.invoke("space travel")
print("🛰️ Fun fact:", result.content)


#Sequential Chains Example
print("\n--- Sequential Chains Example ---")

# Step 1: Prompt to generate a title
title_prompt = PromptTemplate(
    input_variables=["topic"],
    template="Write a catchy blog title about {topic}."
)
title_chain = LLMChain(llm=llm, prompt=title_prompt)

# Step 2: Prompt to generate intro from the title
intro_prompt = PromptTemplate(
    input_variables=["title"],
    template="Write a blog intro based on this title: {title}"
)
intro_chain = LLMChain(llm=llm, prompt=intro_prompt)

# Sequential chain: topic → title → intro
blog_chain = SimpleSequentialChain(chains=[title_chain, intro_chain], verbose=True)

# Run the chain
result = blog_chain.run("space travel for kids")

print("\n📝 Final Blog Intro:\n", result)


# ✅ Load your PDF file
loader = PyPDFLoader("sample.pdf")  # replace with your PDF
documents = loader.load()

# ✅ Split into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(documents)


# ✅ Create embeddings (using HuggingFace)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ✅ Create vector store
vectorstore = FAISS.from_documents(docs, embeddings)

# ✅ Build Retrieval-QA Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    return_source_documents=True
)

# ✅ Ask a question
query = "What are the key findings mentioned in the document?"
result = qa_chain({"query": query})

print("🧠 Answer:", result["result"])
print("\n📄 Sources:")
for doc in result["source_documents"]:
    print("-", doc.metadata["source"])