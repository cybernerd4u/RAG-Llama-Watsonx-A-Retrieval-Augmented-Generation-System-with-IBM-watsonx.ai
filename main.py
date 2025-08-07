import os
from dotenv import load_dotenv
from langchain_ibm import WatsonxLLM, WatsonxEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load environment variables from .env file
load_dotenv(dotenv_path=r"D:\ML Projects\rag-llama-watsonx\.env")

# Debug: Print environment variables to verify
print("API Key:", os.getenv("WATSONX_APIKEY"))
print("Project ID:", os.getenv("WATSONX_PROJECT_ID"))
print("URL:", os.getenv("WATSONX_URL"))
print("User Agent:", os.getenv("USER_AGENT"))

# Retrieve IBM watsonx.ai credentials
watsonx_api_key = os.getenv("WATSONX_APIKEY")
watsonx_project_id = os.getenv("WATSONX_PROJECT_ID")
watsonx_url = os.getenv("WATSONX_URL")

if not all([watsonx_api_key, watsonx_project_id, watsonx_url]):
    raise ValueError("Missing required environment variables: WATSONX_APIKEY, WATSONX_PROJECT_ID, or WATSONX_URL")

# Initialize WatsonxLLM with Llama 3 405B
parameters = {
    "decoding_method": "greedy",
    "max_new_tokens": 500,
    "min_new_tokens": 1,
    "temperature": 0.7,
    "top_k": 50,
    "top_p": 1,
}

try:
    watsonx_llm = WatsonxLLM(
        model_id="meta-llama/llama-3-405b-instruct",
        url=watsonx_url,
        project_id=watsonx_project_id,
        apikey=watsonx_api_key,
        params=parameters
    )
except Exception as e:
    raise Exception(f"Failed to initialize WatsonxLLM: {str(e)}")

# Step 1: Fetch web data
urls = [
    "https://en.wikipedia.org/wiki/IBM_Watsonx",
    "https://en.wikipedia.org/wiki/IBM_Cloud"
]
try:
    loader = WebBaseLoader(urls)
    docs = loader.load()
except Exception as e:
    raise Exception(f"Failed to load web data: {str(e)}")

# Step 2: Clean and chunk data
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
doc_splits = text_splitter.split_documents(docs)
print(f"Created {len(doc_splits)} document chunks")

# Step 3: Set up IBM Slate embeddings and Chroma vector store
try:
    embedding_model = WatsonxEmbeddings(
        model_id="ibm/slate-125m-english-rtrvr",
        url=watsonx_url,
        project_id=watsonx_project_id,
        apikey=watsonx_api_key
    )
    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        embedding=embedding_model,
        persist_directory="./vector_store"
    )
    vectorstore.persist()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
except Exception as e:
    raise Exception(f"Failed to set up Chroma vector store: {str(e)}")

# Step 4: Define prompt template
template = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, say so concisely. Keep the answer clear and limited to three sentences.

Context: {context}

Question: {question}

Answer: """
prompt = PromptTemplate.from_template(template)

# Step 5: Create RAG chain
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | watsonx_llm
    | StrOutputParser()
)

# Step 6: Testing the RAG system with sample queries
sample_queries = [
    "What is the role of watsonx in AI development?",
    "How does IBM Cloud compare to other cloud providers?"
]

for query in sample_queries:
    try:
        response = rag_chain.invoke(query)
        print(f"\nQuery: {query}")
        print(f"Answer: {response}")
    except Exception as e:
        print(f"Error processing query '{query}': {str(e)}")

