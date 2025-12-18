from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from dotenv import load_dotenv



load_dotenv()

def read_splitter(filename):
    pdf_path=Path(__file__).parent/"docs"/f"{filename}"

    loader = PyPDFLoader(pdf_path)
    docs=loader.load()

    ## split the docs into small chunks
    chunk_splitter=RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=500
    )

    ## here we are giving the doc to chunk splitter instance.
    chunks=chunk_splitter.split_documents(docs)

    return chunks

### calling here read splitter function
#read_splitter("Practical NLP.pdf")

## Vector embeding
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")


vector_store=QdrantVectorStore.from_documents(
    documents=read_splitter("Practical NLP.pdf"),
    embedding=embeddings,
    collection_name="learning_rag"
)
print(vector_store)