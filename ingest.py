import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def ingest_pdf():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    index_path = "faiss_index"
    
    loader = PyPDFLoader(r"/Users/prekshithmj/Documents/ML_PROJECTS/LIQUIDE-data-science-intern-assignment/Data.pdf")
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    vectorstore = FAISS.from_documents(splits, embeddings)
    vectorstore.save_local(index_path)
    print(" Vector store ingested successfully.")

if __name__ == "__main__":
    ingest_pdf()