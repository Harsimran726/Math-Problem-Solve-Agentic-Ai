
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
import logging
from tenacity import retry, stop_after_attempt, wait_exponential
from langchain_classic.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings # Updated import
import pdfplumber

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


embedding = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},  # Use 'cuda' if you have a GPU
    encode_kwargs={'normalize_embeddings': True}
)
pdf_path1 = r"C:\Data\Self study\Math Solver Agentic Ai\material\Mathematics for IIT JEE Main and Advanced Algebra - G S N Murti _ U M Swamy.pdf"
pdf_path2 = r"C:\Data\Self study\Math Solver Agentic Ai\material\ncert-books-for-class-12-maths.pdf"

# # # # pdf_path4 = "./Questions.pdf"   # questions pdf


def get_extract_content(pdf_path):
    text =""
    tables=[]
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
                tables.extend(page.extract_tables())
        return text, tables
    except Exception as e:
        logger.error(f"Error extracting content from {pdf_path}: {e}")
        raise

def load_pdf(pdf_paths):
    all_documents = []
    i =0
    for pdf_path in pdf_paths:
        try:
            i +=1
            text, tables = get_extract_content(pdf_path)
            full_text = f"{text}\nTables: {str(tables)}"
            vector_store = text_splitter(full_text)
            all_documents.append(vector_store)
            print(f"TOTAL DOCUMENTS CREATED {i}")
        except Exception as e:
            logger.error(f"Error loading PDF {pdf_path}: {e}")
            continue
    return all_documents

def load_pdf(pdf_paths):
    vector_store = None
    print("inside load pdf")
    for pdf_path in pdf_paths:
        try:
            text, tables = get_extract_content(pdf_path)
            full_text = f"{text}\nTables: {str(tables)}"
            new_vs = text_splitter(full_text)

            if vector_store is None:
                vector_store = new_vs
            else:
                vector_store.merge_from(new_vs)

        except Exception as e:
            logger.error(f"Error loading PDF {pdf_path}: {e}")
            continue
    return vector_store

def text_splitter(full_text):
    text_splitter_instance = RecursiveCharacterTextSplitter(
        chunk_size=800,  
        chunk_overlap=100,  
        )
    chunks = text_splitter_instance.split_text(full_text)
    documents = [Document(page_content=chunk) for chunk in chunks]
    # print(f"DOCUMENTS\n{documents}")
    embedder = embedding
    print(f"here is text spliiter")
    return FAISS.from_documents(documents, embedder)

vector_st = load_pdf([pdf_path1,pdf_path2])
print(f"NOW GOING TO SAVE")
vector_st.save_local("faiss_index")