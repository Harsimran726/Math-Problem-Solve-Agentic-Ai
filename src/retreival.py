from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.storage import LocalFileStore
from langchain_community.vectorstores import Qdrant
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_classic.storage._lc_store import create_kv_docstore


# high perfomance embedding model

# embedding = #

vector_store = Qdrant.from_documents(
    [],
    # embedding,
    location=":memory:",
    collection_name="math_child_chunks"
)


# persistant docstore
fs = LocalFileStore("./parent_store")
store = create_kv_docstore(fs)

# hierarchical splitters for parent and child 

parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000,chunk_overlap=200)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400,chunk_overlap=50)


# initilizing the retriever 

retriever = ParentDocumentRetriever(
    vectorstore=vector_store,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter
)



# addting the docs
retriever.add_documents()