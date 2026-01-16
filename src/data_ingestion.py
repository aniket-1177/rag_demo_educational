from langchain.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_documents(directory: str):
    """Load documents from a directory (supports txt and pdf)."""
    loader = DirectoryLoader(
        directory,
        glob="**/*.*",
        loader_cls=lambda path: PyPDFLoader(path) if path.endswith(".pdf") else TextLoader(path)
    )
    return loader.load()

def chunk_documents(documents, chunk_size: int = 500, chunk_overlap: int = 50):
    """
    Chunk documents with strategies and metadata (5.5: Chunking Strategies and Metadata).
    - Uses recursive splitter for overlapping chunks.
    - Adds metadata like source and page/chunk ID.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
    return chunks