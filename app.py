from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from langchain_community.document_loaders import TextLoader
from qdrant_client import QdrantClient
from langchain_openai import OpenAIEmbeddings

from esg_toolkit.qdrant_utils import QdrantDBWrapper
from langchain.schema import Document as DDocc
from llama_index.readers import Document


load_dotenv()
loader = TextLoader("/home/abaoyang/app/core-abao-test/test.txt")
qdrant_wrapper = QdrantDBWrapper()
embedder = OpenAIEmbeddings()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
doc_count = qdrant_wrapper.count(collection_name="greenhousegas", count_filter=None).count
documents = qdrant_wrapper.scroll(
    collection_name="greenhousegas", limit=doc_count, scroll_filter=None
)[0]
documents = [d.payload for d in documents]
document = [Document.from_langchain_format(DDocc(page_content=d.payload["content"])) for d in documents]
text_splitter.split_documents(documents)


def init_qdrantdb(embedder, collection_name = "greenhousegas_openaiembeddings_abao"):
    host = "34.80.177.36"
    port = 6333
    client = qdrant_wrapper.client
    qdrant = Qdrant(
        client=client,
        collection_name=collection_name,
        embeddings=embedder,
        verbose=True
    )
    return qdrant


def init_splitter():
    return CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

