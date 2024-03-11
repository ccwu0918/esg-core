from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS

from typing import Optional, Union

from langchain.schema import BaseRetriever
from langchain_community.vectorstores import VectorStore
from langchain_community.vectorstores.qdrant import Qdrant

from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain import hub
from langchain.agents import create_openai_functions_agent
from langchain.agents import AgentExecutor

from src.tools.tavily import get_gsrp, tavily_search_tool


loader = WebBaseLoader("https://hackmd.io/@computerVision/BkYrDUDxY")
docs = loader.load()
documents = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200
).split_documents(docs)
vector = FAISS.from_documents(documents, OpenAIEmbeddings())
retriever = vector.as_retriever()
relv_docs  = retriever.get_relevant_documents("PaddleOCR介紹")

retriever_tool = create_retriever_tool(
    retriever,
    "ocr_km_search",
    "Search for information about OCR. For any questions about OCR, you must use this tool!",
)

tools = [tavily_search_tool, retriever_tool]

llm = ChatOpenAI(model="gpt-4-turbo-preview", temperature=0)

# Get the prompt to use - you can modify this!
prompt = hub.pull("hwchase17/openai-functions-agent")
prompt.messages

agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

agent_executor.invoke({"input": "PaddleOCR表現得如何?"})

