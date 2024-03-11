from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
import pandas as pd

from data_init import get_documents

load_dotenv()
collection_name = "greenhousegas"
raw_docs, documents, metadatas = get_documents(collection_name)

text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1000,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
)
texts = text_splitter.create_documents(documents, metadatas=metadatas)
print(texts[0])

## -----------
splitted_raw_docs = []
for text in texts:
    d = text.metadata
    d.update({"content": text.page_content})
    splitted_raw_docs.append(d)

pd.DataFrame(splitted_raw_docs).to_json(
    "splitted_greenhousegas.json", lines=True, orient="records", force_ascii=False
)
pd.DataFrame(splitted_raw_docs).to_csv("splitted_greenhousegas.csv", index=False)
## -----------
