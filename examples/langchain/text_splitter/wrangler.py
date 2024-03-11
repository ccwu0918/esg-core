import pandas as pd
from data_init import get_documents, get_company_name, cleansing, document_store_qdrant
from dotenv import load_dotenv

# load_dotenv()
# collection_name = "cleansed_greenhousegas"
# raw_docs, documents, metadatas = get_documents(collection_name)

# df = pd.DataFrame(raw_docs)
# df["company_name"] = df["content"].apply(get_company_name)
# df["content"] = df["content"].apply(cleansing)
# df.drop(columns=["full_text"], inplace=True)

# industry_dict = pd.read_csv("coid_industry.csv")
# industry_dict = industry_dict.set_index("CompCode").to_dict()["IndCat"]
# df["industry"] = df["co_id"].map(industry_dict)
# # df.to_json("twse-greenhousegas.json", lines=True, orient="records", force_ascii=False)

# df["idx"] = df.index
# document_store_qdrant(data=df.iloc[1000:], url_name="greenhousegas", collection_name="greenhousegas")

# df = pd.read_json("twse-greenhousegas.json", lines=True, orient="records")

##==============


load_dotenv()
collection_name = "greenhousegas"
raw_docs, documents, metadatas = get_documents(collection_name)
df = pd.DataFrame(raw_docs)
