import arrow
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
import re
import pandas as pd
import sys
sys.path.append("..")
from utils.data_utils import read_data
from utils.log_setting import logger
from prompt.base import prompt_dict


load_dotenv()
output_parser = StrOutputParser()
prompt_template = (
    "請根據 篩選資料 和 用戶提問，回答出正確答案並詳細說明理由。\n\n"
    "####\n"
    "開始!\n"
    "篩選資料：\n"
    "{filtered_data}\n"
    "用戶提問： {query}"
)

def get_answer_normal(query, filtered_data, prompt_template):
    if filtered_data is None:
        filtered_data = get_date_range_from_query(query)

    prompt = ChatPromptTemplate.from_template(prompt_template)
    model = ChatOpenAI(model=os.getenv("OPENAI_MODEL_NAME"), temperature=0.75)
    chain = prompt | model | output_parser
    # ans = chain.invoke({"query": query, "filtered_data": filtered_data})

    generator = chain.stream({"query": query, "filtered_data": filtered_data})
    return generator


if __name__ == "__main__":
    ans = get_answer_normal("台泥碳排量")
    print(ans)
