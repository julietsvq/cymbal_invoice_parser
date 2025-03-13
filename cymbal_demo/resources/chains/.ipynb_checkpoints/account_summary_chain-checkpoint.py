from langchain_google_vertexai import ChatVertexAI
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import SystemMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers.json import JsonOutputParser
from helpers import section_parsing_utils
from typing import List
import yaml

global_config = yaml.safe_load(open("app/config.yaml"))
config = yaml.safe_load(open("app/parsing/ocr/few_shot_examples/account_summary.yaml"))

few_shot_examples = section_parsing_utils.create_example_prompt(config["examples"])

class BillItem(BaseModel):
    item_name: str = Field(description="Name of the item")
    amount: str = Field(description="Amount for that item")
    sign: str = Field(description="The sign for that item, written on the left of the item")

class AccountSummary(BaseModel):
    date: str = Field(
        description="The date of the summary, right next to 'Resumen de cuenta al'"
    )
    items: List[BillItem] = Field(description="Items of the summary")

parser = JsonOutputParser(pydantic_object=AccountSummary)

system_message = SystemMessage(
    """You are an expert at OCR and can extract information accurately from images given the already existing text in that image.
    In the image there are a set of items listed with text and an amount below that text. On the left of each item there might be a sign (plus sign "+", minus sign "-" or equals sign "=").
    Typically the first item is missing the sign. If so, make it empty.
    Extract the information in the image using the following format instructions: 
{format_instructions}

---------------
Use the text in the image and do not make up any numbers. If you can't find information about a field, fill it with 'None'.
"""
)
chat_prompt_messages = [system_message, MessagesPlaceholder(variable_name="messages")]
prompt = ChatPromptTemplate(
    chat_prompt_messages,
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

llm = ChatVertexAI(
    project=global_config["environment"]["project_id"],
    location=global_config["environment"]["region"],
    model=global_config["models"]["gemini-flash"],
    temperature=0, 
    max_output_tokens=global_config["models"]["max_output_tokens"]
)

chain = prompt | llm | parser

def post_process_response(response):
    final_response = "Resumen de cuenta al: " + response.get('date', "") + "\n"
    for item in response["items"]:
        if item["sign"] != "":
            final_response = final_response + item["sign"] + " "
        final_response = final_response + item["amount"] + " (" +item["item_name"].replace("\n", " ") +") "
    return final_response
