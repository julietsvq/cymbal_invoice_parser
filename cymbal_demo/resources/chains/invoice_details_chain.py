import os
from helpers import section_parsing_utils
from langchain_google_vertexai import ChatVertexAI
from langchain_core.output_parsers.json import JsonOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder
)
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.pydantic_v1 import BaseModel, Field
import yaml
from typing import List

class Item(BaseModel):
    item_name: str = Field(description="Name of the item")
    value: str = Field(description="Value of that item")

class InvoiceDetails(BaseModel):
    items: List[Item] = Field(description="List of details of the invoice")
    
parser = JsonOutputParser(pydantic_object=InvoiceDetails)

path = os.getcwd()
global_config = yaml.safe_load(open(os.path.abspath(os.path.join(path, os.pardir)) + "/cymbal_demo/resources/config.yaml"))
config = yaml.safe_load(open(os.path.abspath(os.path.join(path, os.pardir)) + "/cymbal_demo/resources/few_shot_examples/invoice_details.yaml"))

few_shot_examples = section_parsing_utils.create_example_prompt(config["examples"])

system_message = SystemMessage(
    """You are an expert at OCR and can extract information accurately from images given the already existing text in that image.
    In the image there are a set of lines listing invoice information. Use the text in the image and do not make up any numbers. Parse all the lines in the image. If a value is missing, use an empty string. Use these format instructions:
    {format_instructions}
"""
)

chat_prompt_messages = [system_message, MessagesPlaceholder(variable_name="messages")]
prompt = ChatPromptTemplate(
    chat_prompt_messages,
    partial_variables={"format_instructions": parser.get_format_instructions()},
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
    final_response = ""
    total_items = len(response["items"])
    for index, item in enumerate(response["items"]):
        final_response += item["item_name"]
        if ":" not in item["item_name"]:
            final_response += ":"
        final_response += " " + item["value"]
        # Add a new line for except for last element
        if index < total_items - 1:
            final_response += "\n"
    return final_response
