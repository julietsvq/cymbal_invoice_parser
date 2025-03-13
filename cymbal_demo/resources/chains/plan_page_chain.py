from helpers import section_parsing_utils
from langchain_google_vertexai import ChatVertexAI
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder
)
from langchain_core.messages import SystemMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers.json import JsonOutputParser
import yaml

global_config = yaml.safe_load(open("app/config.yaml"))
config = yaml.safe_load(open("app/parsing/ocr/few_shot_examples/plan_page.yaml"))

few_shot_examples = section_parsing_utils.create_example_prompt(config["examples"])

class AccountSummary(BaseModel):
    first_table: str = Field(
        description="The first table that includes all info about what's included and not in the plan'"
    )
    second_table: str = Field(description="The second table that includes long-distance international calling prices")
    other_text: str = Field(description="Free-form text around the tables")

parser = JsonOutputParser(pydantic_object=AccountSummary)

system_message = SystemMessage(
    """You are an expert at OCR and can extract information accurately from tables and text shown in an image given the already existing text in that image. 
    In the image you'll find a table with what's included and not included in the plan.
    Then some text.
    Then a second table with the price of calling internationally.
    Then more text.
    
    Extract the tables and text in the image. Do not make up any numbers, use the text in the image provided for that.
    Extract the information in the image using the following format instructions: 
{format_instructions}
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
    final_response= "¿QUE INCLUYE TU PLAN?\n"
    for k in response.keys():
        final_response = final_response + response[k] +"\n"
    return final_response