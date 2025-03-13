from langchain_google_vertexai import ChatVertexAI
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import SystemMessage
from helpers import section_parsing_utils
import yaml

global_config = yaml.safe_load(open("app/config.yaml"))
config = yaml.safe_load(open("app/parsing/ocr/few_shot_examples/other_details.yaml"))

few_shot_examples = section_parsing_utils.create_example_prompt(config["examples"])

system_message = SystemMessage(
    """You are an expert at OCR and can extract information accurately from images given the already existing text in that image.
    In the image there are a set of rows listing charges in a bill. Use the text in the image and do not make up any numbers.
"""
)
chat_prompt_messages = [system_message, MessagesPlaceholder(variable_name="messages")]
prompt = ChatPromptTemplate(chat_prompt_messages)

llm = ChatVertexAI(
    project=global_config["environment"]["project_id"],
    location=global_config["environment"]["region"],
    model=global_config["models"]["gemini-flash"],
    temperature=0, 
    max_output_tokens=global_config["models"]["max_output_tokens"]
)

chain = prompt | llm

def post_process_response(response):
    return response.content.replace("```","")
