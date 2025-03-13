from langchain_google_vertexai import ChatVertexAI
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import SystemMessage
import yaml

global_config = yaml.safe_load(open("app/config.yaml"))

few_shot_examples = []

system_message = SystemMessage(
        """You are an expert at applying OCR and extracting the text from an image. Exact the text as is, not rewording anything.""")
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
    return response.content
    
