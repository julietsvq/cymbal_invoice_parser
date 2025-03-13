from langchain_google_vertexai import ChatVertexAI
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder
)
from langchain_core.messages import SystemMessage
from helpers import section_parsing_utils
import yaml

global_config = yaml.safe_load(open("app/config.yaml"))
config = yaml.safe_load(open("app/parsing/ocr/few_shot_examples/period_details.yaml"))

few_shot_examples = section_parsing_utils.create_example_prompt(config["examples"])


system = f"You are an expert in image bill understanding.\n" \
"Your task is to extract bill information in table format.\n" \
"If you see multiple tables, extract all of them, just separate each table with a double newline\n" \
"You have at your disposal the text extracted from the bills. Such text can contains some mistakes in the layout like line breaking but it is a good reference to make sure the extracted numbers in the tables are accurate.\n" \
"Never report numbers you don't see in the provided text; it is ok instead to add column or record names if missing. For instance you can add a 'total' record or a column name 'BONIFICACION' or 'TOTAL' if missing. A negative number should always have column name 'BONIFICACION'.\n" \
"Here are a few examples of correct parsing:\n"
system_message = SystemMessage(content=system)

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
    return response.content.strip().replace("```", "")
