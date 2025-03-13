import os
import yaml
from functools import partial
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from langchain_core.messages import HumanMessage

import helpers.generic_pdf_parser.generic_pdf_parser as gpp
from helpers import file_cacher 
from helpers import section_parsing_utils
from cymbal_demo.resources.chains import invoice_details_chain#, important_information_chain, other_details_chain, account_summary_chain, lines_info_chain, plan_page_chain, period_details_chain


#sys.path.append(os.path.abspath(os.path.join(path, os.pardir)))

config = yaml.safe_load(open(os.path.abspath(os.path.join(os.getcwd(), os.pardir)) + "/cymbal_demo/resources/config.yaml"))
section_infos = config["ocr"]["cropping"]["section_infos"]

def run_single_chain(parsed_text_uri, chain_and_input):
    """
    Runs an LLM chain.
    
    Args:
        parsed_text_uri: the text contained in the file provided in chain_and_input's second parameter.
        chain_and_input: the chain_file of the chain to run and the input path to a file on GCS to run it with.
    """
    chain_file, file_uri = chain_and_input
    selected_chain = globals()[chain_file]
    
    # We want the chain and the examples from the selected chain
    chain = selected_chain.chain
    few_shot_examples = selected_chain.few_shot_examples
    
    new_content = section_parsing_utils.gcs_path_to_content_for_gemini(file_uri)
    text_in_new_image = section_parsing_utils.get_text_for_section(parsed_text_uri, file_uri)
    new_input_text = {
        "type": "text",
        "text": """The image contains this text:"""
        + text_in_new_image +"\n Output:",
    }

    all_messages = few_shot_examples + [new_input_text, new_content]

    response = chain.invoke([HumanMessage(content=all_messages).dict()])
    return chain_file, selected_chain.post_process_response(response)

def parse_pdf(gcs_path_to_pdf):
    """ Parses the pdf stored under gcs_path_to_pdf and returns a defaultdict containing the parsed text for each section"""
    # Crop the PDF into images according to section_infos
    output = gpp.create_images_from_pdf(gcs_path_to_pdf, section_infos)
    ocr_func_calls = []
    # Depending on what images are available, run the corresponding chains
    for section in output:
        section_name = section["section_info"]["title"]
        ocr_chain = config["ocr"]["ocr_chain_func"].get(section_name, None)
        if ocr_chain is not None and section["image_extracted"] == True:
            ocr_func_calls.append([ocr_chain, section["path"]])
    
    max_workers = len(ocr_func_calls)
    final_results = defaultdict(str)
    # Pass the text contained in each section to the chain to prevent hallucinations.
    parsed_text_uri = gcs_path_to_pdf.replace(".pdf", "_output.json")
    chain_func = partial(run_single_chain, parsed_text_uri)
    # TODO consider async calls to  handle multiple requests within a thread
    with ThreadPoolExecutor(max_workers) as pool:
        for result in pool.map(chain_func, ocr_func_calls):
            chain_file, output = result
            final_results[chain_file] += output + "\n"
    return final_results