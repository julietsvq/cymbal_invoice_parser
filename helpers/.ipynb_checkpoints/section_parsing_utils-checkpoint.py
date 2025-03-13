from google.cloud import storage
import json
import re
import os

def extract_image_text_from_json(bucket_name, blob_name, section_name):
    """
    Downloads a JSON file from GCS and reads its content into a Python dictionary.

    Args:
      bucket_name: The name of the GCS bucket.
      blob_name: The name (path) of the JSON file within the bucket.

    Returns:
      The text of that section or empty if it could not be found.
    """

    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    # Download the JSON file content as a string
    json_string = blob.download_as_string().decode("utf-8")

    # Parse the JSON string into a Python dictionary
    data = json.loads(json_string)
    for section in data:
        if section_name not in section["path"]: # this is to handle plan_page_x
            continue
        if "plan_page" in section_name:
            section_name = "plan_page"
        if section["image_extracted"] == True and section["section_info"]["title"] == section_name:
            full_str = ""
            for item in section['filtered_elements']:
                for elem in item:
                    full_str = full_str + elem
            return full_str
    return ""

def extract_image_text_from_json_for_example(bucket_name, blob_name, section_name):
    cache_dir = ".tmp/few_shot_examples"
    cache_file = os.path.join(cache_dir, os.path.basename(blob_name))

    print("cache file: " + cache_file)
    
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            data = json.load(f)
    else:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        json_string = blob.download_as_string().decode("utf-8")
        data = json.loads(json_string)

        os.makedirs(cache_dir, exist_ok=True)
        with open(cache_file, "w") as f:
            json.dump(data, f)

    for section in data:
        if section_name not in section["path"]:
            continue
        if "plan_page" in section_name:
            section_name = "plan_page"
        if section["image_extracted"] == True and section["section_info"]["title"] == section_name:
            full_str = ""
            for item in section['filtered_elements']:
                for elem in item:
                    full_str = full_str + elem
            return full_str
    return ""

def extract_bucket_and_file_path(gcs_path):
    """
    Extracts the bucket name and file path from a GCS URI, with input validation.

    Args:
    gcs_path: The GCS URI (e.g., 'gs://my-bucket/path/to/file.txt').

    Returns:
    A tuple (bucket_name, file_path) if the input is valid, or None if it's not.
    """

    # Regular expression pattern to match GCS URIs
    pattern = r"^gs://([^/]+)/(.*)$"

    match = re.match(pattern, gcs_path)
    if match:
        bucket_name = match.group(1)
        file_path = match.group(2)
        return bucket_name, file_path
    else:
        print("Invalid GCS path format.")
        return None

def get_text_for_section(parsed_text_uri, gcs_uri):
    """ Given a path like gs://cymbal_invoice_parser/bills_organised/030_648/2024_04/016/016_plan_page_0.png, it returns the text contained in that image, which is stored within the _output.json file"""
    splitted = gcs_uri.replace("gs://", "").split("/")
    bucket_name = splitted[0]
    bucket_path = f"gs://{bucket_name}/"
    json_name = parsed_text_uri.replace(bucket_path, "")
    filename_prefix = parsed_text_uri.replace(".json", "").replace(bucket_path, "").replace("_output", "")
    section_name = gcs_uri.replace(".png", "").replace(f"{bucket_path}{filename_prefix}", "")[1:]
    return extract_image_text_from_json(bucket_name, json_name, section_name=section_name)
    
def get_text_for_section_for_example(parsed_text_uri, gcs_uri):
    """ Given a path like gs://poc-genai/bills_organised/030_648/2024_04/016/016_plan_page_0.png, it returns the text contained in that image, which is stored within the _output.json file"""
    splitted = gcs_uri.replace("gs://", "").split("/")
    bucket_name = splitted[0]
    bucket_path = f"gs://{bucket_name}/"
    json_name = parsed_text_uri.replace(bucket_path, "")
    filename_prefix = parsed_text_uri.replace(".json", "").replace(bucket_path, "").replace("_output", "")
    section_name = gcs_uri.replace(".png", "").replace(f"{bucket_path}{filename_prefix}", "")[1:]
    return extract_image_text_from_json_for_example(bucket_name, json_name, section_name=section_name)

def get_gcs_blob_mime_type(gcs_uri):
    """Fetches the MIME type (content type) of a Google Cloud Storage blob.

    Args:
        gcs_uri (str): The GCS URI of the blob in the format "gs://bucket-name/object-name".

    Returns:
        str: The MIME type of the blob (e.g., "image/jpeg", "text/plain") if found,
             or None if the blob does not exist or an error occurs.
    """
    storage_client = storage.Client()

    try:
        bucket_name, object_name = gcs_uri.replace("gs://", "").split("/", 1)

        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(object_name)
        blob.reload()
        return blob.content_type

    except Exception as e:
        print(f"Error retrieving MIME type for {gcs_uri}: {e}")
        return None  # Indicate failure
    
def gcs_path_to_content_for_gemini(file_uri, mime_type=None):
    if mime_type is None:
        if file_uri.endswith(".pdf"):
            mime_type="application/pdf"
        elif file_uri.endswith(".png"):
            mime_type="image/png"
        elif file_uri.endswith(".jpg"):
            mime_type="image/jpg"
        elif file_uri.endswith(".jpeg"):
            mime_type="image/jpeg"
        else:
            mime_type = get_gcs_blob_mime_type(file_uri)
    return {"type": "media", "file_uri": f"{file_uri}", "mime_type": f"{mime_type}"}

def create_example_prompt(examples, mime_type=None):
    examples_list = []
    for i, example in enumerate(examples):
        path_to_gcs = example.get("path_to_gcs")
        parsed_text_uri = example.get("parsed_text_uri")
        if path_to_gcs:
            media_message = gcs_path_to_content_for_gemini(path_to_gcs, mime_type=mime_type)
            text_in_image_description = ""
            if parsed_text_uri:
                text_in_image = get_text_for_section_for_example(parsed_text_uri, path_to_gcs).strip()
                text_in_image_description = f"This is the text contained in the image:\n```\n{text_in_image}```"
            text_message = {
                "type": "text",
                "text": f"\n\n========== Example {i}:=========\n{text_in_image_description}",
            }
            examples_list += [text_message, media_message]

        text = example["output"].get("text")
        json_dict = example["output"].get("json")
        output_text = ""
        if text is not None:
            output_text += f"{text}\n"
        if json_dict is not None:
            output_text += f"\n{json.dumps(json_dict, indent=2)}\n"
        text_output = {
            "type": "text",
            "text": f"This is the expected output:\n```\n{output_text}```",
        }
        examples_list += [text_output]
    return examples_list