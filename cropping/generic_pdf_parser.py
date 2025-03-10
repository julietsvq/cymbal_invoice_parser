__all__ = ['create_images_from_pdf']  # Specifies the function to be publicly accessible from this module.

from pdfminer.high_level import extract_pages  # Extracts text and other content from PDF files.
import fitz  # Works with PDF documents using PyMuPDF.
from PIL import Image  # Handles image processing and manipulation.
import cv2  # OpenCV library for computer vision tasks.
import numpy as np  # Provides powerful array manipulation and numerical computation capabilities.
from google.cloud import storage  # Interacts with Google Cloud Storage.
import re  # Supports regular expression operations for pattern matching.
import os  # Provides functions for interacting with the operating system.
import io  # Handles various types of I/O operations.
import copy  # Enables creating deep copies of objects.
from urllib.parse import urlparse  # Parses URLs into their components.
import concurrent.futures  # Facilitates concurrent execution of tasks using threads or processes. 
import json

def _save_json_to_path(data, path):
    """
    Saves JSON data to a specified path, which can be either a local file path or a GCS path.

    Args:
        data (dict or list): The JSON-serializable data to be saved.
        path (str): The path where the JSON file should be saved. 
                     It can be a local file path (e.g., '/path/to/file.json') or a GCS path (e.g., 'gs://bucket-name/path/to/file.json').
    """

    if path.startswith('gs://'):
        # GCS path
        parsed_url = urlparse(path)
        bucket_name = parsed_url.netloc
        blob_name = parsed_url.path.lstrip('/')

        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        blob.upload_from_string(json.dumps(data), content_type='application/json')
        print(f"JSON data saved to GCS: {path}")
    else:
        # Local file path
        with open(path, 'w') as f:
            json.dump(data, f, indent=4)  # Save with indentation for better readability
        print(f"JSON data saved to local file: {path}")

def _is_gcs_path(path):
    """
    Checks if the given path is a valid Google Cloud Storage (GCS) path or a local file path.

    Args:
        path (str): The path to be checked.

    Returns:
        bool: True if the path is a valid GCS path, False if it's a local file path.

    Raises:
        ValueError: If the path is neither a valid GCS path nor a local file path.
    """
    # GCS path pattern: gs://bucket-name/blob-name
    gcs_pattern = r'^gs://[\w\-]+/.*$'
    
    if re.match(gcs_pattern, path):
        return True
    elif os.path.exists(path) or os.path.isabs(path):
        return False
    else:
        raise ValueError(f"Invalid path: {path}")
    
def _get_filename_from_gcs_path(gcs_path):
    """
    Extracts the filename from a Google Cloud Storage (GCS) path.

    Args:
        gcs_path (str): The GCS path in the format 'gs://bucket-name/path/to/file.pdf'.

    Returns:
        str: The filename extracted from the GCS path (e.g., 'file.pdf').
    """
    parsed_url = urlparse(gcs_path)
    path = parsed_url.path.lstrip('/')  # Remove leading slash if present
    filename = path.split('/')[-1]  # Get the last part of the path
    return filename

def _upload_file_to_gcs(local_file_path, gcs_path):
    """
    Uploads a local file to a specified Google Cloud Storage (GCS) path.

    Args:
        local_file_path: The path to the local file to be uploaded.
        gcs_path: The full GCS path (e.g., 'gs://your-bucket-name/path/to/file.pdf') where the file should be uploaded.
    """
    parsed_url = urlparse(gcs_path)
    bucket_name = parsed_url.netloc
    blob_path = parsed_url.path.lstrip('/')  # Remove leading slash

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_path)

    blob.upload_from_filename(local_file_path) 

def _upload_files_concurrently(local_file_paths, gcs_paths):
    """
    Uploads multiple local files to their corresponding Google Cloud Storage (GCS) paths concurrently using threads.

    Args:
        local_file_paths: A list of paths to local files to be uploaded.
        gcs_paths: A list of corresponding GCS paths for each local file.
    """

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(_upload_file_to_gcs, local_path, gcs_path) 
                   for local_path, gcs_path in zip(local_file_paths, gcs_paths)]

        # Wait for all uploads to complete
        concurrent.futures.wait(futures)

def _get_pdf_info(pages, 
                    start_text, 
                    end_text = None, 
                    continue_prompt = None,
                    start_after_page = 0):
    """
    Extracts key information about a section within a PDF document.

    This function analyzes a list of page layouts to identify the start and end of a section,
    as well as any "continue" prompts within that section. It collects coordinates and page
    numbers for these elements and stores them in a dictionary for further processing.

    Args:
        pages: A list of page layout objects representing the pages of a document.
        start_text: The text that marks the beginning of the desired section.
        end_text (optional): The text that marks the end of the desired section. Defaults to None.
        continue_prompt (optional): Text indicating a continuation of the section on the next page. Defaults to None.
        start_after_page (int, optional): The page number (0-indexed) after which to start the search. Defaults to 0 (start from the beginning).

    Returns:
        A dictionary containing the following information:
            - "pages": The last processed page number (0-indexed).
            - "start_found": Boolean indicating if the start text was found.
            - "end_found": Boolean indicating if the end text was found.
            - "start_in_page": Page number where the start text is located.
            - "end_in_page": Page number where the end text is located.
            - "start_coordinates": Coordinates (left, top, bottom, right, page_width, page_height) of the start text bounding box.
            - "end_coordinates": Coordinates (left, top, bottom, right, page_width, page_height) of the end text bounding box.
            - "rect_coordinates": An empty list (intended for storing calculated rectangle coordinates in later processing).
            - "continue_coordinates": A list of dictionaries, each containing coordinates of "continue" prompts.
            - "all_components": A list of text components encountered between the start and end markers (if found).
    """
    
    pdf_info = {
        "pages": 0,
        "start_found": False,
        "end_found": False,
        "start_in_page": 0,
        "end_in_page":0,
        "start_coordinates": {},
        "end_coordinates": {},
        "rect_coordinates": [],
        "continue_coordinates": [],
        "all_components" : [],
        "all_elements" : []
    }

    #Find start and end pages for the section
    for page_layout in pages:
        if (page_layout.pageid < start_after_page):
            continue
        pdf_info["pages"] = page_layout.pageid
        for element in page_layout:
            if hasattr(element, 'get_text'):
                if (not pdf_info["start_found"] and start_text in element.get_text()):
                    pdf_info["start_found"] = True
                    pdf_info["start_in_page"] = page_layout.pageid
                    pdf_info["start_coordinates"]["left"] = element.x0
                    pdf_info["start_coordinates"]["bottom"] = element.y0
                    pdf_info["start_coordinates"]["right"] = element.x1
                    pdf_info["start_coordinates"]["top"] = element.y1
                    pdf_info["start_coordinates"]["page_width"] = page_layout.width
                    pdf_info["start_coordinates"]["page_height"] = page_layout.height       
                    
                elif (end_text and pdf_info["start_found"] and (not pdf_info["end_found"]) and (end_text in element.get_text())):
                    pdf_info["end_found"] = True
                    pdf_info["end_in_page"] = page_layout.pageid
                    pdf_info["end_coordinates"]["left"] = element.x0
                    pdf_info["end_coordinates"]["bottom"] = element.y0
                    pdf_info["end_coordinates"]["right"] = element.x1
                    pdf_info["end_coordinates"]["top"] = element.y1
                    pdf_info["end_coordinates"]["page_width"] = page_layout.width
                    pdf_info["end_coordinates"]["page_height"] = page_layout.height      

                elif (continue_prompt and pdf_info["start_found"] and not pdf_info["end_found"] and continue_prompt in element.get_text()):
                    temp_cont_coordinates = {}
                    temp_cont_coordinates["page"] = page_layout.pageid
                    temp_cont_coordinates["left"] = element.x0
                    temp_cont_coordinates["bottom"] = element.y0
                    temp_cont_coordinates["right"] = element.x1
                    temp_cont_coordinates["top"] = element.y1
                    pdf_info["continue_coordinates"].append(temp_cont_coordinates)

                pdf_info["all_components"].append(element.get_text())
                pdf_info["all_elements"].append({"element": element, "page": page_layout.pageid, "height": page_layout.height})
    
    return pdf_info

def _find_rectangles_of_section(pdf_info,
                             include_start_box = True, 
                             include_end_box = True, 
                             fetch_end_page_only = False,
                             use_static_width_from_start_x = None,
                             use_static_height_from_start_x = None,
                             header_height_to_remove_for_middle_pages = 0,
                             top_offset = 0,
                             bottom_offset = 0,
                             left_offset = 0,
                             right_offset = 0):
    """
    Calculates rectangular regions within a PDF based on provided information and configuration.

    This function processes information about a PDF section, including start and end text locations,
    and generates rectangular coordinates for image extraction. It handles various scenarios:

    * Single-page sections.
    * Multi-page sections with start and end text on different pages.
    * Optional inclusion/exclusion of start and end text bounding boxes.
    * Static width/height overrides.
    * Header removal for middle pages.
    * Offsets for fine-tuning rectangle positions.

    Args:
        pdf_info: A dictionary containing information about the PDF section, including:
            - "start_found": Boolean indicating if the start text was found.
            - "end_found": Boolean indicating if the end text was found.
            - "start_in_page": Page number where the start text is located.
            - "end_in_page": Page number where the end text is located.
            - "start_coordinates": Coordinates (left, top, bottom, page_height, page_width) of the start text bounding box.
            - "end_coordinates": Coordinates (left, top, bottom, page_height, page_width) of the end text bounding box.
            - "continue_coordinates": List of dictionaries containing coordinates of "continue" prompts within the section.
            - "rect_coordinates": (Initially empty) List to store calculated rectangle coordinates.
        include_start_box: Boolean indicating whether to include the start text bounding box in the rectangle.
        include_end_box: Boolean indicating whether to include the end text bounding box in the rectangle.
        fetch_end_page_only: Boolean indicating whether to only fetch the end page (relevant when start and end are on different pages).
        use_static_width_from_start_x: If provided, overrides the rectangle width with this value.
        use_static_height_from_start_x: If provided, overrides the rectangle height with this value.
        header_height_to_remove_for_middle_pages: Height of the header to exclude on middle pages of multi-page sections.
        top_offset, bottom_offset, left_offset, right_offset: Offsets to adjust the rectangle's position.

    Returns:
        A list of dictionaries, each representing a calculated rectangle with keys:
            - "page": The page number.
            - "x0", "y0": Top-left coordinates.
            - "x1", "y1": Bottom-right coordinates.
    """
    #return the rectangle if the start and end pages are the same
    if pdf_info["start_found"] and ((pdf_info["end_found"] and pdf_info["start_in_page"] == pdf_info["end_in_page"]) or use_static_height_from_start_x):
        temp_rect_coordinates = {}
        temp_rect_coordinates["page"] = pdf_info["start_in_page"]
        temp_rect_coordinates["x0"] = pdf_info["start_coordinates"]["left"] + left_offset
        if include_start_box:
            temp_rect_coordinates["y0"] = pdf_info["start_coordinates"]["page_height"] - pdf_info["start_coordinates"]["top"] + top_offset
        else:
            temp_rect_coordinates["y0"] = pdf_info["start_coordinates"]["page_height"] - pdf_info["start_coordinates"]["bottom"] + top_offset
        if use_static_width_from_start_x:
            temp_rect_coordinates["x1"] = pdf_info["start_coordinates"]["left"] + use_static_width_from_start_x + right_offset
        else:
            temp_rect_coordinates["x1"] = pdf_info["start_coordinates"]["page_width"] + right_offset
        
        if use_static_height_from_start_x:
            temp_rect_coordinates["y1"] = pdf_info["start_coordinates"]["page_height"] - pdf_info["start_coordinates"]["bottom"] + use_static_height_from_start_x + bottom_offset
        else:            
            if include_end_box:
                temp_rect_coordinates["y1"] = pdf_info["start_coordinates"]["page_height"] - pdf_info["end_coordinates"]["bottom"] + bottom_offset
            else:
                temp_rect_coordinates["y1"] = pdf_info["start_coordinates"]["page_height"] - pdf_info["end_coordinates"]["top"] + bottom_offset
        pdf_info["rect_coordinates"].append(temp_rect_coordinates)
    
    elif pdf_info["start_found"] and ((pdf_info["end_found"] and fetch_end_page_only)):
        temp_rect_coordinates = {}
        temp_rect_coordinates["page"] = pdf_info["end_in_page"]
        temp_rect_coordinates["x0"] = pdf_info["start_coordinates"]["left"] + left_offset
        if include_start_box:
            temp_rect_coordinates["y0"] = pdf_info["start_coordinates"]["page_height"] - pdf_info["start_coordinates"]["top"] + top_offset
        else:
            temp_rect_coordinates["y0"] = pdf_info["start_coordinates"]["page_height"] - pdf_info["start_coordinates"]["bottom"] + top_offset
        if use_static_width_from_start_x:
            temp_rect_coordinates["x1"] = pdf_info["start_coordinates"]["left"] + use_static_width_from_start_x + right_offset
        else:
            temp_rect_coordinates["x1"] = pdf_info["start_coordinates"]["page_width"] + right_offset
        
        if use_static_height_from_start_x:
            temp_rect_coordinates["y1"] = pdf_info["start_coordinates"]["page_height"] - pdf_info["start_coordinates"]["bottom"] + use_static_height_from_start_x + bottom_offset
        else:            
            if include_end_box:
                temp_rect_coordinates["y1"] = pdf_info["start_coordinates"]["page_height"] - pdf_info["end_coordinates"]["bottom"] + bottom_offset
            else:
                temp_rect_coordinates["y1"] = pdf_info["start_coordinates"]["page_height"] - pdf_info["end_coordinates"]["top"] + bottom_offset
        pdf_info["rect_coordinates"].append(temp_rect_coordinates)

    #return one rectangle for each page if the start and end pages are not the same
    elif pdf_info["start_found"] and pdf_info["end_found"]:
        for i in range(pdf_info["start_in_page"], pdf_info["end_in_page"] + 1):
            temp_rect_coordinates = {}
            temp_rect_coordinates["page"] = i

            #for the first page
            if i == pdf_info["start_in_page"]:
                #start the rect at the start_text component
                temp_rect_coordinates["x0"] = pdf_info["start_coordinates"]["left"] + left_offset
                if include_start_box:
                    temp_rect_coordinates["y0"] = pdf_info["start_coordinates"]["page_height"] - pdf_info["start_coordinates"]["top"] + top_offset
                else:
                    temp_rect_coordinates["y0"] = pdf_info["start_coordinates"]["page_height"] - pdf_info["start_coordinates"]["bottom"] + top_offset

                #check if there is continue_prompt on the same page
                prompt_rect = None
                for coord_dict in pdf_info["continue_coordinates"]:
                    if coord_dict["page"] == i:
                        prompt_rect = coord_dict
                        break
        
                #end the rect at the bottom of the page or at the continue prompt
                if use_static_width_from_start_x:
                    temp_rect_coordinates["x1"] = pdf_info["start_coordinates"]["left"] + use_static_width_from_start_x + right_offset
                else:
                    temp_rect_coordinates["x1"] = pdf_info["start_coordinates"]["page_width"] + right_offset

                if prompt_rect:
                    temp_rect_coordinates["y1"] = pdf_info["start_coordinates"]["page_height"] - prompt_rect["bottom"] - 8
                else:
                    temp_rect_coordinates["y1"] = pdf_info["start_coordinates"]["page_height"]

            #for the last page
            elif i == pdf_info["end_in_page"]:
                #start the rect at the top left of the page. Use start_text component's left.
                temp_rect_coordinates["x0"] = pdf_info["start_coordinates"]["left"] + left_offset
                temp_rect_coordinates["y0"] = 0 + header_height_to_remove_for_middle_pages

                #end the rect at the end_text component
                if use_static_width_from_start_x:
                    temp_rect_coordinates["x1"] = pdf_info["start_coordinates"]["left"] + use_static_width_from_start_x + right_offset
                else:
                    temp_rect_coordinates["x1"] = pdf_info["start_coordinates"]["page_width"] + right_offset

                if include_end_box:
                    temp_rect_coordinates["y1"] = pdf_info["start_coordinates"]["page_height"] - pdf_info["end_coordinates"]["bottom"] + bottom_offset
                else:
                    temp_rect_coordinates["y1"] = pdf_info["start_coordinates"]["page_height"] - pdf_info["end_coordinates"]["top"] + bottom_offset
                
            #for the middle pages
            else:
                #start the rect at the top left of the page. Use start_text component's left.
                temp_rect_coordinates["x0"] = pdf_info["start_coordinates"]["left"] + left_offset
                temp_rect_coordinates["y0"] = 0 + header_height_to_remove_for_middle_pages

                #end the rect at the bottom right of the page or at the continue prompt or with a static width
                if use_static_width_from_start_x:
                    temp_rect_coordinates["x1"] = pdf_info["start_coordinates"]["left"] + use_static_width_from_start_x + right_offset
                else:
                    temp_rect_coordinates["x1"] = pdf_info["start_coordinates"]["page_width"] + right_offset

                #check if there is continue_prompt on the same page
                prompt_rect = None
                for coord_dict in pdf_info["continue_coordinates"]:
                    if coord_dict["page"] == i:
                        prompt_rect = coord_dict
                        break

                if prompt_rect:
                    temp_rect_coordinates["y1"] = pdf_info["start_coordinates"]["page_height"] - prompt_rect["top"]
                else:
                    temp_rect_coordinates["y1"] = pdf_info["start_coordinates"]["page_height"]
            
            pdf_info["rect_coordinates"].append(temp_rect_coordinates)

    return pdf_info["rect_coordinates"]

def _save_to_gcs(image, gcs_path):
    """
    Saves an image to Google Cloud Storage (GCS).

    This function handles both PIL Image objects and OpenCV images (NumPy arrays), 
    optimizing them for efficient storage in GCS as PNG files.

    Args:
        image: The image to be saved. It can be either a PIL Image object or a NumPy array 
               representing an OpenCV image.
        gcs_path: The full GCS path where the image should be saved 
                  (e.g., 'gs://your-bucket-name/path/to/image.png').

    Raises:
        TypeError: If the provided `image` is not a supported type (PIL Image or NumPy array).
    """
    
    storage_client = storage.Client()
    bucket_name, blob_name = gcs_path[5:].split("/", 1)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    # Enhanced Image Format Handling
    if isinstance(image, Image.Image):  # PIL Image
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG', optimize=True, quality=95)  # Further reduced quality
        image_bytes = img_byte_arr.getvalue()
    elif isinstance(image, np.ndarray):  # OpenCV image (NumPy array)
        _, image_bytes = cv2.imencode('.png', image, [cv2.IMWRITE_PNG_COMPRESSION, 5])  # Max compression
        image_bytes = image_bytes.tobytes()  # Convert to bytes
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

    blob.upload_from_string(image_bytes, content_type='image/png')  # Explicitly set content type

def _extract_pdf_rectangle(doc, local_temp_dir, output_image_path, rect_coordinates):
    """Extracts and saves images from rectangular regions specified in a PDF document.

    Args:
        doc: A PyMuPDF document object representing the PDF.
        output_image_path: The path where the extracted image(s) should be saved.
                            If it's a GCS path, the image is temporarily saved locally and will be uploaded later.
        rect_coordinates: A list of dictionaries, each containing coordinates ("x0", "y0", "x1", "y1") and the page number ("page")
                          of a rectangular region to extract.

    Returns:
        True if the extraction and saving were successful, False otherwise.
    """

    first_page = doc[0]
    width = first_page.rect.width
    height = first_page.rect.height

    if len(rect_coordinates) == 0:
        return False
        
    first_img = None
    for index, rect in enumerate(rect_coordinates):
        if index == 0:
            first_img = _rectangle_to_img(page = doc[rect["page"] - 1], rect = fitz.Rect(rect["x0"], rect["y0"], rect["x1"], rect["y1"]))
        else:
            temp_img = _rectangle_to_img(page = doc[rect["page"] - 1], rect = fitz.Rect(rect["x0"], rect["y0"], rect["x1"], rect["y1"]))
            first_img = _concat_images(first_img, temp_img)

    # If we are working with GCS, saving the png's into a local temp location. They will be uploaded later to GCS    

    if output_image_path.startswith("gs://"):
        file_name = _get_filename_from_gcs_path(output_image_path)
        output_image_path = file_name
    else:
         splitted = output_image_path.split("/")
         local_temp_dir = "/".join(splitted[:-1]) + "/" + local_temp_dir
         output_image_path = splitted[-1]
    if not os.path.exists(local_temp_dir):
        os.makedirs(local_temp_dir)
        
    output_image_path = f"{local_temp_dir}/{output_image_path}"
    if len(rect_coordinates) == 1:
        first_img.save(output_image_path)
    else:
        cv2.imwrite(output_image_path, first_img)

    return True

def _get_pages_containing(pages, 
                    start_text, 
                    start_after_page = 0,
                    left_offset = 0,
                    right_offset = 0,
                    top_offset = 0,
                    bottom_offset = 0):
    """
    Finds pages within a list of page layouts that contain a specific text and returns their coordinates with optional offsets.

    Args:
        pages: A list of page layout objects representing the pages of a document.
        start_text: The text to search for within the pages.
        start_after_page (int, optional): The page number (0-indexed) after which to start the search. Defaults to 0 (start from the beginning).
        left_offset (int, optional): An offset to add to the left coordinate of the returned rectangle. Defaults to 0.
        right_offset (int, optional): An offset to add to the right coordinate of the returned rectangle. Defaults to 0.
        top_offset (int, optional): An offset to add to the top coordinate of the returned rectangle. Defaults to 0.
        bottom_offset (int, optional): An offset to add to the bottom coordinate of the returned rectangle. Defaults to 0.

    Returns:
        A list of lists, where each inner list contains a dictionary representing the coordinates of a page containing the `start_text`.
        The dictionary has the following keys:
            - "page": The page number (0-indexed).
            - "x0": The left coordinate of the rectangle.
            - "y0": The top coordinate of the rectangle.
            - "x1": The right coordinate of the rectangle.
            - "y1": The bottom coordinate of the rectangle.
    """
    
    rect_coordinates_array = []
    all_components_root = []
    all_elements_root = []
    for page_layout in pages:
        page_found = False
        all_components_array = []
        all_elements_array = []
        if (page_layout.pageid < start_after_page):
            continue
        for index, element in enumerate(page_layout):
            if hasattr(element, 'get_text'):
                if (not page_found and start_text in element.get_text()):
                    temp_rect_coordinates = {}
                    temp_rect_coordinates["page"] = page_layout.pageid
                    temp_rect_coordinates["x0"] = 0 + left_offset
                    temp_rect_coordinates["y0"] = 0 + top_offset
                    temp_rect_coordinates["x1"] = page_layout.width + right_offset
                    temp_rect_coordinates["y1"] = page_layout.height + bottom_offset
                    rect_coordinates_array.append([temp_rect_coordinates])
                    page_found = True
                if (page_found):
                    all_components_array.append(element.get_text())
                    all_elements_array.append({"element": element, "page": page_layout.pageid, "height": page_layout.height})

            if page_found and index == len(page_layout) - 1:
                all_components_root.append(all_components_array)
                all_elements_root.append(all_elements_array)
                
    return rect_coordinates_array, all_components_root, all_elements_root

def _rectangle_to_img(page, rect):
    """Extracts an image from a rectangular region of a PDF page.

    Args:
        page: A PyMuPDF page object representing the PDF page.
        rect: A fitz.Rect object defining the rectangular region to extract.

    Returns:
        A PIL Image object representing the extracted image at 300 DPI resolution.
    """
    # Create a pixmap (image data) from the rectangular region 300 DPI for high quality
    pix = page.get_pixmap(clip=rect, matrix=fitz.Matrix(300/72, 300/72))
    # Convert the pixmap to a PIL Image
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return img

def _concat_images(image1, image2):
    """Vertically concatenates two images, ensuring they have the same width.

    Args:
        image1: The first image (expected as a PIL Image or a NumPy array).
        image2: The second image (expected as a PIL Image or a NumPy array).

    Returns:
        The concatenated image as a NumPy array in OpenCV's BGR format.

    Raises:
        AssertionError: If the images do not have the same width.
    """
    # Ensure images have the same width
    img1_array = np.array(image1)
    img2_array = np.array(image2)

    # Convert RGB to BGR (OpenCV format)
    image1 = cv2.cvtColor(img1_array, cv2.COLOR_RGB2BGR)
    image2 = cv2.cvtColor(img2_array, cv2.COLOR_RGB2BGR)
    assert image1.shape[1] == image2.shape[1], "Images must have the same width"

    # Concatenate vertically
    image3 = cv2.vconcat([image1, image2])
    return image3


def _get_image_filename(pdf_path, suffix):
    """Generates the filename for an extracted image from a PDF.

    This function takes the original PDF path and a suffix, and creates a new filename
    for the extracted image by replacing the ".pdf" extension with the suffix and adding
    a ".png" extension.

    Args:
        pdf_path (str): The path to the original PDF file.
        suffix (str): A string to be added to the filename before the extension.

    Returns:
        str: The generated filename for the extracted image.
    """
    return pdf_path.replace(".pdf", f"_{suffix}.png")

def _get_json_filename(pdf_path, suffix):
    """Generates the filename for an extracted image from a PDF.

    This function takes the original PDF path and a suffix, and creates a new filename
    for the extracted image by replacing the ".pdf" extension with the suffix and adding
    a ".png" extension.

    Args:
        pdf_path (str): The path to the original PDF file.
        suffix (str): A string to be added to the filename before the extension.

    Returns:
        str: The generated filename for the extracted image.
    """
    return pdf_path.replace(".pdf", f"_{suffix}.json")

def _get_pages_and_doc(pdf_path):
    """
    Gets pages and document object from a PDF, handling both local and GCS paths.

    Args:
        pdf_path: The path to the PDF file (local or GCS).

    Returns:
        A tuple containing the list of pages and the document object.
    """
    if _is_gcs_path(pdf_path):
        bucket_name, blob_name = pdf_path[5:].split('/', 1)
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        pdf_contents = blob.download_as_bytes()  # Download once

        doc = fitz.open(stream=pdf_contents, filetype="pdf") 
        pdf_file = io.BytesIO(pdf_contents)  # Create BytesIO from downloaded content
        pdf_file.seek(0) 
        pages = list(extract_pages(pdf_file))
    else:
        doc = fitz.open(pdf_path)
        pages = list(extract_pages(pdf_path))

    return pages, doc

def _filter_elements_by_rectangle(all_elements, rect_coordinates):
    """
    Filters text components from 'all_elements' that fall within the specified rectangles in 'rect_coordinates'.

    Args:
        all_elements: A list of text components, each with keys "element", "page", and "height".
        rect_coordinates: A list of dictionaries representing rectangles, each with keys "page", "x0", "y0", "x1", "y1".

    Returns:
        A list of text components that lie within the specified rectangles, considering the coordinate system differences.
    """

    filtered_elements = []

    for rect in rect_coordinates:
        page_num = rect["page"]
        x0, y0, x1, y1 = rect["x0"], rect["y0"], rect["x1"], rect["y1"]

        # Get the page height directly from the element information
        page_height = next((element["height"] for element in all_elements if element["page"] == page_num), None)
        if page_height is None:
            continue  # Skip if page height couldn't be determined

        y0, y1 = page_height - y1, page_height - y0

        for element_info in all_elements:
            element = element_info["element"]
            element_page = element_info["page"]

            if element_page != page_num:
                continue  # Skip elements not on the target page

            # Check if the element's bounding box intersects with the rectangle
            if (
                element.x0 <= x1 
                and element.x1 >= x0 
                and element.y0 <= y1 
                and element.y1 >= y0 
            ):
                filtered_elements.append(element_info["element"].get_text())

    return [filtered_elements]

def create_images_from_pdf(pdf_path, section_infos, local_temp_dir=".tmp"):
    """
    Extracts images from specified sections of a PDF document.

    This function processes a PDF file and extracts images based on provided section information. 
    It handles both single-page and multi-page sections, allowing flexibility in defining 
    extraction parameters and output image locations.

    Args:
        pdf_path (str): The path to the PDF file (local or GCS).
        section_infos (list): A list of dictionaries, each containing information about a section to extract:
            - "title" (str): The title or identifier for the section.
            - "start_text" (str): The text that marks the beginning of the section.
            - "end_text" (str, optional): The text that marks the end of the section.
            - "continue_prompt" (str, optional): Text indicating a continuation of the section on the next page.
            - "start_after_page" (int, optional): The page number (0-indexed) after which to start the search.
            - "fetch_all_pages_including" (bool, optional): If True, extracts all pages containing the start_text, ignoring end_text. Defaults to False
            - Other optional parameters for fine-tuning extraction (see _find_rectangles_of_section for details).

    Returns:
        A list of dictionaries, each containing information about an extracted image:
            - "path": The path where the image was saved (local or GCS).
            - "image_extracted": True if the extraction was successful, False otherwise.
            - "section_info": The original section information dictionary.
            - "pdf_info": (Optional) Additional information about the PDF section if `fetch_all_pages_including` is False
    """
    output_images = []
    
    pages, doc = _get_pages_and_doc(pdf_path) 

    for section in section_infos:
        if not "fetch_all_pages_including" in section or not section["fetch_all_pages_including"]:
            pdf_info_parameters = copy.copy(section)
            pdf_info_parameters.pop("title", None) 
            pdf_info_parameters.pop("include_start_box", None)
            pdf_info_parameters.pop("include_end_box", None)
            pdf_info_parameters.pop("fetch_end_page_only", None)
            pdf_info_parameters.pop("use_static_width_from_start_x", None)
            pdf_info_parameters.pop("use_static_height_from_start_x", None)
            pdf_info_parameters.pop("header_height_to_remove_for_middle_pages", None)
            pdf_info_parameters.pop("top_offset", None)
            pdf_info_parameters.pop("bottom_offset", None)
            pdf_info_parameters.pop("left_offset", None)
            pdf_info_parameters.pop("right_offset", None)
            pdf_info_parameters.pop("fetch_all_pages_including", None)

            pdf_info = _get_pdf_info(copy.copy(pages), **pdf_info_parameters)

            find_rectangle_parameters = copy.copy(section)
            find_rectangle_parameters.pop("title", None) 
            find_rectangle_parameters.pop("start_text", None) 
            find_rectangle_parameters.pop("end_text", None) 
            find_rectangle_parameters.pop("continue_prompt", None) 
            find_rectangle_parameters.pop("start_after_page", None) 
            find_rectangle_parameters.pop("fetch_all_pages_including", None) 

            rect_coordinates = _find_rectangles_of_section(pdf_info, **find_rectangle_parameters)

            output_image_path = _get_image_filename(pdf_path, section["title"])
            extraction_result = _extract_pdf_rectangle(
                doc=doc,
                local_temp_dir=local_temp_dir,
                output_image_path=output_image_path,
                rect_coordinates=rect_coordinates,
            )
            revised_elements_text = _filter_elements_by_rectangle(pdf_info["all_elements"], rect_coordinates)
            if "all_elements" in pdf_info:
                del pdf_info["all_elements"]
            if "all_components" in pdf_info:
                del pdf_info["all_components"]

            if local_temp_dir not in output_image_path and "gs://" not in output_image_path:
                splitted = output_image_path.split("/")
                output_image_path = "/".join(splitted[:-1]) + "/" + local_temp_dir + "/" + splitted[-1]
            output_images.append({"path" : output_image_path, "image_extracted": extraction_result, "section_info" : section, "pdf_info": pdf_info, "filtered_elements": revised_elements_text})
        else:
            get_pages_parameters = copy.copy(section)
            get_pages_parameters.pop("title", None) 
            get_pages_parameters.pop("end_text", None) 
            get_pages_parameters.pop("continue_prompt", None) 
            get_pages_parameters.pop("fetch_all_pages_including", None) 
            get_pages_parameters.pop("include_start_box", None)
            get_pages_parameters.pop("include_end_box", None)
            get_pages_parameters.pop("fetch_end_page_only", None)
            get_pages_parameters.pop("use_static_width_from_start_x", None)
            get_pages_parameters.pop("use_static_height_from_start_x", None)    

            rect_coordinates_array, all_components_root, all_elements_root = _get_pages_containing(copy.copy(pages), **get_pages_parameters)
            
            for index, rect_coordinates in enumerate(rect_coordinates_array):        
                image_title = section["title"] + "_" + str(index)
                output_image_path = _get_image_filename(pdf_path, image_title)
                extraction_result = _extract_pdf_rectangle(
                    doc=doc,
                    local_temp_dir=local_temp_dir,
                    output_image_path=output_image_path,
                    rect_coordinates=rect_coordinates
                )
                revised_elements_text = _filter_elements_by_rectangle(all_elements_root[index], rect_coordinates)
                
                if local_temp_dir not in output_image_path and "gs://" not in output_image_path:
                    splitted = output_image_path.split("/")
                    output_image_path = "/".join(splitted[:-1]) + "/" + local_temp_dir + "/" + splitted[-1]       
                output_images.append({"path" : output_image_path, "image_extracted": extraction_result, "section_info" : section, "pdf_info" : {}, "filtered_elements": revised_elements_text})
    # Close the PDF document
    doc.close()

    local_file_paths = []
    gcs_paths = []

    for output_image in output_images:
        if output_image["image_extracted"] and _is_gcs_path(output_image["path"]):
            local_file_path = _get_filename_from_gcs_path(output_image["path"])
            local_file_path = f"{local_temp_dir}/{local_file_path}"
            local_file_paths.append(local_file_path)
            gcs_paths.append(output_image["path"])

    splitted = pdf_path.split("/")
    base = "/".join(splitted[:-1])
    if not pdf_path.startswith("gs://"):
        json_path = base + "/" + local_temp_dir + "/" + splitted[-1].replace(".pdf", "_output.json")
    else:
        json_path = local_temp_dir + "/" + splitted[-1].replace(".pdf", ".json")
        local_file_paths.append(json_path)
        gcs_paths.append(base + "/" + splitted[-1].replace(".pdf", "_output.json"))

    with open(json_path, "w") as outfile:
        json.dump(output_images, outfile, indent=4)
    
    # Upload concurrently outside the loop
    _upload_files_concurrently(local_file_paths, gcs_paths)
    # _save_json_to_path(output_images, _get_json_filename(pdf_path, "output"))

    return output_images