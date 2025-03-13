from google.cloud import storage
import os
from concurrent.futures import ThreadPoolExecutor

import warnings
warnings.filterwarnings("ignore") 

def download_gcp_folder(bucket_name, folder_path, local_tmp_folder):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        local_few_shot_folder = local_tmp_folder + folder_path
        if os.path.exists(local_few_shot_folder):
            print(f"Local folder '{local_few_shot_folder}' already exists. Skipping few shot file caching.")
            return
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        
        def download_blob(blob):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                local_path = os.path.join(local_tmp_folder, blob.name)
                if blob.name.split("/")[-1] != "": # file
                    blob.download_to_filename(local_path)
                else: # folder
                    os.makedirs(local_path, exist_ok=True)

        blobs = list(bucket.list_blobs(prefix=folder_path))
        with ThreadPoolExecutor() as executor:
            for blob in blobs:
                executor.submit(download_blob, blob)

        print(f"Few shot file caching complete")

bucket_name = "cymbal_invoice_parser"
folder_path = "few_shot_examples/"
local_tmp_folder = "app/.tmp/"

download_gcp_folder(bucket_name, folder_path, local_tmp_folder)