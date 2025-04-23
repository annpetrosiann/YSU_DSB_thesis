import os
import pdfplumber
import json
from pathlib import Path
import logging
import time

# Setting up logging
logging.basicConfig(
    #filename="pdf_processing.log",
    level=logging.INFO,
    format="%(asctime)s:%(levelname)s:%(message)s",
)

# Root directory where the folders with PDFs are located
root_dir = 'change_with_your_input_path'
output_base = 'change_with_your_output_path'

# Ensuring the base directory exists
os.makedirs(output_base, exist_ok=True)
logging.info(
    f"Created or confirmed the existence of the output base directory at {output_base}"
)

# Start timing the process
start_time = time.time()

# Walk through the directory tree
for subdir, dirs, files in os.walk(root_dir):
    for filename in files:
        # Check if the file is a PDF
        if filename.endswith(".pdf"):
            file_path = os.path.join(subdir, filename)
            logging.info(f"Processing file {filename}")
            try:
                # Open the PDF file
                with pdfplumber.open(file_path) as pdf:
                    full_text = ""
                    # Extract text from each page
                    for page in pdf.pages:
                        text = page.extract_text()
                        if text:  # Check if text was extracted
                            full_text += text + "\n"

                # Prepare the JSON data
                data_to_save = {
                    "title": Path(filename).stem + ".pdf",  # Filename without extension
                    "data": full_text,
                }

                # Construct a new folder path based on the original folder structure
                relative_subdir = Path(subdir).relative_to(root_dir)
                new_folder_path = os.path.join(output_base, relative_subdir)
                os.makedirs(new_folder_path, exist_ok=True)

                # Define the output JSON file path
                json_filename = Path(filename).stem + ".json"
                json_file_path = os.path.join(new_folder_path, json_filename)

                # Save the extracted data as JSON
                with open(json_file_path, "w") as json_file:
                    json.dump(data_to_save, json_file, indent=4, ensure_ascii=False)
                    logging.info(
                        f"Data for {filename} saved successfully to {json_file_path}"
                    )

            except Exception as e:
                logging.error(f"Failed to process file {filename}. Error: {str(e)}")

            print(f"Data for {filename} saved to {json_file_path}")

# Calculate and log the total processing time
end_time = time.time()
total_time = end_time - start_time
logging.info(f"Total processing time: {total_time:.2f} seconds")
print(f"Total processing time: {total_time:.2f} seconds")