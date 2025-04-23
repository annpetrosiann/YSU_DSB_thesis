import os
import json
from pathlib import Path
import time

## same bank jsons -->  one json
def combine_json_files(root_dir, output_dir):
    start_time = time.time()

    # Create the base directory for combined data if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Iterate over all directories and subdirectories in the root directory
    for subdir, dirs, _ in os.walk(root_dir):
        combined_data = []
        # Define the new subfolder path in the output directory
        new_subdir = Path(output_dir) / Path(subdir).name
        new_subdir.mkdir(exist_ok=True)  # Create the subdirectory if it doesn't exist

        # Define the output file path in the new subfolder
        output_file_path = new_subdir / f"{Path(subdir).name}.json"

        # Process each JSON file in the original directory
        for file in Path(subdir).glob('*.json'):
            with open(file, 'r') as json_file:
                data = json.load(json_file)
                combined_data.append(data)

        # Write combined data to a new JSON file in the new directory
        if combined_data:  # Check if there is any data to write
            with open(output_file_path, 'w') as json_out:
                json.dump(combined_data, json_out, indent=4, ensure_ascii=False)
                print(f"Combined JSON saved to {output_file_path}")

    elapsed_time = time.time() - start_time
    print(f"Time taken to combine JSON files: {elapsed_time:.2f} seconds")


## bank jsons -> final json
def combine_jsons(folder_path):
    start_time = time.time()

    combined_data = []

    # Walk through the directory
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # Check if the file is a JSON file
            if file.endswith('.json'):
                full_path = os.path.join(root, file)
                # Open and read the JSON file
                with open(full_path, 'r') as f:
                    data = json.load(f)
                    combined_data.append(data)

    # Write the combined data to a new JSON file
    with open('combined_data.json', 'w') as f:
        json.dump(combined_data, f, indent=4)

    elapsed_time = time.time() - start_time
    print(f"Time taken to combine all JSONs into one: {elapsed_time:.2f} seconds")


# Root directory where the folders with JSON files are located
root_dir = 'change_with_your_input_path'
output_dir = 'change_with_your_output_path'
combine_json_files(root_dir, output_dir)
combine_jsons(output_dir)
