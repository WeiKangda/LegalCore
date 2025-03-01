import os
import json
import re
from utils import convert_to_maven_ere_style, convert_to_llm_style

def process_txt_files(input_folder, output_file, processing_function):
    """
    Processes all .txt files in a folder using the provided function and stores the results in a JSONL file.

    Args:
        input_folder (str): Path to the folder containing .txt files.
        output_file (str): Path to the output .jsonl file.
        processing_function (function): Function to process the contents of each .txt file. 
                                         Should take a string (file content) as input and return a dict.
    """
    results = []

    # Iterate through all files in the folder
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".txt"):
            file_path = os.path.join(input_folder, file_name)
            
            # Process the content and collect the result
            try:
                result = processing_function(file_path)
                results.append(result)
            except Exception as e:
                print(f"Error processing {file_name}: {e}")
    
    # Write the results to a JSONL file
    with open(output_file, 'w', encoding='utf-8') as jsonl_file:
        for item in results:
            jsonl_file.write(json.dumps(item) + "\n")

    print(f"Processing complete. Results saved to {output_file}.")

if __name__ == "__main__":
    input_folder = "./data"
    output_file = "./data/data.jsonl"
    process_txt_files(input_folder, output_file, convert_to_llm_style)

    input_folder = "./data"
    output_file = "./data/train.jsonl"
    process_txt_files(input_folder, output_file, convert_to_maven_ere_style)