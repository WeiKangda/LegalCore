import os
import sys
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, pipeline, AutoModelForCausalLM
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from baseline.event_coreference import event_coreference_end2end
from baseline.event_detection import event_detection
from post_processing.utils import load_jsonl, append_to_jsonl, process_coreference, create_coreference_clusters, replace_elements_with_mentions, mentions_to_clusters, extract_mentions
from pre_processing.utils import replace_elements
from pre_processing.utils import generate_paths
from eval import save_metrics_to_file, calculate_micro_macro_muc, calculate_micro_macro_b3, calculate_micro_macro_ceaf_e, calculate_micro_macro_blanc, calculate_micro_macro_f1
import api_utils
def replace_elements_by_index(lst, replacements):
    """
    Replaces elements in a list based on a list of tuples containing 
    (index, trigger_word) using the provided replace_elements function.

    Parameters:
        lst (list): The list of elements to modify.
        replacements (list of tuples): A list of tuples where each tuple contains 
                                        (index, trigger_word).
        singleton_index (int): The starting index to use for singleton formatting.

    Returns:
        list: The modified list.
    """
    singleton_index = 0
    for idx, trigger_word in replacements:
        #print(idx, idx+len(trigger_word.split(" "))-1, trigger_word)
        if 0 <= idx < len(lst):
            lst[idx] = trigger_word  # Replace element at index
            lst = replace_elements(lst, idx, idx+len(trigger_word.split(" "))-1, singleton_index)
            singleton_index += 1
        else:
            raise IndexError(f"Index {idx} is out of range for the list.")
    return " ".join(lst)

# if __name__ == "__main__":
def run_end2end(model_name,is_commercial,data_path,output_path,inference_mode):
    # Load model and tokenizer
    # model_name = "meta-llama/Llama-3.1-8B-Instruct"
    print(model_name)
    if is_commercial:
        tokenizer = None
        model = api_utils.load_model(model_name, 0)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name,trust_remote_code=True)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model = AutoModelForCausalLM.from_pretrained(model_name, pad_token_id=tokenizer.eos_token_id, device_map="auto", trust_remote_code=True)
        model.eval()

    # Load data
    all_data = load_jsonl(data_path)
    # Set the output and result file path for event detection
    task_name="end2end_detection"
    base_dir=output_path
    output_file, final_result_file = generate_paths(base_dir, task_name, model_name, inference_mode)


    # Event prediction
    all_predicted = [] 
    all_gold = []
    for data in tqdm(all_data):
        result = event_detection(model,is_commercial, tokenizer, data,inference_mode)
        append_to_jsonl(output_file, result)
        all_gold.append(extract_mentions(data["events"]))
        all_predicted.append(result["mentions"])
        print("Gold mentions:" + str(extract_mentions(data["events"])))
        print("Predicted mentions:" + str(result["mentions"]))
        print("########################")
        # Add the predicted events to plain text
        data['text_with_predicted_event'] = replace_elements_by_index(data['tokens'], result["mentions"])
    
    # Calculate and save scores for event detection
    final_result = calculate_micro_macro_f1(all_predicted, all_gold)
    print(final_result)
    save_metrics_to_file(final_result, final_result_file)

    # Set the output and result file path for event coreference
    task_name = "end2end_coreference"
    output_file, final_result_file = generate_paths(base_dir, task_name, model_name, inference_mode)


    # Event coreference
    all_predicted = [] 
    all_gold = []
    for data in tqdm(all_data):
        result = event_coreference_end2end(model,is_commercial, tokenizer, data, inference_mode)
        append_to_jsonl(output_file, result)
        all_predicted.append(result["clusters"])
        all_gold.append(mentions_to_clusters(data["events"]))
        print("Gold mentions:" + str(mentions_to_clusters(data["events"])))
        print("Predicted mentions:" + str(result["clusters"]))
        print("########################")

    # Calculate and save scores for event coreference
    final_result = {}
    muc = calculate_micro_macro_muc(all_gold, all_predicted)
    print("MUC:" + str(muc))
    b3 = calculate_micro_macro_b3(all_gold, all_predicted)
    print("B^3:" + str(b3))
    ceaf_e = calculate_micro_macro_ceaf_e(all_gold, all_predicted)
    print("CEAF_e:" + str(ceaf_e))
    blanc = calculate_micro_macro_blanc(all_gold, all_predicted)
    print("BLANC:" + str(blanc))

    final_result["MUC"] = muc
    final_result["B^3"] = b3
    final_result["CEAF_e"] = ceaf_e
    final_result["BLANC"] = blanc
    save_metrics_to_file(final_result, final_result_file)