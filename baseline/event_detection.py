import re
import os
import sys
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, pipeline, AutoModelForCausalLM
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from post_processing.utils import extract_spans_and_triggers, update_offsets, load_jsonl, extract_mentions, append_to_jsonl
from pre_processing.utils import generate_paths
from eval import calculate_micro_macro_f1, save_metrics_to_file
import api_utils
def generate_response(model,is_commercial, tokenizer, prompt):
    msgs = [
        {
            "role": "system",
            "content": "You are a helpful assistant.",
        },
        {
            "role": "user",
            "content": f"Please analyze the following text to detect all events. We define an event as any occurrence, action, process or event state \
                        which deserves a place upon a timeline, and could have any syntactic realization as verbs, nominalizations, nouns, or even adjectives. \
                        Please respond concisely and directly to the point, avoiding unnecessary elaboration or verbosity.\
                        If an event is detected, kindly provide its span as index and trigger word/phrase, formatting your response as: \
                        Span: event span index \
                        Trigger: trigger word/phrase \
                        Span: event span index \
                        Trigger: trigger word/phrase \
                        ...\
                        If no event is identified, simply return None. Text: {prompt}" + "Response:"
        }
    ]
    if is_commercial:
        response = model.eval_call(prompt, debug=False)
    else:
        input_ids = tokenizer.apply_chat_template(
            msgs,
            padding=True,
            return_tensors="pt",
        )
        # Generate text from the model
        output = model.generate(
            input_ids=input_ids,
            max_new_tokens=1024,
        )
        prompt_length = input_ids.shape[1]
        response = tokenizer.decode(output[0][prompt_length:], skip_special_tokens=True)
    return response

def event_detection(model,is_commercial, tokenizer, data):
    cumulative_offsets = 0
    sentences = data['sentences']
    result = {"id": data["id"], "mentions":[]}
    for sentence in sentences:
        length = len(re.split(r'\s+', sentence))
        response = generate_response(model,is_commercial, tokenizer, sentence)
        spans_and_triggers = extract_spans_and_triggers(response)
        processed_spans_and_triggers = update_offsets(spans_and_triggers, sentence)
        for processed_spans_and_trigger in processed_spans_and_triggers:
            if processed_spans_and_trigger['offset'] == -1:
                continue
            result["mentions"].append((processed_spans_and_trigger['offset'] + cumulative_offsets, processed_spans_and_trigger['trigger_word'])) 
        cumulative_offsets += length

    return result

def run_event_detection(model_name,is_commercial,data_path,output_path,inference_mode):
    print(model_name)
    if is_commercial:
        tokenizer=None
        model=api_utils.load_model(model_name,0)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model = AutoModelForCausalLM.from_pretrained(model_name, pad_token_id=tokenizer.eos_token_id,device_map="auto")
        model.eval()

    #prompt = "Sponsor  acknowledges  that Sponsor shall  cooperate  with the Concessionaire  regarding  logistics and management of the Sponsor's food products, and appropriate storage and dispensation of the food products."
    #response = generate_response(model, tokenizer, prompt)
    #print(response)
    # all_data = load_jsonl("./annotation_validation/jonathan_annotations/data.jsonl")
    # #data = all_data[0]
    # #result = event_detection(model, tokenizer, data)
    # #print(result)
    # output_file = "./annotation_validation/jonathan_annotations/detection_output.jsonl"
    # final_result_file = "./annotation_validation/jonathan_annotations/detection_result.txt"
    all_data = load_jsonl(data_path)
    # Set the output and result file path for event detection
    task_name = "detection"
    base_dir = output_path
    output_file, final_result_file = generate_paths(base_dir, task_name, model_name, inference_mode)

    all_predicted = []
    all_gold = []
    for data in tqdm(all_data):
        result = event_detection(model,is_commercial, tokenizer, data)
        append_to_jsonl(output_file, result)
        all_gold.append(extract_mentions(data["events"]))
        all_predicted.append(result["mentions"])
        print("Gold mentions:" + str(extract_mentions(data["events"])))
        print("Predicted mentions:" + str(result["mentions"]))
        print("########################")
        #break
    
    final_result = calculate_micro_macro_f1(all_predicted, all_gold)
    print(final_result)
    save_metrics_to_file(final_result, final_result_file)
    