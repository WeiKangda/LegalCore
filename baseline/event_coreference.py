import os
import sys
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, pipeline, AutoModelForCausalLM
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from post_processing.utils import load_jsonl, append_to_jsonl, process_coreference, create_coreference_clusters, replace_elements_with_mentions, mentions_to_clusters
from eval import save_metrics_to_file, calculate_micro_macro_muc, calculate_micro_macro_b3, calculate_micro_macro_ceaf_e, calculate_micro_macro_blanc

def generate_response(model, tokenizer, prompt):
    msgs = [
        {
            "role": "system",
            "content": "You are a helpful assistant.",
        },
        {
            "role": "user",
            "content": "Please analyze the following text to detect all coreference relations among events. \
                        Two events have a coreference relation if they refer to the same event in space and time. \
                        Coreference relation is symmetrical, i.e., non-directional: If A coreference B, then B coreference A. \
                        It is also transitive: If A coreference B and B coreference C, then A coreference C. \
                        An event should only be linked with one of its nearest antecedents occurring before itself. \
                        There is no need to link with multiple antecedents as this information is redundant. \
                        All events are denoted as {EXX trigger_word} in the text. Format your response as: 'EXX COREFERENCE EXX'. \
                        Hint: Coreferential event mentions usually have the same trigger_word.\
                        If multiple coreference relations exist, list each relation on a new line. \
                        If no coreference relation is detected, simply return 'None'. Please respond concisely and directly to the point, avoiding unnecessary elaboration or verbosity.\
                        Text: " + str(prompt) + "Response:"
        }
    ]

    input_ids = tokenizer.apply_chat_template(
        msgs,
        padding=True,
        return_tensors="pt",
    ).to("cuda" if torch.cuda.is_available() else "cpu")
    # Generate text from the model
    output = model.generate(
        input_ids=input_ids,
        max_new_tokens=1024,
    )
    prompt_length = input_ids.shape[1]
    response = tokenizer.decode(output[0][prompt_length:], skip_special_tokens=True)
    return response

def event_coreference(model, tokenizer, data):
    result = {"id": data["id"]}
    text = data['singleton_text']
    mention_list = data["events"]
    response = generate_response(model, tokenizer, text)
    coreference_tuples = process_coreference(response)
    clusters = create_coreference_clusters(coreference_tuples)
    result["clusters"] = replace_elements_with_mentions(clusters, mention_list)

    return result

def event_coreference_end2end(model, tokenizer, data):
    result = {"id": data["id"]}
    text = data['text_with_predicted_event']
    mention_list = data["events"]
    response = generate_response(model, tokenizer, text)
    coreference_tuples = process_coreference(response)
    clusters = create_coreference_clusters(coreference_tuples)
    result["clusters"] = replace_elements_with_mentions(clusters, mention_list)

    return result

if __name__ == "__main__":
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    print(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(model_name, pad_token_id=tokenizer.eos_token_id)
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    all_data = load_jsonl("./annotation_validation/jonathan_annotations/data.jsonl")
    output_file = "./annotation_validation/jonathan_annotations/coreference_output.jsonl"
    final_result_file = "./annotation_validation/jonathan_annotations/coreference_result.txt"

    all_predicted = [] 
    all_gold = []
    for data in tqdm(all_data):
        result = event_coreference(model, tokenizer, data)
        append_to_jsonl(output_file, result)
        all_predicted.append(result["clusters"])
        all_gold.append(mentions_to_clusters(data["events"]))
        print("Gold mentions:" + str(mentions_to_clusters(data["events"])))
        print("Predicted mentions:" + str(result["clusters"]))
        print("########################")

    final_result = {}
    muc = calculate_micro_macro_muc(all_predicted, all_gold)
    print("MUC:" + str(muc))
    b3 = calculate_micro_macro_b3(all_predicted, all_gold)
    print("B^3:" + str(b3))
    ceaf_e = calculate_micro_macro_ceaf_e(all_predicted, all_gold)
    print("CEAF_e:" + str(ceaf_e))
    blanc = calculate_micro_macro_blanc(all_predicted, all_gold)
    print("BLANC:" + str(blanc))

    final_result["MUC"] = muc
    final_result["B^3"] = b3
    final_result["CEAF_e"] = ceaf_e
    final_result["BLANC"] = blanc
    #print(final_result)
    save_metrics_to_file(final_result, final_result_file)