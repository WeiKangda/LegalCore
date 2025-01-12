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
    '''all_data = load_jsonl("./annotation_validation/jonathan_annotations/data.jsonl")
    data = all_data[0]
    lst = data['tokens']
    replacements = [[20, "made"], [23, "April"], [30, "by"], [31, "and"], [32, "between"], [63, "24"], [65, "Halle"], [67, "Germany"], [73, "referred"], [88, "wish"], [90, "collaborate"], [93, "discovery"], [95, "development"], [102, "treatment"], [93, "discovery"], [95, "development"], [102, "treatment"], [179, "For"], [187, "ownership"], [230, "possession"], [236, "elect"], [237, "appoint"], [238, "direct"], [240, "remove"], [263, "otherwise"], [299, "Hit"], [290, "determined"], [297, "meet"], [320, "Invention"], [328, "discovery"], [330, "proprietary"], [338, "made"], [340, "generated"], [359, "performing"], [361, "Research"], [362, "Plan"], [393, "means"], [397, "attached"], [398, "hereto"], [397, "attached"], [405, "means"], [424, "determination"], [431, "evoking"], [442, "protease"], [445, "coronavirus"], [451, "ribonuclease"], [454, "interaction"], [458, "mutants"], [460, "variants"], [463, "molecule"], [465, "component"], [475, "truncated"], [458, "mutants"], [460, "variants"], [463, "molecule"], [465, "component"], [590, "will"], [596, "set"], [596, "set"], [622, "identifying"], [629, "modulate"], [640, "provide"], [642, "deliverables"], [643, "set"], [644, "forth"], [654, "obtain"], [656, "authorizations"], [658, "approvals"], [660, "licenses"], [661, "required"], [706, "may"], [708, "be"], [708, "be"], [713, "written"], [725, "desire"], [737, "negotiate"], [743, "amendment"], [765, "enter"], [775, "will"], [786, "updates"], [794, "via"], [795, "teleconference"], [797, "or"], [796, "videoconference"], [797, "or"], [799, "and"], [803, "make"], [806, "available"], [812, "discuss"], [813, "and"], [814, "provide"], [823, "Delivery"], [824, "of"], [826, "In"], [838, "deliver"], [843, "generated"], [848, "since"], [856, "will"], [862, "request"], [872, "respond"], [875, "requests"], [897, "perform"], [906, "generate"], [921, "completion"], [944, "select"], [958, "providing"], [988, "Selection"], [992, "designated"], [1000, "Commencing"], [1002, "selection"], [1026, "development"], [1028, "manufacture"], [1031, "commercialization"], [1050, "Following"], [1056, "have"], [1074, "included"], [1105, "use"], [1131, "for"], [1131, "for"], [1131, "for"], [1155, "finds"], [1161, "use"], [1174, "notify"], [1181, "has"], [1186, "negotiation"], [1195, "notification"], [1232, "may"], [1239, "perform"], [1263, "will"], [1253, "and"], [1277, "enter"], [1279, "a"], [1301, "prohibiting"], [1420, "includes"], [1422, "screening"], [1424, "including"], [1428, "of"], [1422, "screening"], [1428, "of"], [1436, "against"], [1443, "will"], [1445, "complete"], [1447, "accurate"], [1448, "records"], [1451, "activities"], [1452, "performed"], [1467, "made"], [1469, "generated"], [1466, "Inventions"], [1479, "performance"], [1504, "will"], [1513, "inspect"], [1504, "will"], [1518, "request"], [1520, "provide"], [1533, "exercise"], [1535, "performance"], [1540, "rights"], [1542, "obligations"], [1550, "disclosed"], [1568, "will"], [1578, "following"], [1579, "expiration"], [1581, "termination"], [1592, "required"], [1604, "represents"], [1621, "has"], [1637, "investigations"], [1639, "claims"], [1641, "proceedings"], [1647, "pending"], [1649, "threatened"], [1667, "will"], [1685, "debarred"], [1689, "agrees"], [1691, "undertakes"], [1694, "notify"], [1709, "debarred"], [1714, "initiated"], [1726, "debarment"], [1728, "initiation"], [1731, "occurs"], [1726, "debarment"], [1751, "performance"], [1762, "pay"], [1762, "pay"], [1762, "pay"], [1762, "pay"], [1762, "pay"], [1762, "pay"], [1762, "pay"], [1762, "pay"], [1762, "pay"], [1830, "reimburse"], [1830, "reimburse"], [1928, "OntoChem"], [1931, "a"], [1935, "acquire"], [1946, "will"], [1947, "agree"], [1955, "OntoChem"], [1962, "have"], [1963, "synthesized"], [1962, "have"], [1974, "will"], [1975, "agree"], [1980, "Biological"], [1989, "have"], [1991, "test"], [2002, "will"], [2003, "agree"], [2029, "payable"], [2034, "following"], [2035, "each"], [2036, "anniversary"], [2039, "date"], [2042, "Selection"], [2043, "Notice"], [2044, "until"], [2045, "five"], [2047, "years"], [2048, "after"], [2055, "first"], [2051, "commercial"], [2052, "sale"], [2053, "of"], [2055, "first"], [2056, "product"], [2057, "incorporating"], [2058, "a"], [2059, "compound"], [2060, "from"], [2061, "such"], [2062, "Lead"], [2073, "Scaffold"], [2064, "subject"], [2070, "to"], [2066, "Section"], [2068, "with"], [2069, "respect"], [2070, "to"], [2071, "any"], [2072, "Terminated"], [2073, "Scaffold"], [2075, "defined"], [2078, "Milestone"], [2082, "pay"], [2086, "milestone"], [2087, "payment"], [2091, "Dollars"], [2096, "following"], [2103, "in"], [2109, "for"], [2124, "Payments"], [2129, "made"], [2128, "be"], [2129, "made"], [2133, "by"], [2134, "wire"], [2135, "transfer"], [2136, "of"], [2139, "funds"], [2152, "to"], [2141, "such"], [2142, "bank"], [2143, "account"], [2144, "as"], [2145, "designated"], [2146, "in"], [2147, "writing"], [2148, "by"], [2149, "OntoChem"], [2150, "from"], [2151, "time"], [2152, "to"], [2151, "time"], [2192, "will"], [2194, "complete"], [2199, "accounting"], [2201, "related"], [2206, "incurred"], [2214, "will"], [2219, "during"], [2220, "regular"], [2221, "business"], [2222, "hours"], [2223, "upon"], [2224, "reasonable"], [2225, "notice"], [2226, "by"], [2227, "Anixa"], [2228, "or"], [2229, "its"], [2230, "duly"], [2231, "authorized"], [2232, "representative"], [2233, "at"], [2227, "Anixa"], [2235, "expense"], [2236, "for"], [2237, "three"], [2239, "years"], [2240, "following"], [2244, "the"], [2242, "end"], [2243, "of"], [2244, "the"], [2245, "calendar"], [2246, "year"], [2247, "in"], [2248, "which"], [2249, "such"], [2250, "expenses"], [2251, "are"], [2252, "invoiced"], [2332, "will"], [2322, "terminated"], [2341, "completion"], [2344, "Research"], [2338, "Effective"], [2354, "terminated"], [2354, "terminated"], [2366, "notice"]]
    print(data['text'])
    text_with_predicted_event = replace_elements_by_index(lst, replacements)
    print(text_with_predicted_event)'''

    # Load model and tokenizer
    # model_name = "meta-llama/Llama-3.1-8B-Instruct"
    print(model_name)
    if is_commercial:
        tokenizer = None
        model = api_utils.load_model(model_name, 0)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model = AutoModelForCausalLM.from_pretrained(model_name, pad_token_id=tokenizer.eos_token_id,device_map="auto")
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
        result = event_detection(model,is_commercial, tokenizer, data)
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
        result = event_coreference_end2end(model,is_commercial, tokenizer, data)
        append_to_jsonl(output_file, result)
        all_predicted.append(result["clusters"])
        all_gold.append(mentions_to_clusters(data["events"]))
        print("Gold mentions:" + str(mentions_to_clusters(data["events"])))
        print("Predicted mentions:" + str(result["clusters"]))
        print("########################")

    # Calculate and save scores for event coreference
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
    save_metrics_to_file(final_result, final_result_file)