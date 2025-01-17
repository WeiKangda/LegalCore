from event_detection import load_jsonl,extract_mentions
import tqdm
data_path="./annotation_validation/jonathan_annotations/data.jsonl"
all_data = load_jsonl(data_path)
all_gold=[]
for data in tqdm(all_data):
    all_gold.append(extract_mentions(data["events"]))

print(all_gold[0])