import os
import re
import sys
import torch
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import KFold
from torch.utils.data import Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, TrainingArguments, Trainer
from post_processing.utils import load_jsonl, extract_predicted_trigger_words, extract_gold_trigger_words
from eval import calculate_micro_macro_f1_supervised, save_metrics_to_file

class CustomDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        input_encodings = self.tokenizer(
            example["input"], truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt"
        )
        target_encodings = self.tokenizer(
            example["target"], truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt"
        )
        return {
            "input_ids": input_encodings["input_ids"].squeeze(),
            "attention_mask": input_encodings["attention_mask"].squeeze(),
            "labels": target_encodings["input_ids"].squeeze(),
        }

if __name__ == "__main__":
    all_data = load_jsonl("./data/data.jsonl")
    raw_dataset = []

    for data in all_data:
        input_text = data["text"]
        target_text = data["singleton_text"]
        input_text_sentences = re.split(r'(?<=[.!?])\s+', input_text.strip())
        target_text_sentences = re.split(r'(?<=[.!?])\s+', target_text.strip())
        assert len(input_text_sentences) == len(target_text_sentences)
        for i in range(len(input_text_sentences)):
            datapoint = {}
            datapoint["input"] = input_text_sentences[i]
            datapoint["target"] = target_text_sentences[i]
            raw_dataset.append(datapoint)
        
    model_name = "t5-large"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    new_tokens = ["{", "}"]
    tokens_to_add = [token for token in new_tokens if token not in tokenizer.get_vocab()]
    tokenizer.add_tokens(tokens_to_add)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    all_gold = []
    all_predicted = []
    total_true_positives, total_false_positives, total_false_negatives = 0, 0, 0
    for fold, (train_idx, val_idx) in enumerate(kf.split(raw_dataset)):
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        model.resize_token_embeddings(len(tokenizer))

        print(f"Starting fold {fold + 1}...")
        train_data = [raw_dataset[i] for i in train_idx]
        val_data = [raw_dataset[i] for i in val_idx]
        print(f"# of training data: {len(train_data)}, # of validation data: {len(val_data)}")

        train_dataset = CustomDataset(train_data, tokenizer)
        val_dataset = CustomDataset(val_data, tokenizer)

        training_args = TrainingArguments(
            output_dir=f"./{model_name}-detection-results-fold-{fold + 1}",
            evaluation_strategy="epoch",
            per_device_train_batch_size=4,
            per_device_eval_batch_size=1,
            num_train_epochs=5,
            save_strategy="epoch",
            logging_dir=f"./{model_name}-detection-logs-fold-{fold + 1}",
            learning_rate=1e-4,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset
        )

        trainer.train()

        # Predict and decode predictions
        step = 128
        for i in tqdm(range(0, len(val_data), step)):
            test_dataset = CustomDataset(val_data[i:i+step], tokenizer)
            predictions = trainer.predict(test_dataset)
            decoded_preds = tokenizer.batch_decode(np.argmax(predictions.predictions[0], axis=-1), skip_special_tokens=True)
            #print(decoded_preds)
            decoded_labels = tokenizer.batch_decode(predictions.label_ids, skip_special_tokens=True)
            #print(decoded_labels)

            for decoded_pred in decoded_preds:
                all_predicted.append(extract_predicted_trigger_words(decoded_pred))

            for decoded_label in decoded_labels:
                all_gold.append(extract_gold_trigger_words(decoded_label))
        
        result = calculate_micro_macro_f1_supervised(all_gold, all_predicted)
        true_positives, false_positives, false_negatives = result['total_true_positives'], result['total_false_positives'], result['total_false_negatives']
        total_true_positives += true_positives
        total_false_positives += false_positives
        total_false_negatives += false_negatives

        print(f"Finished fold {fold + 1}")
    
    micro_precision = total_true_positives / (total_true_positives + total_false_positives) if (total_true_positives + total_false_positives) > 0 else 0
    micro_recall = total_true_positives / (total_true_positives + total_false_negatives) if (total_true_positives + total_false_negatives) > 0 else 0
    micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0

    # Print the results
    print(f"Precision: {micro_precision}")
    print(f"Recall: {micro_recall}")
    print(f"F-1: {micro_f1}")

    with open(f'{model_name}-detection-results.txt', 'w') as file:
        file.write(f"Precision: {micro_precision}\n")
        file.write(f"Recall: {micro_recall}\n")
        file.write(f"F-1: {micro_f1}")
