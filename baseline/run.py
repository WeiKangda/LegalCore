import argparse
from event_detection import run_event_detection
from event_coreference import run_event_coreference
from end2end import run_end2end

def map_model_name(simple_name):
    """Map simple model name to full model name."""
    model_mapping = {
        "Llama-3.1-8b-instruct": "meta-llama/Llama-3.1-8B-Instruct",
        "Mistral-7b": "mistralai/Mistral-Nemo-Instruct-2407",
        "QWen": "Qwen/Qwen2.5-14B-Instruct",
        "Phi": "microsoft/Phi-3.5-mini-instruct",
        "Phi-small": "microsoft/Phi-3-small-8k-instruct",
        "GPT-4-Turbo": "GPT-4-Turbo"  # For OpenAI API logic
    }
    return model_mapping.get(simple_name, None)

def main():
    parser = argparse.ArgumentParser(description="Run event-related tasks.")
    parser.add_argument(
        "--setting",
        type=str,
        required=True,
        choices=["event_detection", "event_coreference", "end2end"],
        help="Choose the setting to run: 'event_detection', 'event_coreference', or 'end2end'."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        choices=[
            "Llama-3.1-8b-instruct", "Mistral-7b", "QWen", "Phi", "Phi-small", "GPT-4-Turbo"
        ],
        help="Choose the model: 'Llama-3.1-8b-instruct', 'Mistral-7b', 'QWen', 'Phi', 'Phi-small', 'GPT-4-Turbo'."
    )
    parser.add_argument("--data_path", type=str, default="./annotation_validation/jonathan_annotations/data.jsonl", help="The path to the dataset.")
    parser.add_argument(
        "--inference_mode",
        type=str,
        required=True,
        choices=["zero_shot", "one_shot", "two_shot"],
        help="Choose the inference_mode setting."
    )
    args = parser.parse_args()

    # Map simple model name to full model name
    full_model_name = map_model_name(args.model_name)
    if not full_model_name:
        raise ValueError(f"Invalid model name: {args.model_name}")

    if args.setting == "event_detection":
        print(f"Running event detection with model: {full_model_name}, data: {args.data_path}, inference_mode: {args.inference_mode}")
        run_event_detection(full_model_name, args.data_path, args.inference_mode)

    elif args.setting == "event_coreference":
        print(f"Running event coreference with model: {full_model_name}, data: {args.data_path}, inference_mode: {args.inference_mode}")
        run_event_coreference(full_model_name, args.data_path, args.inference_mode)

    elif args.setting == "end2end":
        print(f"Running end-to-end task with model: {full_model_name}, data: {args.data_path}, inference_mode: {args.inference_mode}")
        run_end2end(full_model_name, args.data_path, args.inference_mode)

if __name__ == "__main__":
    main()
