import argparse
from event_detection import run_event_detection
from event_coreference import run_event_coreference
from end2end import run_end2end

def map_model_name(simple_name):
    """Map simple model name to full model name and indicate if it is a commercial model."""
    model_mapping = {
        "Llama-3.1-8b-instruct": ("meta-llama/Llama-3.1-8B-Instruct", False),
        "Mistral-7b": ("mistralai/Mistral-7B-Instruct-v0.2", False),
        "Mistral-Nemo": ("mistralai/Mistral-Nemo-Instruct-2407",False),
        "QWen-7b": ("Qwen/Qwen2.5-7B-Instruct", False),
        "QWen-14b": ("Qwen/Qwen2.5-14B-Instruct",False),
        "Phi": ("microsoft/Phi-3.5-mini-instruct", False),
        "Phi-small": ("microsoft/Phi-3-small-8k-instruct", False),
        "deepseek_llama-8b": ("deepseek-ai/DeepSeek-R1-Distill-Llama-8B", False),
        "deepseek_Qwen-14b": ("deepseek-ai/DeepSeek-R1-Distill-Qwen-14B", False),
        "GPT-4-Turbo": ("gpt-4-turbo", True),  # Mark as a commercial model
        "Gemini-2": ("gemini-2", True)
    }
    return model_mapping.get(simple_name, (None, None))

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
            "Llama-3.1-8b-instruct", "Mistral-7b","Mistral-Nemo", "QWen-7b", "QWen-14b", "Phi", "Phi-small", "GPT-4-Turbo", "deepseek_llama-8b", "deepseek_Qwen-14b"
        ],
        help="Choose the model"
    )
    parser.add_argument("--data_path", type=str, default="./data/data.jsonl", help="The path to the dataset.")
    parser.add_argument("--output_path", type=str)

    parser.add_argument(
        "--inference_mode",
        type=str,
        required=True,
        choices=["zero_shot", "one_shot", "two_shot"],
        help="Choose the inference_mode setting."
    )
    args = parser.parse_args()

    # Map simple model name to full model name
    full_model_name, is_commercial = map_model_name(args.model_name)
    # if "GPT" in args.model_name:
    #     print("not implemented yet")
    #     return
    if not full_model_name:
        raise ValueError(f"Invalid model name: {args.model_name}")

    if args.setting == "event_detection":
        print(f"Running event detection with model: {full_model_name}, data: {args.data_path}, inference_mode: {args.inference_mode}")
        run_event_detection(full_model_name,is_commercial, args.data_path,args.output_path, args.inference_mode)

    elif args.setting == "event_coreference":
        print(f"Running event coreference with model: {full_model_name}, data: {args.data_path}, inference_mode: {args.inference_mode}")
        run_event_coreference(full_model_name,is_commercial, args.data_path,args.output_path, args.inference_mode)

    elif args.setting == "end2end":
        print(f"Running end-to-end task with model: {full_model_name}, data: {args.data_path}, inference_mode: {args.inference_mode}")
        run_end2end(full_model_name,is_commercial, args.data_path,args.output_path, args.inference_mode)

if __name__ == "__main__":
    main()
