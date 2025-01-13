import os
from api_utils import GPT


def test_gpt_api():
    print("Testing GPT API with multiple calls...")

    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    gpt_model = GPT(model_name=model_name, temperature=0.7)

    # Example prompts for testing rate limits
    prompts = [
        "What is the capital of France?",
        "Who wrote 'Pride and Prejudice'?",
        "What is the square root of 144?",
        "Explain the theory of relativity in simple terms.",
        "What is the current population of the world?"
    ]

    for i, prompt in enumerate(prompts):
        try:
            response = gpt_model.eval_call(prompt, n=1, debug=False)
            print(f"Prompt {i + 1}: {prompt}")
            print(f"Response {i + 1}: {response}")
        except Exception as e:
            print(f"Error on prompt {i + 1}: {e}")


if __name__ == "__main__":
    test_gpt_api()
