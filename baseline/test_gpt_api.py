from api_utils import GPT
if __name__ == "__main__":
    model_name = "gpt-4-turbo"  # 替换为实际可用模型名称
    gpt = GPT(model_name, temperature=0.7, api_idx=0)
    test_prompt = [{"role": "user", "content": "Hello, how are you?"}]
    response = gpt(test_prompt, debug=False)
    print("Response:", response)