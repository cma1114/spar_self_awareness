#!/usr/bin/env python3
"""
Debug script to verify tokenization and last token position with chat templates.
"""

from transformers import AutoTokenizer

MODEL_ID = "meta-llama/Llama-3.3-70B-Instruct"

def main():
    print(f"Loading tokenizer for {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Test Prompt
    prompt = "What is 2 + 2?"
    messages = [{"role": "user", "content": prompt}]

    # Apply Chat Template
    print("\n--- Applying Chat Template ---")
    formatted_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    print(f"Formatted Prompt:\n{repr(formatted_prompt)}")

    # Tokenize
    print("\n--- Tokenizing ---")
    inputs = tokenizer(formatted_prompt, return_tensors="pt", add_special_tokens=True)
    input_ids = inputs["input_ids"][0]

    print(f"Total tokens: {len(input_ids)}")

    # Inspect last few tokens
    print("\n--- Last 5 Tokens ---")
    for i in range(max(0, len(input_ids) - 5), len(input_ids)):
        token_id = input_ids[i].item()
        token_str = tokenizer.decode([token_id])
        print(f"Index {i}: ID={token_id:<6} String={repr(token_str)}")

    print("\n--- Verification ---")
    last_token = tokenizer.decode([input_ids[-1].item()])
    print(f"Last token is: {repr(last_token)}")

    if "assistant" in formatted_prompt.lower() and last_token.strip() == "":
        # Llama 3 often ends with a special token or newline after header
        print("Note: Llama 3 chat template often ends with a header token.")

    print("Please manually confirm if this looks like the start of assistant generation.")

if __name__ == "__main__":
    main()
