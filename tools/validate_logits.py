import argparse
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

def generate_reference_logits(model_path, prompt, output_dir):
    print(f"Loading HF model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    
    print(f"Tokenizing prompt: '{prompt}'")
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids
    print(f"Input IDs: {input_ids.tolist()}")
    
    print("Running forward pass...")
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits
        
    # We want the logits for the last token in the sequence (for the next token prediction)
    # Shape of logits: [batch_size, seq_len, vocab_size]
    last_token_logits = logits[0, -1, :].numpy()
    
    print(f"Logits shape: {last_token_logits.shape}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save input ids to a text file for the C++ test to read
    with open(os.path.join(output_dir, "input_ids.txt"), "w") as f:
        f.write(" ".join(map(str, input_ids[0].tolist())))
        
    # Save the logits as a raw binary file (float32)
    logits_file = os.path.join(output_dir, "hf_logits.bin")
    last_token_logits.astype(np.float32).tofile(logits_file)
    print(f"Saved reference logits to {logits_file}")
    
    # Also save the top 5 predicted tokens for quick debugging
    top_5_idx = np.argsort(last_token_logits)[-5:][::-1]
    top_5_vals = last_token_logits[top_5_idx]
    
    print("\nTop 5 predictions (HF):")
    for i, (idx, val) in enumerate(zip(top_5_idx, top_5_vals)):
        token = tokenizer.convert_ids_to_tokens(int(idx))
        print(f"{i+1}. Token ID: {idx}, Token: '{token}', Logit: {val:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate reference logits using HuggingFace")
    parser.add_argument("--model_path", type=str, required=True, help="Path to HF model")
    parser.add_argument("--prompt", type=str, default="Hello, how are you?", help="Input prompt")
    parser.add_argument("--output_dir", type=str, default="validation_data", help="Output directory")
    args = parser.parse_args()
    
    generate_reference_logits(args.model_path, args.prompt, args.output_dir)
