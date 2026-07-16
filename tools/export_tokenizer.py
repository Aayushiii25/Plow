"""
Export an HF tokenizer to InferSpore's binary .flm format.

Usage:
    python tools/export_tokenizer.py \
        --model_path /path/to/hf/model_or_tokenizer \
        --output tokenizer.flm

The .flm format:
    [4 bytes] magic: "ISPT" (InferSpore Tokenizer)
    [4 bytes] vocab_size (int32)
    [4 bytes] bos_token_id (int32)
    [4 bytes] eos_token_id (int32)
    For each token (0..vocab_size-1):
        [4 bytes] token_string_length (int32)
        [N bytes] token_string (UTF-8)
        [4 bytes] token_score (float32, used for merge priority)
"""

import argparse
import struct
import json
import os
from pathlib import Path


def export_from_sentencepiece(sp_model_path, output_path, bos_id=1, eos_id=2):
    """Export from a SentencePiece .model file."""
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=str(sp_model_path))
    
    vocab_size = sp.GetPieceSize()
    print(f"Vocab size: {vocab_size}")
    print(f"BOS id: {bos_id}, EOS id: {eos_id}")
    
    with open(output_path, 'wb') as f:
        # Magic
        f.write(b'ISPT')
        # Header
        f.write(struct.pack('<i', vocab_size))
        f.write(struct.pack('<i', bos_id))
        f.write(struct.pack('<i', eos_id))
        
        for i in range(vocab_size):
            piece = sp.IdToPiece(i)
            score = sp.GetScore(i)
            piece_bytes = piece.encode('utf-8')
            f.write(struct.pack('<i', len(piece_bytes)))
            f.write(piece_bytes)
            f.write(struct.pack('<f', score))
    
    print(f"Exported {vocab_size} tokens to {output_path}")


def export_from_hf_tokenizer(model_path, output_path):
    """Export from an HF tokenizer (auto-detects format)."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else 1
    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 2
    vocab_size = tokenizer.vocab_size
    
    print(f"Vocab size: {vocab_size}")
    print(f"BOS id: {bos_id}, EOS id: {eos_id}")
    
    # Try to get SentencePiece model for scores
    sp = None
    sp_model_path = os.path.join(model_path, "tokenizer.model")
    if os.path.exists(sp_model_path):
        try:
            import sentencepiece as spm
            sp = spm.SentencePieceProcessor(model_file=sp_model_path)
        except ImportError:
            pass
    
    with open(output_path, 'wb') as f:
        # Magic
        f.write(b'ISPT')
        # Header
        f.write(struct.pack('<i', vocab_size))
        f.write(struct.pack('<i', bos_id))
        f.write(struct.pack('<i', eos_id))
        
        for i in range(vocab_size):
            if sp is not None:
                piece = sp.IdToPiece(i)
                score = sp.GetScore(i)
            else:
                piece = tokenizer.convert_ids_to_tokens(i)
                if piece is None:
                    piece = f"<unk_{i}>"
                score = -float(i)  # decreasing score as fallback
            
            piece_bytes = piece.encode('utf-8')
            f.write(struct.pack('<i', len(piece_bytes)))
            f.write(piece_bytes)
            f.write(struct.pack('<f', score))
    
    print(f"Exported {vocab_size} tokens to {output_path}")
    
    # Verify round-trip
    test_text = "Hello, world!"
    hf_tokens = tokenizer.encode(test_text, add_special_tokens=False)
    print(f"Verification - '{test_text}' -> {hf_tokens}")
    hf_decoded = tokenizer.decode(hf_tokens)
    print(f"Decoded back: '{hf_decoded}'")


def main():
    parser = argparse.ArgumentParser(description="Export HF tokenizer to InferSpore .flm format")
    parser.add_argument('--model_path', '-i', type=str, required=True,
                        help="Path to HF model directory or tokenizer.model file")
    parser.add_argument('--output', '-o', type=str, default="tokenizer.flm",
                        help="Output path for .flm file")
    args = parser.parse_args()
    
    model_path = args.model_path
    
    # Auto-detect format
    if model_path.endswith('.model'):
        export_from_sentencepiece(model_path, args.output)
    else:
        export_from_hf_tokenizer(model_path, args.output)


if __name__ == "__main__":
    main()
