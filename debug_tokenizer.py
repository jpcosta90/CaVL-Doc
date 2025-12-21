
import sys
import os
sys.path.insert(0, os.path.join(os.getcwd(), "src"))
from cavl_doc.models.backbone_loader import load_model
import torch

print("Loading model to check tokenizer...")
backbone, processor, tokenizer, _, _ = load_model(
    model_name="InternVL3-2B",
    load_in_4bit=False,
    projection_output_dim=1536
)

print(f"Pad Token ID: {tokenizer.pad_token_id}")
print(f"EOS Token ID: {tokenizer.eos_token_id}")
print(f"UNK Token ID: {tokenizer.unk_token_id}")

img_context_token = "<IMG_CONTEXT>"
if img_context_token in tokenizer.get_vocab():
    print(f"{img_context_token} ID: {tokenizer.convert_tokens_to_ids(img_context_token)}")
else:
    print(f"{img_context_token} NOT in vocab")

print(f"Vocab size: {len(tokenizer)}")

# Check if pad_id conflicts with img_context
img_id = tokenizer.convert_tokens_to_ids(img_context_token)
if tokenizer.pad_token_id == img_id:
    print("CRITICAL: Pad Token ID equals IMG_CONTEXT Token ID!")
else:
    print("Pad Token ID is distinct from IMG_CONTEXT Token ID.")
