from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import subprocess

# Merge LoRA weights with base model
base_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
finetuned_path = "./tinyllama-rpg-finetuned"

model = AutoModelForCausalLM.from_pretrained(base_model)
tokenizer = AutoTokenizer.from_pretrained(base_model)

# Load and merge PEFT/LoRA weights
from peft import PeftModel
model = PeftModel.from_pretrained(model, finetuned_path)
model = model.merge_and_unload()

# Save the merged model
output_dir = "./tinyllama-rpg-merged"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
