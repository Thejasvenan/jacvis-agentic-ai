# Fine-Tuned TinyLlama RPG Model - README

## Overview
This repository contains a fine-tuned TinyLlama-1.1B-Chat model specialized for RPG (Role-Playing Game) tasks, converted to GGUF format for use with Ollama and other GGML-based inference engines.

## Step-by-Step Process

### Step 1: Data Collection
*Purpose*: Gather RPG-specific training data
python
#### collect_mtllm_data.py
#### Collects and formats RPG task data
#### Creates dataset with system prompt format

- *Input*: Raw RPG data sources
- *Output*: Formatted training dataset
- *Format*: Task-completion structure with system prompts

### Step 2: Fine-Tuning with LoRA
*Purpose*: Train the model on RPG tasks using Parameter Efficient Fine-Tuning
python
## Training script (PEFT/LoRA approach)
base_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
## Apply LoRA adapters to specific layers
## Train on RPG dataset

- *Method*: LoRA (Low-Rank Adaptation)
- *Benefits*: Efficient training, preserves base model knowledge
- *Output*: LoRA adapter weights in ./tinyllama-rpg-finetuned/

### Step 3: Merge LoRA Weights
*Purpose*: Combine LoRA adapters with base model to create a complete model
python
## modelConvert.py
## Load base model
model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

## Load and merge LoRA weights
model = PeftModel.from_pretrained(model, "./tinyllama-rpg-finetuned")
merged_model = model.merge_and_unload()

## Save merged model
merged_model.save_pretrained("./tinyllama-rpg-merged")
tokenizer.save_pretrained("./tinyllama-rpg-merged")

- *Input*: Base model + LoRA adapters
- *Process*: Merge weights into single model
- *Output*: Complete HuggingFace model in ./tinyllama-rpg-merged/

### Step 4: Convert to GGUF Format
*Purpose*: Convert HuggingFace model to GGUF format for Ollama compatibility
bash
## Install dependencies
pip install sentencepiece

## Convert using llama.cpp
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
python convert_hf_to_gguf.py --outfile tinyllama-rpg.gguf --outtype f16 ../tinyllama-rpg-merged

- *Input*: HuggingFace model directory
- *Tool*: llama.cpp conversion script
- *Output*: tinyllama-rpg.gguf file (~2.2GB)

### Step 5: Create Ollama Model Definition
*Purpose*: Configure model parameters for Ollama
dockerfile
## Modelfile
FROM tinyllama-rpg.gguf
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER stop "USER:"
PARAMETER stop "ASSISTANT:"
SYSTEM "This is a task you must complete by returning only the output. Do not include explanations, code, or extra text—only the result."

- *Configuration*: Set inference parameters
- *System Prompt*: Define model behavior
- *Format*: Ollama Modelfile syntax

### Step 6: Import into Ollama
*Purpose*: Make the model available for use
bash
## Create model in Ollama
ollama create tinyllama-rpg -f Modelfile

## Verify model is available
ollama list

- *Import*: Register model with Ollama
- *Naming*: Model becomes available as tinyllama-rpg

### Step 7: Use the Model
*Purpose*: Run inference with the fine-tuned model
bash
# Command line usage
ollama run tinyllama-rpg

## Or programmatically
import ollama
response = ollama.chat(
    model='tinyllama-rpg',
    messages=[{'role': 'user', 'content': 'Your RPG prompt here'}]
)
print(response['message']['content'])

- *Interface*: Command line or API
- *Behavior*: Returns direct, task-focused responses
- *Specialization*: Optimized for RPG tasks

### File Structure After Each Step


JAC_Project/
├── collect_mtllm_data.py           # Step 1: Data collection
├── tinyllama-rpg-finetuned/        # Step 2: LoRA adapters
├── modelConvert.py                 # Step 3: Merge script
├── tinyllama-rpg-merged/           # Step 3: Merged model
├── llama.cpp/                      # Step 4: Conversion tools
│   └── tinyllama-rpg.gguf         # Step 4: GGUF model
├── Modelfile                       # Step 5: Ollama config
└── README.md                       # Documentation


## Key Dependencies by Step

- *Step 1-3*: transformers, peft, torch, datasets
- *Step 4*: sentencepiece, llama.cpp
- *Step 5-7*: ollama

## Model Specifications

- *Base Model*: TinyLlama-1.1B-Chat-v1.0
- *Fine-tuning*: LoRA adapters
- *Size*: ~1.1B parameters
- *Format*: GGUF F16 precision
- *Context*: 2048 tokens
- *Specialization*: RPG task completion

## Usage Notes

- The model outputs direct responses without explanations
- Optimized for task completion rather than conversation
- Maintains efficiency through LoRA fine-tuning approach
- Compatible with any GGML-based inference
