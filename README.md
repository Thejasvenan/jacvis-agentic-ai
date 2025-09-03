# 🎮 Fine-Tuned TinyLlama RPG Model

## 📌 Overview

This repository hosts a **fine-tuned TinyLlama-1.1B-Chat model** specialized for **RPG (Role-Playing Game) task generation**.
The final model is exported to **GGUF format** for use with **Ollama** and other GGML-based inference engines.

---

## 🚀 Workflow

### **Step 1: Data Collection**

📂 *Goal*: Gather and format RPG-specific training data.

```bash
# collect_mtllm_data.py
# - Extracts RPG examples
# - Formats into task-completion style with system prompts
```

* **Input**: Raw RPG data sources
* **Output**: Training dataset (JSONL / text)

---

### **Step 2: Fine-Tuning with LoRA**

📂 *Goal*: Train TinyLlama efficiently with RPG tasks.

```python
base_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# Apply LoRA adapters and train on RPG dataset
```

* **Method**: [LoRA (Low-Rank Adaptation)](https://arxiv.org/abs/2106.09685)
* **Output**: LoRA adapter weights → `./tinyllama-rpg-finetuned/`

---

### **Step 3: Merge LoRA Weights**

📂 *Goal*: Create a single complete model.

```python
# modelConvert.py
model = AutoModelForCausalLM.from_pretrained(base_model)
tokenizer = AutoTokenizer.from_pretrained(base_model)

# Merge LoRA adapters
model = PeftModel.from_pretrained(model, "./tinyllama-rpg-finetuned")
merged_model = model.merge_and_unload()

merged_model.save_pretrained("./tinyllama-rpg-merged")
tokenizer.save_pretrained("./tinyllama-rpg-merged")
```

* **Input**: Base model + LoRA adapters
* **Output**: Hugging Face model → `./tinyllama-rpg-merged/`

---

### **Step 4: Convert to GGUF**

📂 *Goal*: Make the model compatible with Ollama & llama.cpp.

```bash
pip install sentencepiece
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp

python convert_hf_to_gguf.py \
  --outfile tinyllama-rpg.gguf \
  --outtype f16 \
  ../tinyllama-rpg-merged
```

* **Output**: `tinyllama-rpg.gguf` (\~2.2GB)

---

### **Step 5: Create Ollama Model Definition**

📂 *Goal*: Configure inference parameters.

```dockerfile
# Modelfile
FROM tinyllama-rpg.gguf
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER stop "USER:"
PARAMETER stop "ASSISTANT:"

SYSTEM "Return only the RPG output. No explanations or extra text."
```

---

### **Step 6: Import into Ollama**

📂 *Goal*: Make the model available locally.

```bash
ollama create tinyllama-rpg -f Modelfile
ollama list   # verify installation
```

---

### **Step 7: Run the Model**

📂 *Goal*: Generate RPG responses.

```bash
# CLI
ollama run tinyllama-rpg
```

```python
# Python API
import ollama
response = ollama.chat(
    model="tinyllama-rpg",
    messages=[{"role": "user", "content": "Generate a dungeon quest with 3 levels"}]
)
print(response['message']['content'])
```

---

## 📂 File Structure

```
JAC_Project/
├── collect_mtllm_data.py       # Step 1: Data collection
├── tinyllama-rpg-finetuned/    # Step 2: LoRA adapters
├── modelConvert.py             # Step 3: Merge script
├── tinyllama-rpg-merged/       # Step 3: Merged model
├── llama.cpp/                  # Step 4: Conversion tools
│   └── tinyllama-rpg.gguf      # Step 4: GGUF model
├── Modelfile                   # Step 5: Ollama config
└── README.md                   # Documentation
```

---

## 🔧 Key Dependencies

* **Step 1–3**: `transformers`, `peft`, `torch`, `datasets`
* **Step 4**: `sentencepiece`, `llama.cpp`
* **Step 5–7**: `ollama`

---

## 📊 Model Specifications

* **Base**: `TinyLlama-1.1B-Chat-v1.0`
* **Fine-tuning**: LoRA adapters
* **Parameters**: \~1.1B
* **Format**: GGUF (F16 precision)
* **Context Window**: 2048 tokens
* **Specialization**: RPG quest & level generation

---

## ⚡ Usage Notes

* Outputs **task-focused RPG content** (not general conversation).
* Designed for **low-resource inference** via LoRA + GGUF.
* Compatible with **Ollama**, `llama.cpp`, and GGML runtimes.
