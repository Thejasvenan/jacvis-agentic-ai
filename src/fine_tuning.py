import json
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
from config.settings import BASE_MODEL, TRAINING_SETTINGS, FINE_TUNED_MODEL_PATH

class RPGFineTuner:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        self.model = prepare_model_for_kbit_training(self.model)
        
        self.lora_config = LoraConfig(
            r=TRAINING_SETTINGS["lora_r"],
            lora_alpha=TRAINING_SETTINGS["lora_alpha"],
            lora_dropout=TRAINING_SETTINGS["lora_dropout"],
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        
        self.model = get_peft_model(self.model, self.lora_config)
        self.model.print_trainable_parameters()
    
    def prepare_dataset(self, dataset_path: str):
        with open(dataset_path, 'r') as f:
            data = json.load(f)
        
        formatted_data = []
        for item in data:
            prompt = item["prompt"]
            
            if "response_text" in item:
                response = item["response_text"]
            else:
                response = json.dumps({k: v for k, v in item.items() if k != "prompt"})
            
            text = f"### Instruction:\n{prompt}\n\n### Response:\n{response}</s>"
            formatted_data.append({"text": text})
        
        dataset = Dataset.from_list(formatted_data)
        
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"], 
                truncation=True, 
                padding=False, 
                max_length=1024,
                return_tensors=None
            )
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        return tokenized_dataset
    
    def train(self, dataset_path: str):

        dataset = self.prepare_dataset(dataset_path)
        
        training_args = TrainingArguments(
            output_dir=FINE_TUNED_MODEL_PATH,
            per_device_train_batch_size=TRAINING_SETTINGS["per_device_train_batch_size"],
            gradient_accumulation_steps=TRAINING_SETTINGS["gradient_accumulation_steps"],
            warmup_steps=TRAINING_SETTINGS["warmup_steps"],
            max_steps=TRAINING_SETTINGS["max_steps"],
            learning_rate=TRAINING_SETTINGS["learning_rate"],
            fp16=TRAINING_SETTINGS["fp16"],
            logging_steps=TRAINING_SETTINGS["logging_steps"],
            optim=TRAINING_SETTINGS["optim"],
            save_strategy="steps",
            save_steps=200,
            report_to=None,
            ddp_find_unused_parameters=False
        )
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator
        )
        
        trainer.train()
        
        trainer.save_model()
        self.tokenizer.save_pretrained(FINE_TUNED_MODEL_PATH)
        
        print(f"Training complete. Model saved to {FINE_TUNED_MODEL_PATH}")
        
        return trainer