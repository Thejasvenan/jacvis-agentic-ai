
import json
import random
import asyncio
import os
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass
import time


@dataclass
class PromptTemplate:
    function_name: str
    base_prompt: str
    parameter_variations: List[Dict[str, Any]]
    response_schema: Dict[str, Any]


class DatasetGenerator:
    
    def __init__(self, 
                 collected_data_dir: str = "mtllm_dataset",
                 output_dir: str = "fine_tuning_dataset",
                 big_model_api: str = "openai",  
                 big_model_name: str = "gpt-4"):
        
        self.collected_data_dir = Path(collected_data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.big_model_api = big_model_api
        self.big_model_name = big_model_name
        
        # Initialize LLM client based on API choice
        if big_model_api == "openai":
            import openai
            self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        elif big_model_api == "gemini":
            import google.generativeai as genai
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            self.client = genai.GenerativeModel(big_model_name)
        else:
            raise ValueError("big_model_api must be 'openai' or 'gemini'")
    
    def extract_prompt_templates(self) -> List[PromptTemplate]:
        templates = {}
        
        by_function_dir = self.collected_data_dir / "by_function"
        
        if not by_function_dir.exists():
            print(f"❌ No function data found in {by_function_dir}")
            return []
        
        for function_dir in by_function_dir.iterdir():
            if function_dir.is_dir():
                function_name = function_dir.name
                examples = []
                
                for example_file in function_dir.glob("*.json"):
                    try:
                        with open(example_file) as f:
                            data = json.load(f)
                            examples.append(data)
                    except Exception as e:
                        print(f"❌ Error reading {example_file}: {e}")
                        continue
                
                if examples:
                    base_example = examples[0]
                    base_prompt = base_example["messages"][1]["content"]  
                    
                    parameter_variations = self._extract_parameter_variations(examples)
                    
                    response_schema = base_example.get("metadata", {}).get("schema", {})
                    
                    templates[function_name] = PromptTemplate(
                        function_name=function_name,
                        base_prompt=base_prompt,
                        parameter_variations=parameter_variations,
                        response_schema=response_schema 
                    )
        
        return list(templates.values())
    
    def _extract_parameter_variations(self, examples: List[Dict]) -> List[Dict[str, Any]]:
        variations = []
        
        for example in examples:
            user_prompt = example["messages"][1]["content"]
            params = self._parse_parameters_from_prompt(user_prompt)
            variations.append(params)
        
        return variations
    
    def _parse_parameters_from_prompt(self, prompt: str) -> Dict[str, Any]:
        """Parse parameters from the prompt text."""
        params = {}
        lines = prompt.split('\n')
        
        for line in lines:
            if '=' in line and not line.startswith('self ='):
                try:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    try:
                        if value.startswith('[') and value.endswith(']'):
                            params[key] = "list"
                        elif value.isdigit():
                            params[key] = int(value)
                        elif value.replace('.', '').isdigit():
                            params[key] = float(value)
                        else:
                            params[key] = value
                    except:
                        params[key] = value
                        
                except ValueError:
                    continue
        
        return params
    
    async def generate_synthetic_data(self, 
                                     templates: List[PromptTemplate], 
                                     samples_per_function: int = 50) -> List[Dict]:
        """Generate synthetic training data using big models."""
        
        all_training_data = []
        
        for template in templates:
            print(f" Generating {samples_per_function} samples for {template.function_name}")
            
            try:
                function_data = await self._generate_function_samples(
                    template, samples_per_function
                )
                all_training_data.extend(function_data)
                
                function_output_file = self.output_dir / f"{template.function_name}_dataset.jsonl"
                self._save_jsonl(function_data, function_output_file)
                
            except Exception as e:
                print(f" Error generating samples for {template.function_name}: {e}")
                continue
        
        if all_training_data:
            combined_output_file = self.output_dir / "combined_dataset.jsonl"
            self._save_jsonl(all_training_data, combined_output_file)
        
        return all_training_data
    
    async def _generate_function_samples(self, 
                                        template: PromptTemplate, 
                                        num_samples: int) -> List[Dict]:
        
        samples = []
        
        for i in range(num_samples):
            try:
                sample = await self._generate_single_sample(template, i)
                if sample:
                    samples.append(sample)
                
                await asyncio.sleep(0.5)
                
            except Exception as e:
                print(f" Error generating sample {i} for {template.function_name}: {e}")
                continue
        
        return samples
    
    async def _generate_single_sample(self, 
                                     template: PromptTemplate, 
                                     sample_id: int) -> Dict:
        """Generate a single training sample."""
        
        try:
            # Create varied prompt
            varied_prompt = self._create_varied_prompt(template, sample_id)
            
            # Get response from big model
            response = await self._call_big_model(varied_prompt)
            
            if not response:
                return None
            
            # Format as training example
            training_example = {
                "messages": [
                    {
                        "role": "system",
                        "content": "This is a task you must complete by returning only the output.\nDo not include explanations, code, or extra text—only the result."
                    },
                    {
                        "role": "user",
                        "content": varied_prompt
                    },
                    {
                        "role": "assistant",
                        "content": response
                    }
                ],
                "metadata": {
                    "function_name": template.function_name,
                    "sample_id": sample_id,
                    "generated_at": time.time()
                }
            }
            
            return training_example
            
        except Exception as e:
            print(f"❌ Error generating sample {sample_id} for {template.function_name}: {e}")
            return None
    
    def _create_varied_prompt(self, template: PromptTemplate, sample_id: int) -> str:
        """Create a varied version of the base prompt."""
        
        base_lines = template.base_prompt.split('\n')
        function_name = base_lines[0] if base_lines else template.function_name
        
        # Generate varied parameters
        if template.parameter_variations:
            base_params = random.choice(template.parameter_variations)
            varied_params = self._vary_parameters(base_params, template.function_name, sample_id)
        else:
            varied_params = self._generate_default_parameters(template.function_name, sample_id)
        
        # Reconstruct prompt
        prompt_lines = [function_name, ""]
        
        # Add self parameter if it exists in original
        for line in base_lines[1:]:
            if line.startswith('self ='):
                prompt_lines.append(self._create_varied_self_param(line, sample_id))
                break
        
        # Add varied parameters
        for key, value in varied_params.items():
            prompt_lines.append(f"{key} = {value}")
        
        return '\n'.join(prompt_lines)
    
    def _vary_parameters(self, base_params: Dict, function_name: str, sample_id: int) -> Dict:
        """Create parameter variations."""
        
        varied = base_params.copy()
        
        if function_name == "create_next_level":
            if "difficulty" in varied:
                varied["difficulty"] = random.randint(1, 10)
            if "level_width" in varied:
                varied["level_width"] = random.choice([15, 20, 25, 30])
            if "level_height" in varied:
                varied["level_height"] = random.choice([15, 20, 25, 30])
                
        elif function_name == "create_next_map":
            pass  # Map generation depends on level parameter
        
        return varied
    
    def _generate_default_parameters(self, function_name: str, sample_id: int) -> Dict:
        """Generate default parameters for functions."""
        
        if function_name == "create_next_level":
            return {
                "last_levels": "[]",
                "difficulty": random.randint(1, 5),
                "level_width": random.choice([15, 20, 25]),
                "level_height": random.choice([15, 20, 25])
            }
        elif function_name == "create_next_map":
            sample_level = {
                "name": f"Level {random.randint(1, 20)}",
                "difficulty": random.randint(1, 5),
                "width": random.choice([15, 20, 25]),
                "height": random.choice([15, 20, 25]),
                "num_wall": random.randint(20, 80),
                "num_enemies": random.randint(3, 10),
                "time_countdown": random.randint(60, 300),
                "n_retries_allowed": random.randint(1, 5)
            }
            return {"level": str(sample_level)}
        
        return {}
    
    def _create_varied_self_param(self, original_self_line: str, sample_id: int) -> str:
        """Create varied self parameter with accumulated data."""
        
        current_level = (sample_id % 10) + 1
        current_difficulty = (sample_id % 5) + 1
        
        prev_levels = []
        for i in range(min(3, current_level)):
            level = {
                "name": f"Level {current_level - i}",
                "difficulty": max(1, current_difficulty - i),
                "width": random.choice([15, 20, 25]),
                "height": random.choice([15, 20, 25]),
                "num_wall": random.randint(20, 60),
                "num_enemies": random.randint(2, 8),
                "time_countdown": random.randint(60, 200),
                "n_retries_allowed": random.randint(1, 4)
            }
            prev_levels.append(level)
        
        self_param = (
            f"self = LevelManager("
            f"current_level={current_level}, "
            f"current_difficulty={current_difficulty}, "
            f"prev_levels={prev_levels}, "
            f"prev_level_maps=[])"
        )
        
        return self_param
    
    async def _call_big_model(self, prompt: str) -> str:
        """Call big model to generate response."""
        
        try:
            if self.big_model_api == "openai":
                response = await self.client.chat.completions.create(
                    model=self.big_model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": "This is a task you must complete by returning only the output.\nDo not include explanations, code, or extra text—only the result."
                        },
                        {
                            "role": "user", 
                            "content": prompt
                        }
                    ],
                    temperature=0.7,
                    max_tokens=1000
                )
                return response.choices[0].message.content
                
            elif self.big_model_api == "gemini":
                messages = f"""System: This is a task you must complete by returning only the output.
Do not include explanations, code, or extra text—only the result.

User: {prompt}"""
                
                response = await asyncio.to_thread(
                    self.client.generate_content, 
                    messages,
                    generation_config={"temperature": 0.7, "max_output_tokens": 1000}
                )
                return response.text
            
        except Exception as e:
            print(f" Error calling big model: {e}")
            return None
    
    def _save_jsonl(self, data: List[Dict], filename: Path):
        """Save data in JSONL format for training."""
        with open(filename, 'w') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
        
        print(f" Saved {len(data)} samples to {filename}")