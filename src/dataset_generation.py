import time
import json
import random
from typing import List, Dict, Any
import google.generativeai as genai
from tqdm import tqdm

from config.settings import API_RATE_LIMITS, RPG_SETTINGS, GEMINI_API_KEY, DATASET_PATH, MAX_DATASET_SIZE
from src.utils import RateLimiter, rate_limiter, estimate_tokens

class RPGDatasetGenerator:
    def __init__(self):
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not found. Please set it in your environment variables.")
        
        genai.configure(api_key=GEMINI_API_KEY)
        
        try:
            models = list(genai.list_models())
            print("Available models:")
            for m in models:
                print(f" - {m.name} | supports: {m.supported_generation_methods}")

            model_name = None
            for m in models:
                if "generateContent" in m.supported_generation_methods:
                    if "gemini" in m.name.lower():
                        model_name = m.name
                        break

            if not model_name:
                raise ValueError("No suitable Gemini model found that supports generateContent")

            print(f"Using model: {model_name}")
            self.gemini_model = genai.GenerativeModel(model_name)
        except Exception as e:
            print(f"Error discovering models: {e}")
            print("Falling back to gemini-1.5-flash (most stable default)")
            self.gemini_model = genai.GenerativeModel("gemini-1.5-flash")

        self.rate_limiter = RateLimiter()

    @rate_limiter
    def generate_with_gemini(self, prompt: str) -> str:
        """Generate RPG level with Gemini API with rate limiting"""
        try:
            response = self.gemini_model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Gemini API error: {e}")
            if "quota" in str(e).lower() or "limit" in str(e).lower():
                print("Quota exceeded. Waiting before retrying...")
                time.sleep(300)  
            return None
    
    def create_rpg_prompt(self, theme: str, difficulty: str, size: str) -> str:
        prompt_template = f"""Design a {difficulty} difficulty, {size} RPG level with a {theme} theme. 
        Create a complete level specification including:
        
        1. Player spawn point with coordinates and characteristics
        2. Enemies with types, positions, behavior patterns, health, and damage
        3. Terrain features with types and positions
        4. Reward locations with types and values
        5. Win condition for the level
        6. Environmental storytelling elements
        
        Format your response as a valid JSON object with this structure:
        {{
            "theme": "{theme}",
            "difficulty": "{difficulty}",
            "size": "{size}",
            "player_spawn": {{"x": 10, "y": 10, "characteristics": ["warrior", "strong"]}},
            "enemies": [
                {{"type": "goblin", "x": 25, "y": 30, "behavior": "patrol", "health": 50, "damage": 10}},
                {{"type": "orc", "x": 40, "y": 45, "behavior": "guard", "health": 100, "damage": 20}}
            ],
            "terrain": [
                {{"type": "mountain", "x": 15, "y": 15, "width": 5, "height": 3}},
                {{"type": "river", "x": 30, "y": 20, "width": 10, "height": 1}}
            ],
            "rewards": [
                {{"type": "health_potion", "x": 20, "y": 25, "value": 25}},
                {{"type": "gold", "x": 35, "y": 40, "value": 100}}
            ],
            "win_condition": "defeat_all_enemies",
            "environment_story": "An ancient battlefield where heroes once fought"
        }}
        
        Please provide only the JSON output without any additional text."""
        
        return prompt_template
    
    def extract_json_from_response(self, response: str) -> Dict[str, Any]:
        if not response:
            return None
            
        try:
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                return json.loads(json_str)
            else:
                return self.parse_text_response(response)
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            return self.parse_text_response(response)
    
    def parse_text_response(self, response: str) -> Dict[str, Any]:
        theme = "unknown"
        difficulty = "medium"
        size = "medium"
        
        lines = response.split('\n')
        for line in lines:
            line_lower = line.lower()
            for theme_option in RPG_SETTINGS["themes"]:
                if theme_option in line_lower:
                    theme = theme_option
                    break
            for diff_option in RPG_SETTINGS["difficulty_levels"]:
                if diff_option in line_lower:
                    difficulty = diff_option
                    break
            for size_option in RPG_SETTINGS["level_sizes"]:
                if size_option in line_lower:
                    size = size_option
                    break
        
        return {
            "theme": theme,
            "difficulty": difficulty,
            "size": size,
            "response_text": response[:500] + "..." if len(response) > 500 else response,
            "parsed_from_text": True
        }
    
    def generate_rpg_level(self) -> Dict[str, Any]:
        theme = random.choice(RPG_SETTINGS["themes"])
        difficulty = random.choice(RPG_SETTINGS["difficulty_levels"])
        size = random.choice(RPG_SETTINGS["level_sizes"])
        
        prompt = self.create_rpg_prompt(theme, difficulty, size)
        print(f"Generating level with theme: {theme}, difficulty: {difficulty}, size: {size}")
        
        response = self.generate_with_gemini(prompt)
        
        if not response:
            print("Failed to get response from Gemini API")
            return None
        
        try:
            level_data = self.extract_json_from_response(response)
            
            if not level_data:
                print("Failed to extract data from response")
                return None
                
            level_data["generation_source"] = "gemini"
            level_data["prompt"] = prompt
            
            return level_data
        except Exception as e:
            print(f"Error processing response: {e}")
            print(f"Response was: {response}")
            return None
    
    def generate_dataset(self, target_size: int = None, output_path: str = None):
        if target_size is None:
            target_size = min(MAX_DATASET_SIZE, API_RATE_LIMITS["gemini"]["daily_limit"] // 2)
        
        if output_path is None:
            output_path = DATASET_PATH
        
        dataset = []
        pbar = tqdm(total=target_size, desc="Generating RPG levels")
        
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        try:
            with open(output_path, 'r') as f:
                existing_data = json.load(f)
                dataset.extend(existing_data)
                pbar.update(len(existing_data))
                print(f"Loaded {len(existing_data)} existing levels")
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        
        failed_attempts = 0
        max_failed_attempts = 10
        
        while len(dataset) < target_size and failed_attempts < max_failed_attempts:
            level = self.generate_rpg_level()
            if level:
                dataset.append(level)
                pbar.update(1)
                failed_attempts = 0  
                
                with open(output_path, 'w') as f:
                    json.dump(dataset, f, indent=2)
            else:
                failed_attempts += 1
                print(f"Failed generation attempt {failed_attempts}/{max_failed_attempts}")
            
            time.sleep(API_RATE_LIMITS["gemini"]["request_delay"] * 1.5)
        
        pbar.close()
        
        if failed_attempts >= max_failed_attempts:
            print(f"Stopped due to {max_failed_attempts} consecutive failures")
        
        print(f"Generated {len(dataset)} RPG levels. Saved to {output_path}")
        return dataset