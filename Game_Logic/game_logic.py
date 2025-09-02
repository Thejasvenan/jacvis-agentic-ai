import json
import os
import time
import random
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import google.generativeai as genai
from pathlib import Path

@dataclass
class Position:
    x: int
    y: int

@dataclass  
class Wall:
    start_pos: Position
    end_pos: Position

@dataclass
class Level:
    name: str
    difficulty: int
    width: int
    height: int
    num_wall: int
    num_enemies: int
    time_countdown: int
    n_retries_allowed: int

@dataclass
class Map:
    level: Level
    walls: List[Wall]
    small_obstacles: List[Position]
    enemies: List[Position]
    player_pos: Position

class GeminiDatasetGenerator:
    def __init__(self, api_key: str, model_name: str = "gemini-2.0-flash-exp"):
        genai.configure(api_key=api_key)
        self.model_name = model_name
        self.model = genai.GenerativeModel(model_name)
        self.dataset = []
        self.failed_generations = []
        
    def make_gemini_request(self, prompt: str, response_schema: Dict) -> Optional[Dict]:
        """Make API request to Gemini with structured output"""
        try:
            # Configure generation with JSON schema
            generation_config = genai.GenerationConfig(
                response_mime_type="application/json",
                response_schema=response_schema,
                temperature=0.7
            )
            
            print(f"Making Gemini request: {prompt[:80]}...")
            
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            # Parse the JSON response
            result = json.loads(response.text)
            print(f"✓ Generated response with {len(str(result))} characters")
            return result
            
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            print(f"Raw response: {response.text[:200]}...")
            self.failed_generations.append({
                "prompt": prompt,
                "error": f"JSON decode error: {str(e)}",
                "raw_response": response.text if 'response' in locals() else None,
                "timestamp": datetime.now().isoformat()
            })
            return None
            
        except Exception as e:
            print(f"API request failed: {e}")
            self.failed_generations.append({
                "prompt": prompt,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
            return None
    
    def get_level_schema(self) -> Dict:
        """Get the schema for Level generation"""
        return {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "difficulty": {"type": "integer"},
                "width": {"type": "integer"},
                "height": {"type": "integer"},
                "num_wall": {"type": "integer"},
                "num_enemies": {"type": "integer"},
                "time_countdown": {"type": "integer"},
                "n_retries_allowed": {"type": "integer"}
            },
            "required": ["name", "difficulty", "width", "height", "num_wall", "num_enemies", "time_countdown", "n_retries_allowed"]
        }
    
    def get_map_schema(self) -> Dict:
        """Get the schema for Map generation"""
        return {
            "type": "object",
            "properties": {
                "level": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "difficulty": {"type": "integer"},
                        "width": {"type": "integer"},
                        "height": {"type": "integer"},
                        "num_wall": {"type": "integer"},
                        "num_enemies": {"type": "integer"},
                        "time_countdown": {"type": "integer"},
                        "n_retries_allowed": {"type": "integer"}
                    },
                    "required": ["name", "difficulty", "width", "height", "num_wall", "num_enemies", "time_countdown", "n_retries_allowed"]
                },
                "walls": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "start_pos": {
                                "type": "object",
                                "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}},
                                "required": ["x", "y"]
                            },
                            "end_pos": {
                                "type": "object", 
                                "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}},
                                "required": ["x", "y"]
                            }
                        },
                        "required": ["start_pos", "end_pos"]
                    }
                },
                "small_obstacles": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}},
                        "required": ["x", "y"]
                    }
                },
                "enemies": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}},
                        "required": ["x", "y"]
                    }
                },
                "player_pos": {
                    "type": "object",
                    "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}},
                    "required": ["x", "y"]
                }
            },
            "required": ["level", "walls", "small_obstacles", "enemies", "player_pos"]
        }
    
    def create_level_prompt(self, level_manager_state: Dict, last_levels: List[Dict], 
                           difficulty: int, width: int, height: int) -> str:
        """Create prompt for create_next_level function matching MTLLM format"""
        
        # Format previous levels for prompt
        prev_levels_str = []
        for level in last_levels:
            prev_levels_str.append(
                f"Level(name='{level['name']}', difficulty={level['difficulty']}, "
                f"width={level['width']}, height={level['height']}, num_wall={level['num_wall']}, "
                f"num_enemies={level['num_enemies']}, time_countdown={level['time_countdown']}, "
                f"n_retries_allowed={level['n_retries_allowed']})"
            )
        
        prev_levels_formatted = "[" + ", ".join(prev_levels_str) + "]" if prev_levels_str else "[]"
        
        prompt = f"""This is a task you must complete by returning only the output.
Do not include explanations, code, or extra text—only the result.

create_next_level

self = LevelManager(current_level={level_manager_state['current_level']}, current_difficulty={level_manager_state['current_difficulty']}, prev_levels={prev_levels_formatted}, prev_level_maps=[])
last_levels = {prev_levels_formatted}
difficulty = {difficulty}
level_width = {width}
level_height = {height}"""
        
        return prompt
    
    def create_map_prompt(self, level_manager_state: Dict, level: Dict) -> str:
        """Create prompt for create_next_map function matching MTLLM format"""
        
        # Format level for prompt
        level_str = (
            f"Level(name='{level['name']}', difficulty={level['difficulty']}, "
            f"width={level['width']}, height={level['height']}, num_wall={level['num_wall']}, "
            f"num_enemies={level['num_enemies']}, time_countdown={level['time_countdown']}, "
            f"n_retries_allowed={level['n_retries_allowed']})"
        )
        
        # Format previous levels
        prev_levels_str = []
        for prev_level in level_manager_state.get('prev_levels', []):
            prev_levels_str.append(
                f"Level(name='{prev_level['name']}', difficulty={prev_level['difficulty']}, "
                f"width={prev_level['width']}, height={prev_level['height']}, num_wall={prev_level['num_wall']}, "
                f"num_enemies={prev_level['num_enemies']}, time_countdown={prev_level['time_countdown']}, "
                f"n_retries_allowed={prev_level['n_retries_allowed']})"
            )
        
        prev_levels_formatted = "[" + ", ".join(prev_levels_str) + "]" if prev_levels_str else "[]"
        
        prompt = f"""This is a task you must complete by returning only the output.
Do not include explanations, code, or extra text—only the result.

create_next_map

self = LevelManager(current_level={level_manager_state['current_level']}, current_difficulty={level_manager_state['current_difficulty']}, prev_levels={prev_levels_formatted}, prev_level_maps=[])
level = {level_str}"""
        
        return prompt
    
    def validate_level_response(self, response: Dict) -> bool:
        """Validate level generation response"""
        required_fields = ["name", "difficulty", "width", "height", "num_wall", "num_enemies", "time_countdown", "n_retries_allowed"]
        return all(field in response for field in required_fields)
    
    def validate_map_response(self, response: Dict) -> Dict:
        """Validate map generation response and return quality metrics"""
        required_fields = ["level", "walls", "small_obstacles", "enemies", "player_pos"]
        if not all(field in response for field in required_fields):
            return {"valid": False, "score": 0, "issues": ["Missing required fields"]}
        
        issues = []
        score = 100
        
        try:
            # Validate level consistency
            level = response["level"]
            actual_wall_count = len(response["walls"])
            expected_wall_count = level.get("num_wall", 0)
            
            if actual_wall_count != expected_wall_count:
                issues.append(f"Wall count mismatch: expected {expected_wall_count}, got {actual_wall_count}")
                score -= 20
            
            # Validate enemy count
            actual_enemy_count = len(response["enemies"])
            expected_enemy_count = level.get("num_enemies", 0)
            
            if actual_enemy_count != expected_enemy_count:
                issues.append(f"Enemy count mismatch: expected {expected_enemy_count}, got {actual_enemy_count}")
                score -= 20
            
            # Validate position bounds
            width = level.get("width", 20)
            height = level.get("height", 20)
            
            player_pos = response["player_pos"]
            if not (1 <= player_pos["x"] <= width and 1 <= player_pos["y"] <= height):
                issues.append("Player position out of bounds")
                score -= 20
            
            # Check enemy positions
            for i, enemy in enumerate(response["enemies"]):
                if not (1 <= enemy["x"] <= width and 1 <= enemy["y"] <= height):
                    issues.append(f"Enemy {i+1} position out of bounds")
                    score -= 5
            
            # Check obstacle positions
            for i, obs in enumerate(response["small_obstacles"]):
                if not (1 <= obs["x"] <= width and 1 <= obs["y"] <= height):
                    issues.append(f"Obstacle {i+1} position out of bounds")
                    score -= 3
            
        except (KeyError, TypeError) as e:
            issues.append(f"Response structure error: {str(e)}")
            score = 0
        
        return {
            "valid": len(issues) == 0,
            "score": max(score, 0),
            "issues": issues
        }
    
    def generate_training_examples(self, num_examples: int = 50, 
                                 difficulty_range: tuple = (1, 5),
                                 size_range: tuple = (15, 25)) -> None:
        """Generate training examples for both functions"""
        
        print(f"Generating {num_examples} training examples...")
        print(f"Difficulty range: {difficulty_range}")
        print(f"Size range: {size_range}")
        
        successful_generations = 0
        
        for i in range(num_examples):
            print(f"\n--- Generating example {i+1}/{num_examples} ---")
            
            # Simulate different game states
            current_level = (i % 10) + 1
            current_difficulty = min(random.randint(*difficulty_range), max(difficulty_range))
            width = random.randint(*size_range)
            height = random.randint(*size_range)
            
            # Create simulated previous levels (up to 3)
            prev_levels = []
            if current_level > 1:
                num_prev = min(3, current_level - 1)
                for j in range(num_prev):
                    prev_difficulty = max(1, current_difficulty - random.randint(0, 2))
                    prev_level = {
                        "name": f"Level {current_level - j - 1}",
                        "difficulty": prev_difficulty,
                        "width": random.randint(15, 25),
                        "height": random.randint(15, 25),
                        "num_wall": random.randint(5, 20),
                        "num_enemies": random.randint(2, 8),
                        "time_countdown": random.randint(60, 180),
                        "n_retries_allowed": random.randint(2, 5)
                    }
                    prev_levels.append(prev_level)
            
            level_manager_state = {
                "current_level": current_level,
                "current_difficulty": current_difficulty,
                "prev_levels": prev_levels
            }
            
            # Generate Level Example
            level_prompt = self.create_level_prompt(
                level_manager_state, prev_levels, current_difficulty, width, height
            )
            
            level_response = self.make_gemini_request(
                level_prompt, 
                self.get_level_schema()
            )
            
            if level_response and self.validate_level_response(level_response):
                print("✓ Level generation successful")
                
                # Convert to MTLLM format
                level_messages = [
                    {
                        "role": "system",
                        "content": "This is a task you must complete by returning only the output.\nDo not include explanations, code, or extra text—only the result.\n"
                    },
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": level_prompt.split('\n', 2)[2]}]  # Remove system message part
                    }
                ]
                
                self.dataset.append({
                    "messages": level_messages,
                    "assistant": level_response,
                    "function": "create_next_level",
                    "metadata": {
                        "example_id": i,
                        "current_level": current_level,
                        "difficulty": current_difficulty,
                        "timestamp": datetime.now().isoformat()
                    }
                })
                
                # Generate corresponding Map Example
                map_prompt = self.create_map_prompt(level_manager_state, level_response)
                map_response = self.make_gemini_request(
                    map_prompt,
                    self.get_map_schema()
                )
                
                if map_response:
                    validation = self.validate_map_response(map_response)
                    print(f"✓ Map generation successful (score: {validation['score']}/100)")
                    
                    if validation['issues']:
                        print(f"  Issues: {validation['issues']}")
                    
                    # Convert to MTLLM format
                    map_messages = [
                        {
                            "role": "system",
                            "content": "This is a task you must complete by returning only the output.\nDo not include explanations, code, or extra text—only the result.\n"
                        },
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": map_prompt.split('\n', 2)[2]}]  # Remove system message part
                        }
                    ]
                    
                    self.dataset.append({
                        "messages": map_messages,
                        "assistant": map_response,
                        "function": "create_next_map",
                        "metadata": {
                            "example_id": i,
                            "current_level": current_level,
                            "difficulty": current_difficulty,
                            "validation": validation,
                            "timestamp": datetime.now().isoformat()
                        }
                    })
                    
                    successful_generations += 1
                else:
                    print("✗ Map generation failed")
            else:
                print("✗ Level generation failed")
            
            # Add delay to respect API limits
            time.sleep(1)
        
        print(f"\n=== Generation Complete ===")
        print(f"Successful generations: {successful_generations}/{num_examples}")
        print(f"Total dataset entries: {len(self.dataset)}")
        print(f"Failed generations: {len(self.failed_generations)}")
    
    def export_openai_format(self, filename: str = None, quality_threshold: int = 70) -> str:
        """Export dataset in OpenAI fine-tuning format"""
        if filename is None:
            filename = f"rpg_finetuning_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        
        # Filter by quality for map examples
        filtered_dataset = []
        for example in self.dataset:
            if example["function"] == "create_next_level":
                filtered_dataset.append(example)
            elif example["function"] == "create_next_map":
                validation = example["metadata"].get("validation", {"score": 100})
                if validation["score"] >= quality_threshold:
                    filtered_dataset.append(example)
        
        # Convert to OpenAI format
        with open(filename, 'w') as f:
            for example in filtered_dataset:
                openai_example = {
                    "messages": example["messages"] + [
                        {"role": "assistant", "content": json.dumps(example["assistant"])}
                    ]
                }
                f.write(json.dumps(openai_example) + '\n')
        
        print(f"Exported {len(filtered_dataset)} high-quality examples to {filename}")
        return filename
    
    def export_analysis_report(self, filename: str = None) -> str:
        """Export detailed analysis of the generated dataset"""
        if filename is None:
            filename = f"dataset_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Analyze dataset quality
        level_examples = [ex for ex in self.dataset if ex["function"] == "create_next_level"]
        map_examples = [ex for ex in self.dataset if ex["function"] == "create_next_map"]
        
        map_scores = []
        map_issues = []
        for ex in map_examples:
            validation = ex["metadata"].get("validation", {"score": 0, "issues": []})
            map_scores.append(validation["score"])
            map_issues.extend(validation["issues"])
        
        # Count common issues
        issue_counts = {}
        for issue in map_issues:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1
        
        analysis = {
            "generation_timestamp": datetime.now().isoformat(),
            "model_used": self.model_name,
            "total_examples": len(self.dataset),
            "level_examples": len(level_examples),
            "map_examples": len(map_examples),
            "failed_generations": len(self.failed_generations),
            "map_quality_stats": {
                "average_score": sum(map_scores) / len(map_scores) if map_scores else 0,
                "min_score": min(map_scores) if map_scores else 0,
                "max_score": max(map_scores) if map_scores else 0,
                "high_quality_count": sum(1 for score in map_scores if score >= 80),
                "medium_quality_count": sum(1 for score in map_scores if 50 <= score < 80),
                "low_quality_count": sum(1 for score in map_scores if score < 50)
            },
            "common_issues": issue_counts,
            "function_distribution": {
                "create_next_level": len(level_examples),
                "create_next_map": len(map_examples)
            },
            "failed_generation_reasons": [fg["error"] for fg in self.failed_generations[:10]]  # First 10 errors
        }
        
        with open(filename, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        print(f"Analysis report saved to {filename}")
        return filename
    
    def save_raw_dataset(self, filename: str = None) -> str:
        """Save the complete dataset with metadata"""
        if filename is None:
            filename = f"raw_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        dataset_export = {
            "metadata": {
                "generation_timestamp": datetime.now().isoformat(),
                "model_used": self.model_name,
                "total_examples": len(self.dataset),
                "failed_generations": len(self.failed_generations)
            },
            "dataset": self.dataset,
            "failed_generations": self.failed_generations
        }
        
        with open(filename, 'w') as f:
            json.dump(dataset_export, f, indent=2)
        
        print(f"Raw dataset saved to {filename}")
        return filename

def main():
    """Main function to generate training data"""
    print("RPG Level Generation Fine-tuning Dataset Generator")
    print("=" * 60)
    
    # Get API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: Please set your GEMINI_API_KEY environment variable")
        print("Get your key from: https://aistudio.google.com/app/apikey")
        return
    
    # Initialize generator
    print("Initializing Gemini API...")
    try:
        generator = GeminiDatasetGenerator(api_key)
        print(f"✓ Using model: {generator.model_name}")
    except Exception as e:
        print(f"Failed to initialize Gemini API: {e}")
        return
    
    # Generate training examples
    num_examples = 20  # Start small for testing
    print(f"\nGenerating {num_examples} training examples...")
    
    try:
        generator.generate_training_examples(
            num_examples=num_examples,
            difficulty_range=(1, 5),
            size_range=(15, 25)
        )
        
        # Export results
        print("\n" + "="*60)
        print("EXPORTING RESULTS")
        print("="*60)
        
        training_file = generator.export_openai_format(quality_threshold=60)
        analysis_file = generator.export_analysis_report()
        raw_file = generator.save_raw_dataset()
        
        print(f"\n📁 Files generated:")
        print(f"  🎯 Training data: {training_file}")
        print(f"  📊 Analysis report: {analysis_file}")
        print(f"  💾 Raw dataset: {raw_file}")
        
        # Quick stats
        level_count = sum(1 for ex in generator.dataset if ex["function"] == "create_next_level")
        map_count = sum(1 for ex in generator.dataset if ex["function"] == "create_next_map")
        
        print(f"\n📈 Quick Stats:")
        print(f"  Level examples: {level_count}")
        print(f"  Map examples: {map_count}")
        print(f"  Success rate: {len(generator.dataset)/(num_examples*2)*100:.1f}%")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Generation interrupted by user")
        if generator.dataset:
            print("Saving partial results...")
            generator.export_openai_format(filename="partial_dataset.jsonl")
            generator.save_raw_dataset(filename="partial_raw_dataset.json")
    except Exception as e:
        print(f"\n❌ Error during generation: {e}")
        if generator.dataset:
            print("Saving partial results...")
            generator.export_openai_format(filename="error_recovery_dataset.jsonl")

if __name__ == "__main__":
    main()