
import os
from pathlib import Path


def setup_environment():
    
    dirs = [
        "mtllm_dataset",
        "mtllm_dataset/raw_data", 
        "mtllm_dataset/training_data",
        "mtllm_dataset/by_function",
        "fine_tuning_dataset"
    ]
    
    for dir_name in dirs:
        Path(dir_name).mkdir(exist_ok=True)
        print(f" Created directory: {dir_name}")
    
    print(" Environment setup complete!")


def check_environment():
    
    required_dirs = [
        "mtllm_dataset",
        "fine_tuning_dataset"
    ]
    
    required_env_vars = [
        "OPENAI_API_KEY",
        "GEMINI_API_KEY"
    ]
    
    print("🔍 Checking environment...")
    
    for dir_name in required_dirs:
        if Path(dir_name).exists():
            print(f" Directory exists: {dir_name}")
        else:
            print(f"Missing directory: {dir_name}")
    
    api_key_found = False
    for env_var in required_env_vars:
        if os.getenv(env_var):
            print(f" Environment variable set: {env_var}")
            api_key_found = True
        else:
            print(f"  Environment variable not set: {env_var}")
    
    if not api_key_found:
        print(" No API key found! Set either OPENAI_API_KEY or GEMINI_API_KEY")
        return False
    
    return True


if __name__ == "__main__":
    setup_environment()
    check_environment()