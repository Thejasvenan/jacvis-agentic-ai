import argparse
import os
from src.dataset_generation import RPGDatasetGenerator
from src.finetuning import RPGFineTuner
from config.settings import DATASET_PATH, MAX_DATASET_SIZE, GEMINI_API_KEY

def main():
    parser = argparse.ArgumentParser(description="Auto-Adaptive Fine-tuning for Jac MTLLM using RPG Game Generation")
    parser.add_argument("--generate-data", action="store_true", help="Generate RPG dataset")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--dataset-size", type=int, default=MAX_DATASET_SIZE, help="Size of dataset to generate")
    parser.add_argument("--dataset-path", type=str, default=DATASET_PATH, help="Path to dataset")
    
    args = parser.parse_args()
    
    if not GEMINI_API_KEY:
        print("Error: GEMINI_API_KEY not found. Please set it in your environment variables.")
        return
    
    if args.generate_data:
        print("Generating RPG dataset...")
        generator = RPGDatasetGenerator()
        generator.generate_dataset(target_size=args.dataset_size, output_path=args.dataset_path)
    
    if args.train:
        print("Training model...")
        if not os.path.exists(args.dataset_path):
            print(f"Dataset not found at {args.dataset_path}. Generating first...")
            generator = RPGDatasetGenerator()
            generator.generate_dataset(target_size=args.dataset_size, output_path=args.dataset_path)
        
        finetuner = RPGFineTuner()
        finetuner.train(args.dataset_path)

if __name__ == "__main__":
    main()