"""Main script to run routing model benchmarks."""

import sys
from pathlib import Path
from data_processing import load_snips_dataset, preprocess_data, create_splits
from models import (
    KeywordBaseline,
    TFIDFLogisticRegression,
    DistilBERTModel,
    BERTMiniModel
)
from benchmarking import BenchmarkRunner


def main():
    """Main execution function."""
    # Load dataset from HuggingFace (default) or local files
    use_huggingface = True
    
    # Check if local dataset path is provided as command line argument
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
        use_huggingface = False
        print(f"Loading SNIPS dataset from local path: {dataset_path}")
    else:
        print("Loading SNIPS dataset from HuggingFace...")
    
    try:
        if use_huggingface:
            raw_data = load_snips_dataset(use_huggingface=True)
        else:
            raw_data = load_snips_dataset(dataset_path=dataset_path, use_huggingface=False)
        print(f"Loaded {len(raw_data)} examples")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("\nTrying fallback to local dataset...")
        # Fallback to local dataset
        project_root = Path(__file__).parent.parent
        dataset_path = project_root / "data" / "snips" / "2017-06-custom-intent-engines"
        if dataset_path.exists():
            raw_data = load_snips_dataset(dataset_path=str(dataset_path), use_huggingface=False)
            print(f"Loaded {len(raw_data)} examples from local files")
        else:
            print(f"Error: Could not load dataset from HuggingFace or local path: {dataset_path}")
            sys.exit(1)
    
    # Preprocess data
    print("Preprocessing data...")
    processed_data = preprocess_data(raw_data)
    
    # Create train/test splits
    print("Creating train/test splits...")
    train_data, test_data = create_splits(processed_data, test_size=0.2, random_state=42)
    print(f"Train: {len(train_data)} examples, Test: {len(test_data)} examples")
    
    # Initialize models
    print("\nInitializing models...")
    models = {
        "KeywordBaseline": KeywordBaseline(),
        "TFIDFLogisticRegression": TFIDFLogisticRegression(),
        "DistilBERTModel": DistilBERTModel(),
        "BERTMiniModel": BERTMiniModel()
    }
    
    # Train models
    print("\nTraining models...")
    for model_name, model in models.items():
        print(f"  Training {model_name}...")
        try:
            model.train(train_data)
            print(f"    ✓ {model_name} trained successfully")
        except Exception as e:
            print(f"    ✗ {model_name} training failed: {e}")
            # Remove failed models
            models.pop(model_name, None)
    
    if not models:
        print("No models available for benchmarking. Exiting.")
        return
    
    # Run benchmarks
    print(f"\nRunning benchmarks on {len(test_data)} test examples...")
    benchmark = BenchmarkRunner()
    results = benchmark.run_benchmark(models, test_data)
    
    # Print summary
    benchmark.print_summary()
    
    # Save results (optional)
    try:
        output_dir = Path(__file__).parent / "results"
        output_dir.mkdir(exist_ok=True)
        
        summary_df = benchmark.generate_summary_table()
        summary_df.to_csv(output_dir / "benchmark_results.csv", index=False)
        print(f"\nResults saved to {output_dir / 'benchmark_results.csv'}")
    except Exception as e:
        print(f"Warning: Could not save results: {e}")


if __name__ == "__main__":
    main()

