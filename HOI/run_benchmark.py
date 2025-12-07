"""Main script to run HOI model benchmarks."""

import torch
from pathlib import Path
from data_processing import UCF101Loader, DatasetLoader
from models import YOLOVideoMAEModel, LLaVAModel, Qwen2VLModel
from benchmark import BenchmarkRunner


def main():
    # Detect device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Initialize models (skip unavailable ones)
    print("Loading models...")
    model_candidates = {
        "YOLOv8_VideoMAE": YOLOVideoMAEModel(device=device),
        "LLaVA-1.5": LLaVAModel(device=device),
        "Qwen2-VL": Qwen2VLModel(device=device)
    }
    
    models = {}
    for name, model in model_candidates.items():
        if hasattr(model, 'available') and model.available:
            models[name] = model
            print(f"✓ {name} loaded successfully")
        else:
            error_msg = getattr(model, 'error', 'Unknown error')
            print(f"✗ {name} skipped: {error_msg}")
    
    if len(models) == 0:
        print("No models available. Exiting.")
        return
    
    print(f"\n{len(models)} model(s) ready for benchmarking")
    
    # Load dataset
    dataset_path = "data/videos"  # Update this path to your UCF101 dataset
    loader: DatasetLoader = UCF101Loader(dataset_path)
    videos = loader.load_videos()
    
    if len(videos) == 0:
        print(f"No videos found in {dataset_path}")
        return
    
    print(f"Loaded {len(videos)} videos from dataset")
    
    # Initialize benchmark runner
    benchmark = BenchmarkRunner(models)
    
    # Run benchmark
    print("Starting benchmark...")
    benchmark.run_on_dataset(videos)
    
    # Save results
    benchmark.save_results("results/benchmark_results.json")
    benchmark.save_report("results/benchmark_report.csv")
    
    # Print summary
    report = benchmark.generate_report()
    if len(report) > 0:
        print("\nBenchmark Summary:")
        print(report.groupby('model').agg({
            'inference_time': ['mean', 'std', 'min', 'max'],
            'success': 'sum'
        }).round(2))
    else:
        print("No benchmark results to summarize.")


if __name__ == "__main__":
    main()

