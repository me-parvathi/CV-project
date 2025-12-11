"""Main script to run episodic memory recall benchmarks on MECCANO dataset."""

import torch
from pathlib import Path
from data_processing import MECCANOLoader, DatasetLoader
from models import MomentRetrievalModel, CLIPModel, FAISSIndex, MMAction2Model, EpisodicMemoryModel
from benchmark import BenchmarkRunner
import time


def main():
    """Main function to run benchmarks."""
    # Detect device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print("=" * 80)
    
    # Initialize models (skip unavailable ones)
    print("Loading models...")
    model_candidates = {
        "Moment-DETR": MomentRetrievalModel(device=device),
        "CLIP": CLIPModel(device=device),
        "FAISS": FAISSIndex(device=device),
        "MMAction2": MMAction2Model(device=device)
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
    print("=" * 80)
    
    # Load dataset
    # Update this path to your MECCANO dataset location
    dataset_path = "/home/pb3071/videos/meccano"  # Default path
    annotation_file = None  # Will auto-detect if None
    
    # Allow override via environment or command line args
    import sys
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
    if len(sys.argv) > 2:
        annotation_file = sys.argv[2]
    
    print(f"\nLoading MECCANO dataset from: {dataset_path}")
    if annotation_file:
        print(f"Using annotation file: {annotation_file}")
    
    loader: DatasetLoader = MECCANOLoader(dataset_path, annotation_file=annotation_file)
    queries = loader.load_queries()
    
    if len(queries) == 0:
        print(f"No queries found in dataset at {dataset_path}")
        print("\nExpected dataset structure:")
        print("  - Video files (.mp4) in dataset directory or subdirectories")
        print("  - Annotation file (JSON/CSV) with video paths, queries, and ground truth")
        print("\nAnnotation file should contain:")
        print("  - video_path or video_id: path to video file")
        print("  - query or text_query: text query string")
        print("  - start_frame/end_frame or start_time/end_time: temporal ground truth")
        print("  - bbox or bounding_box: spatial ground truth (optional)")
        return
    
    print(f"Loaded {len(queries)} queries from dataset")
    
    # For FAISS, build index from all videos if needed
    if "FAISS" in models:
        print("\nBuilding FAISS index...")
        video_paths = list(set([q["video_path"] for q in queries]))
        try:
            models["FAISS"].build_index(video_paths)
            print("✓ FAISS index built successfully")
        except Exception as e:
            print(f"✗ Error building FAISS index: {e}")
            # Remove FAISS from models if index building failed
            if "FAISS" in models:
                del models["FAISS"]
                print("FAISS model removed from benchmark")
    
    # Initialize benchmark runner
    benchmark = BenchmarkRunner(models)
    
    # Run benchmark
    print("\n" + "=" * 80)
    print("Starting benchmark...")
    print("=" * 80)
    
    start_time = time.time()
    benchmark.run_on_dataset(queries)
    total_time = time.time() - start_time
    
    print(f"\nBenchmark completed in {total_time:.2f} seconds")
    
    # Create results directory
    results_dir = Path("Memory/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save results
    results_json_path = results_dir / "benchmark_results.json"
    results_csv_path = results_dir / "benchmark_report.csv"
    
    print(f"\nSaving results...")
    benchmark.save_results(str(results_json_path))
    benchmark.save_report(str(results_csv_path))
    print(f"✓ Results saved to {results_json_path}")
    print(f"✓ Report saved to {results_csv_path}")
    
    # Generate and print summary
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    
    summary = benchmark.compute_summary_statistics()
    
    if len(summary) == 0:
        print("No results to summarize.")
        return
    
    # Print per-model summary
    for model_name, stats in summary.items():
        print(f"\n{model_name}:")
        print(f"  Queries processed: {stats.get('num_queries', 0)}")
        print(f"  Success rate: {stats.get('success_rate', 0):.2%}")
        
        # Latency metrics
        if 'avg_latency' in stats:
            print(f"  Average latency: {stats['avg_latency']:.4f} seconds")
            if 'std_latency' in stats:
                print(f"  Latency std: {stats['std_latency']:.4f} seconds")
            if 'min_latency' in stats:
                print(f"  Min latency: {stats['min_latency']:.4f} seconds")
            if 'max_latency' in stats:
                print(f"  Max latency: {stats['max_latency']:.4f} seconds")
        
        # Retrieval speed
        if 'retrieval_speed_qps' in stats:
            print(f"  Retrieval speed: {stats['retrieval_speed_qps']:.2f} queries/second")
        
        # Accuracy metrics
        print("  Accuracy metrics:")
        for metric in ['recall_at_1', 'recall_at_5', 'temporal_iou', 'spatial_iou']:
            avg_key = f"avg_{metric}"
            std_key = f"std_{metric}"
            if avg_key in stats:
                avg_val = stats[avg_key]
                std_val = stats.get(std_key, 0)
                print(f"    {metric}: {avg_val:.4f} ± {std_val:.4f}")
    
    # Print comparison table
    print("\n" + "=" * 80)
    print("COMPARISON TABLE")
    print("=" * 80)
    
    report = benchmark.generate_report()
    if len(report) > 0:
        # Group by model and compute averages
        comparison_data = []
        for model_name in report['model'].unique():
            model_df = report[report['model'] == model_name]
            row = {'Model': model_name}
            
            # Latency
            if 'latency_seconds' in model_df.columns:
                latencies = model_df['latency_seconds'].dropna()
                if len(latencies) > 0:
                    row['Avg Latency (s)'] = f"{latencies.mean():.4f}"
                    row['Retrieval Speed (qps)'] = f"{len(model_df) / latencies.sum():.2f}" if latencies.sum() > 0 else "N/A"
            
            # Recall metrics
            for metric in ['recall_at_1', 'recall_at_5']:
                if metric in model_df.columns:
                    values = model_df[metric].dropna()
                    if len(values) > 0:
                        row[metric.replace('_', ' ').title()] = f"{values.mean():.4f}"
            
            # Temporal IoU
            if 'temporal_iou' in model_df.columns:
                values = model_df['temporal_iou'].dropna()
                if len(values) > 0:
                    row['Temporal IoU'] = f"{values.mean():.4f}"
            
            comparison_data.append(row)
        
        # Print as table
        if comparison_data:
            import pandas as pd
            comparison_df = pd.DataFrame(comparison_data)
            print("\n" + comparison_df.to_string(index=False))
    
    print("\n" + "=" * 80)
    print("Benchmark complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()

