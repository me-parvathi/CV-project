"""Benchmarking framework for HOI models."""

import json
import time
import pandas as pd
from pathlib import Path
from typing import Dict, List
from datetime import datetime


class BenchmarkRunner:
    """Runs benchmarks on models with dataset loaders."""
    
    def __init__(self, models: Dict):
        """
        Initialize benchmark runner.
        
        Args:
            models: Dict mapping model names to HOIModel instances.
        """
        self.models = models
        self.results = []
    
    def run_benchmark(self, video_path: str) -> Dict:
        """
        Run all models on a video and collect results.
        
        Args:
            video_path: Path to video file.
            
        Returns:
            Dict with benchmark results for all models.
        """
        from pathlib import Path
        
        # Skip if video file doesn't exist
        if not Path(video_path).exists():
            return None
        
        benchmark_results = {
            "video": video_path,
            "timestamp": datetime.now().isoformat(),
            "model_results": {}
        }
        
        for model_name, model in self.models.items():
            start_time = time.time()
            
            try:
                result = model.process_video(video_path)
                elapsed_time = time.time() - start_time
                
                result["inference_time"] = elapsed_time
                result["success"] = True
                benchmark_results["model_results"][model_name] = result
                
            except Exception as e:
                elapsed_time = time.time() - start_time
                benchmark_results["model_results"][model_name] = {
                    "error": str(e),
                    "inference_time": elapsed_time,
                    "success": False
                }
        
        self.results.append(benchmark_results)
        return benchmark_results
    
    def run_on_dataset(self, videos: List[Dict]) -> None:
        """
        Run benchmark on all videos from dataset.
        
        Args:
            videos: List of video dicts from dataset loader.
        """
        from pathlib import Path
        
        skipped = 0
        for video_info in videos:
            video_path = video_info["video_path"]
            
            # Skip if video file doesn't exist
            if not Path(video_path).exists():
                skipped += 1
                continue
            
            result = self.run_benchmark(video_path)
            if result is None:
                skipped += 1
        
        if skipped > 0:
            print(f"Skipped {skipped} video(s) (file not found or invalid)")
    
    def save_results(self, output_path: str = "results/benchmark_results.json") -> None:
        """
        Save all benchmark results to JSON.
        
        Args:
            output_path: Path to save JSON file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
    
    def generate_report(self) -> pd.DataFrame:
        """
        Generate summary report as DataFrame.
        
        Returns:
            DataFrame with benchmark summary.
        """
        report_data = []
        
        for result in self.results:
            video = result["video"]
            for model_name, model_result in result["model_results"].items():
                row = {
                    "video": video,
                    "model": model_name,
                    "inference_time": model_result.get("inference_time", None),
                    "success": model_result.get("success", False)
                }
                report_data.append(row)
        
        return pd.DataFrame(report_data)
    
    def save_report(self, output_path: str = "results/benchmark_report.csv") -> None:
        """
        Save benchmark report to CSV.
        
        Args:
            output_path: Path to save CSV file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df = self.generate_report()
        df.to_csv(output_path, index=False)

