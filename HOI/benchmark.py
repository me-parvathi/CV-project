"""Benchmarking framework for HOI models."""

import json
import time
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


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
    
    def _get_memory_usage(self) -> Dict:
        """Get current memory usage."""
        memory_info = {}
        
        # CPU memory
        if PSUTIL_AVAILABLE:
            process = psutil.Process()
            memory_info["cpu_memory_mb"] = process.memory_info().rss / 1024 / 1024
        
        # GPU memory
        if TORCH_AVAILABLE and torch.cuda.is_available():
            memory_info["gpu_memory_mb"] = torch.cuda.memory_allocated() / 1024 / 1024
            memory_info["gpu_memory_reserved_mb"] = torch.cuda.memory_reserved() / 1024 / 1024
        
        return memory_info
    
    def _extract_avg_confidence(self, result: Dict) -> Optional[float]:
        """
        Extract average confidence from model results.
        
        Note: Text-based models (LLaVA, Qwen2-VL) don't provide numeric confidence
        scores, so this will return None for those models. This is expected behavior.
        """
        confidences = []
        
        # Extract object confidences
        if "objects_detected" in result:
            for obj in result["objects_detected"]:
                if "confidence" in obj:
                    confidences.append(obj["confidence"])
        
        # Extract action confidences
        if "actions_detected" in result:
            for action in result["actions_detected"]:
                if "confidence" in action:
                    confidences.append(action["confidence"])
        
        # Extract from structured_output
        if "structured_output" in result:
            so = result["structured_output"]
            if "action_confidence" in so:
                confidences.append(so["action_confidence"])
        
        return sum(confidences) / len(confidences) if confidences else None
    
    def _compare_with_ground_truth(self, result: Dict, ground_truth: Optional[Dict]) -> Optional[Dict]:
        """Compare model results with ground truth if available."""
        if not ground_truth:
            return None
        
        metrics = {}
        
        # Action accuracy (if ground truth has action)
        if "action" in ground_truth and "structured_output" in result:
            so = result["structured_output"]
            gt_action = ground_truth["action"].lower()
            
            # Try different field names for action prediction
            predicted_action_text = ""
            if "primary_action" in so:
                predicted_action_text = str(so["primary_action"]).lower()
            elif "action_description" in so:
                predicted_action_text = str(so["action_description"]).lower()
            elif "action_detected" in so:
                predicted_action_text = str(so["action_detected"]).lower()
            
            # For text-based models, do fuzzy matching (check if GT action appears in description)
            if predicted_action_text:
                # Exact match
                exact_match = predicted_action_text == gt_action
                # Fuzzy match: check if ground truth action name appears in the description
                fuzzy_match = gt_action in predicted_action_text or predicted_action_text in gt_action
                metrics["action_correct"] = exact_match or fuzzy_match
        
        # Object detection accuracy (if ground truth has objects)
        if "objects" in ground_truth:
            gt_objects = {obj.lower() for obj in ground_truth["objects"]} if isinstance(ground_truth["objects"], list) else set()
            
            if gt_objects:
                predicted_objects = set()
                
                # For structured detection (YOLO-style)
                if "objects_detected" in result and isinstance(result["objects_detected"], list):
                    predicted_objects = {obj.get("class", "").lower() for obj in result["objects_detected"] if isinstance(obj, dict)}
                
                # For text-based descriptions (LLaVA, Qwen2-VL)
                elif "structured_output" in result:
                    so = result["structured_output"]
                    objects_text = ""
                    if "objects_mentioned" in so:
                        objects_text = str(so["objects_mentioned"]).lower()
                    elif "objects_detected" in so:
                        objects_text = str(so["objects_detected"]).lower()
                    
                    # Extract object names from text by checking if GT objects are mentioned
                    # This is a simple heuristic - could be improved with NLP
                    if objects_text:
                        for gt_obj in gt_objects:
                            if gt_obj in objects_text or any(word in objects_text for word in gt_obj.split()):
                                predicted_objects.add(gt_obj)
                
                if predicted_objects or gt_objects:
                    intersection = predicted_objects & gt_objects
                    metrics["object_precision"] = len(intersection) / len(predicted_objects) if predicted_objects else 0.0
                    metrics["object_recall"] = len(intersection) / len(gt_objects) if gt_objects else 0.0
                    metrics["object_f1"] = 2 * metrics["object_precision"] * metrics["object_recall"] / (metrics["object_precision"] + metrics["object_recall"]) if (metrics["object_precision"] + metrics["object_recall"]) > 0 else 0.0
        
        return metrics if metrics else None
    
    def run_benchmark(self, video_path: str, ground_truth: Optional[Dict] = None) -> Dict:
        """
        Run all models on a video and collect results.
        
        Args:
            video_path: Path to video file.
            ground_truth: Optional ground truth dict with 'action' and/or 'objects' keys.
            
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
            memory_before = self._get_memory_usage()
            
            try:
                result = model.process_video(video_path)
                elapsed_time = time.time() - start_time
                memory_after = self._get_memory_usage()
                
                # Extract average confidence
                avg_confidence = self._extract_avg_confidence(result)
                
                # Compare with ground truth if available
                accuracy_metrics = self._compare_with_ground_truth(result, ground_truth)
                
                result["inference_time"] = elapsed_time
                result["success"] = True
                if avg_confidence is not None:
                    result["avg_confidence"] = avg_confidence
                if memory_after:
                    result["memory_usage"] = memory_after
                    if memory_before and "gpu_memory_mb" in memory_after:
                        result["memory_usage"]["gpu_memory_delta_mb"] = (
                            memory_after["gpu_memory_mb"] - memory_before.get("gpu_memory_mb", 0)
                        )
                if accuracy_metrics:
                    result["accuracy_metrics"] = accuracy_metrics
                
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
            videos: List of video dicts from dataset loader. May include 'ground_truth' key.
        """
        from pathlib import Path
        
        skipped = 0
        for video_info in videos:
            video_path = video_info["video_path"]
            
            # Skip if video file doesn't exist
            if not Path(video_path).exists():
                skipped += 1
                continue
            
            # Extract ground truth if available
            ground_truth = video_info.get("ground_truth")
            
            result = self.run_benchmark(video_path, ground_truth=ground_truth)
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
                    "success": model_result.get("success", False),
                    "avg_confidence": model_result.get("avg_confidence", None)
                }
                
                # Add memory metrics
                if "memory_usage" in model_result:
                    mem = model_result["memory_usage"]
                    row["cpu_memory_mb"] = mem.get("cpu_memory_mb", None)
                    row["gpu_memory_mb"] = mem.get("gpu_memory_mb", None)
                    row["gpu_memory_delta_mb"] = mem.get("gpu_memory_delta_mb", None)
                
                # Add accuracy metrics if available
                if "accuracy_metrics" in model_result:
                    acc = model_result["accuracy_metrics"]
                    row["action_correct"] = acc.get("action_correct", None)
                    row["object_precision"] = acc.get("object_precision", None)
                    row["object_recall"] = acc.get("object_recall", None)
                    row["object_f1"] = acc.get("object_f1", None)
                
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

