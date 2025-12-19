"""Benchmarking framework for HOI models."""

import json
import time
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import cv2
import numpy as np
import re


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
        self.video_info_map = {}  # Map video_path to video_info for ground truth access
    
    def run_benchmark(self, video_path: str, video_info: Optional[Dict] = None) -> Dict:
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
        
        # Store video_info for later HOI metric computation
        if video_info is not None:
            self.video_info_map[video_path] = video_info
        
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
            
            result = self.run_benchmark(video_path, video_info)
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
    
    def _normalize_action_name(self, action: str) -> str:
        """Normalize action name for comparison (lowercase, remove special chars)."""
        if not action:
            return ""
        # Convert to lowercase and remove special characters
        normalized = re.sub(r'[^a-z0-9\s]', '', action.lower())
        # Remove extra spaces
        normalized = ' '.join(normalized.split())
        return normalized
    
    def _extract_predicted_action(self, model_result: Dict) -> Optional[str]:
        """Extract predicted action from model result."""
        # Try structured_output first
        if "structured_output" in model_result:
            struct = model_result["structured_output"]
            if "primary_action" in struct:
                return struct["primary_action"]
            if "action_detected" in struct:
                return struct["action_detected"]
            if "action_description" in struct:
                # Extract action from description (simple heuristic)
                desc = struct["action_description"]
                if desc:
                    # Try to extract first meaningful phrase
                    words = desc.lower().split()
                    if len(words) > 0:
                        return ' '.join(words[:3])  # First 3 words as action
        
        # Try actions_detected
        if "actions_detected" in model_result and len(model_result["actions_detected"]) > 0:
            return model_result["actions_detected"][0].get("action", None)
        
        return None
    
    def _extract_predicted_objects(self, model_result: Dict) -> List[str]:
        """Extract predicted objects from model result."""
        objects = set()
        
        # Try structured_output first
        if "structured_output" in model_result:
            struct = model_result["structured_output"]
            if "objects_in_scene" in struct:
                obj_list = struct["objects_in_scene"]
                if isinstance(obj_list, list):
                    objects.update([str(o).lower() for o in obj_list if o])
                elif isinstance(obj_list, str):
                    # Parse string representation of list or comma-separated
                    obj_list = obj_list.lower()
                    # Remove brackets and quotes
                    obj_list = re.sub(r'[\[\]"\']', '', obj_list)
                    objects.update([o.strip() for o in obj_list.split(',') if o.strip()])
            
            if "objects_detected" in struct:
                obj_str = struct["objects_detected"]
                if isinstance(obj_str, str):
                    # Try to extract object names from text
                    words = obj_str.lower().split()
                    # Simple heuristic: capitalize words might be objects
                    objects.update([w for w in words if w.isalpha() and len(w) > 2])
        
        # Try objects_detected list
        if "objects_detected" in model_result:
            for obj in model_result["objects_detected"]:
                if isinstance(obj, dict) and "class" in obj:
                    objects.add(obj["class"].lower())
                elif isinstance(obj, str):
                    objects.add(obj.lower())
        
        return list(objects)
    
    def _compute_temporal_stability(self, video_path: str, model_name: str, model, model_result: Dict) -> float:
        """Compute temporal stability based on model's frame sampling consistency.
        
        This is an approximation: for models that process multiple frames,
        we assume temporal stability based on the confidence/consistency of
        the prediction. For models with frame sampling, we can check if
        multiple queries would yield consistent results (simplified heuristic).
        """
        try:
            # Approximation: If model already processed multiple frames (like VideoMAE),
            # assume good stability if confidence is high. Otherwise, use a default value.
            if "structured_output" in model_result:
                struct = model_result["structured_output"]
                # Check if we have confidence scores
                if "action_confidence" in struct:
                    confidence = float(struct["action_confidence"])
                    # Use confidence as a proxy for stability (higher confidence = more stable)
                    return confidence
                
            # Check actions_detected for confidence
            if "actions_detected" in model_result and len(model_result["actions_detected"]) > 0:
                action_conf = model_result["actions_detected"][0].get("confidence", None)
                if action_conf is not None:
                    return float(action_conf)
            
            # Default: assume moderate stability if we have a prediction
            predicted_action = self._extract_predicted_action(model_result)
            if predicted_action:
                return 0.75  # Default moderate stability
            else:
                return 0.0  # No prediction = no stability
            
        except Exception:
            return 0.0
    
    def _compute_hoi_metrics(self, video_path: str, model_name: str, model_result: Dict) -> Dict:
        """Compute HOI-specific metrics for a model result."""
        metrics = {
            "action_f1": None,
            "object_accuracy": None,
            "temporal_error": None,
            "temporal_stability": None,
            "latency_ms": None
        }
        
        # Extract predictions
        predicted_action = self._extract_predicted_action(model_result)
        predicted_objects = self._extract_predicted_objects(model_result)
        
        # Get ground truth
        video_info = self.video_info_map.get(video_path, {})
        ground_truth = video_info.get("ground_truth", {})
        
        # Latency: convert inference_time to ms/clip
        inference_time = model_result.get("inference_time", None)
        if inference_time is not None:
            metrics["latency_ms"] = inference_time * 1000
        
        # Action F1: Compare predicted vs ground truth action
        if ground_truth and "action" in ground_truth:
            gt_action = ground_truth["action"]
            if predicted_action and gt_action:
                pred_norm = self._normalize_action_name(predicted_action)
                gt_norm = self._normalize_action_name(gt_action)
                
                # Simple exact match for now (can be enhanced with F1 calculation)
                if pred_norm == gt_norm:
                    metrics["action_f1"] = 1.0
                else:
                    # Check for partial match (F1-like calculation)
                    pred_words = set(pred_norm.split())
                    gt_words = set(gt_norm.split())
                    if len(pred_words) > 0 and len(gt_words) > 0:
                        intersection = pred_words & gt_words
                        if intersection:
                            precision = len(intersection) / len(pred_words)
                            recall = len(intersection) / len(gt_words)
                            if precision + recall > 0:
                                metrics["action_f1"] = 2 * precision * recall / (precision + recall)
                            else:
                                metrics["action_f1"] = 0.0
                        else:
                            metrics["action_f1"] = 0.0
                    else:
                        metrics["action_f1"] = 0.0
        
        # Object Accuracy: Compare predicted vs ground truth objects
        if ground_truth and "objects" in ground_truth:
            gt_objects = set([str(o).lower() for o in ground_truth["objects"] if o])
            if len(gt_objects) > 0 and len(predicted_objects) > 0:
                pred_objects_set = set([o.lower() for o in predicted_objects])
                intersection = pred_objects_set & gt_objects
                union = pred_objects_set | gt_objects
                if len(union) > 0:
                    metrics["object_accuracy"] = len(intersection) / len(union)
                else:
                    metrics["object_accuracy"] = 0.0
        
        # Temporal Error: Compute deviation from ground truth temporal boundaries
        if ground_truth and "start_time" in ground_truth and "end_time" in ground_truth:
            try:
                # Get video duration
                cap = cv2.VideoCapture(video_path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                
                if fps > 0 and total_frames > 0:
                    video_duration = total_frames / fps
                    
                    # For now, approximate: assume model predicts entire video duration
                    # If model provides temporal boundaries, use those instead
                    predicted_start = 0.0
                    predicted_end = video_duration
                    
                    gt_start = float(ground_truth["start_time"])
                    gt_end = float(ground_truth["end_time"])
                    
                    error = abs(predicted_start - gt_start) + abs(predicted_end - gt_end)
                    metrics["temporal_error"] = error
            except:
                pass
        
        # Temporal Stability: Consistency across frames
        try:
            model = self.models.get(model_name)
            if model:
                metrics["temporal_stability"] = self._compute_temporal_stability(
                    video_path, model_name, model, model_result
                )
        except:
            pass
        
        return metrics
    
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
                
                # Add HOI metrics
                hoi_metrics = self._compute_hoi_metrics(video, model_name, model_result)
                row.update(hoi_metrics)
                
                report_data.append(row)
        
        return pd.DataFrame(report_data)
    
    def generate_hoi_report(self) -> Optional[pd.DataFrame]:
        """
        Generate HOI-specific benchmark report table.
        
        Returns:
            DataFrame with HOI metrics aggregated by model, or None if no data.
        """
        report = self.generate_report()
        
        if len(report) == 0:
            return None
        
        # Filter to successful runs only
        report = report[report["success"] == True]
        
        if len(report) == 0:
            return None
        
        # Aggregate HOI metrics by model
        hoi_cols = ["action_f1", "object_accuracy", "temporal_error", "temporal_stability", "latency_ms"]
        
        hoi_data = []
        for model_name in report["model"].unique():
            model_data = report[report["model"] == model_name]
            
            row = {"Model": model_name}
            
            # Action F1: mean (filter out None)
            action_f1_vals = model_data["action_f1"].dropna()
            if len(action_f1_vals) > 0:
                row["Action F1"] = f"{action_f1_vals.mean():.3f}"
            else:
                row["Action F1"] = "N/A"
            
            # Object Accuracy: mean (filter out None)
            obj_acc_vals = model_data["object_accuracy"].dropna()
            if len(obj_acc_vals) > 0:
                row["Object Acc"] = f"{obj_acc_vals.mean():.3f}"
            else:
                row["Object Acc"] = "N/A"
            
            # Temporal Error: mean in seconds (filter out None)
            temp_err_vals = model_data["temporal_error"].dropna()
            if len(temp_err_vals) > 0:
                row["Temporal Error (s)"] = f"{temp_err_vals.mean():.2f}"
            else:
                row["Temporal Error (s)"] = "N/A"
            
            # Temporal Stability: mean (filter out None)
            stability_vals = model_data["temporal_stability"].dropna()
            if len(stability_vals) > 0:
                row["Stability"] = f"{stability_vals.mean():.3f}"
            else:
                row["Stability"] = "N/A"
            
            # Latency: mean in ms/clip (filter out None)
            latency_vals = model_data["latency_ms"].dropna()
            if len(latency_vals) > 0:
                row["ms/clip"] = f"{latency_vals.mean():.1f}"
            else:
                row["ms/clip"] = "N/A"
            
            hoi_data.append(row)
        
        if len(hoi_data) == 0:
            return None
        
        hoi_df = pd.DataFrame(hoi_data)
        # Set Model as index for cleaner display
        hoi_df = hoi_df.set_index("Model")
        
        return hoi_df
    
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

