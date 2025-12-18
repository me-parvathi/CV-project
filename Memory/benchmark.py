"""Benchmarking framework for episodic memory recall evaluation."""

import json
import time
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime


def recall_at_k(y_true: List[int], y_pred: List[int], k: int) -> float:
    """
    Compute Recall@K metric.
    
    Args:
        y_true: List of ground truth indices (correct moment indices).
        y_pred: List of predicted indices (ranked by model).
        k: Number of top predictions to consider.
        
    Returns:
        Recall@K score (0.0 to 1.0).
    """
    if len(y_true) == 0:
        return 0.0
    
    y_pred_at_k = y_pred[:k] if len(y_pred) > k else y_pred
    true_set = set(y_true)
    pred_set = set(y_pred_at_k)
    
    intersection = len(true_set & pred_set)
    recall = intersection / len(true_set) if len(true_set) > 0 else 0.0
    
    return recall


def temporal_iou(pred_intervals: List[Tuple[float, float]], 
                 true_intervals: List[Tuple[float, float]]) -> float:
    """
    Compute Temporal Intersection over Union (IoU).
    
    Args:
        pred_intervals: List of (start_time, end_time) tuples for predictions.
        true_intervals: List of (start_time, end_time) tuples for ground truth.
        
    Returns:
        Average Temporal IoU score (0.0 to 1.0).
    """
    if len(pred_intervals) == 0 or len(true_intervals) == 0:
        return 0.0
    
    ious = []
    
    # For each prediction, find best matching ground truth
    for pred_start, pred_end in pred_intervals:
        best_iou = 0.0
        for true_start, true_end in true_intervals:
            # Calculate intersection
            intersection_start = max(pred_start, true_start)
            intersection_end = min(pred_end, true_end)
            intersection = max(0.0, intersection_end - intersection_start)
            
            # Calculate union
            union_start = min(pred_start, true_start)
            union_end = max(pred_end, true_end)
            union = union_end - union_start
            
            # Calculate IoU
            iou = intersection / union if union > 0 else 0.0
            best_iou = max(best_iou, iou)
        
        ious.append(best_iou)
    
    return np.mean(ious) if len(ious) > 0 else 0.0


def spatial_localization_metrics(pred_boxes: List[List[float]], 
                                 true_boxes: List[List[float]],
                                 box_format: str = 'xyxy') -> Dict[str, float]:
    """
    Compute Spatial Localization Metrics (IoU for bounding boxes).
    
    Args:
        pred_boxes: List of predicted bounding boxes.
        true_boxes: List of ground truth bounding boxes.
        box_format: Format of boxes - 'xyxy' (x1, y1, x2, y2) or 'xywh' (x, y, w, h).
        
    Returns:
        Dict with 'spatial_iou' and optionally other metrics.
    """
    if len(pred_boxes) == 0 or len(true_boxes) == 0:
        return {"spatial_iou": 0.0}
    
    ious = []
    
    # Convert boxes to xyxy format if needed
    def to_xyxy(box, fmt):
        if fmt == 'xyxy':
            return box
        elif fmt == 'xywh':
            x, y, w, h = box
            return [x, y, x + w, y + h]
        else:
            return box
    
    pred_xyxy = [to_xyxy(box, box_format) for box in pred_boxes]
    true_xyxy = [to_xyxy(box, box_format) for box in true_boxes]
    
    # For each prediction, find best matching ground truth
    for pred_box in pred_xyxy:
        best_iou = 0.0
        for true_box in true_xyxy:
            # Calculate intersection
            x1_i = max(pred_box[0], true_box[0])
            y1_i = max(pred_box[1], true_box[1])
            x2_i = min(pred_box[2], true_box[2])
            y2_i = min(pred_box[3], true_box[3])
            
            if x2_i > x1_i and y2_i > y1_i:
                intersection = (x2_i - x1_i) * (y2_i - y1_i)
            else:
                intersection = 0.0
            
            # Calculate areas
            pred_area = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
            true_area = (true_box[2] - true_box[0]) * (true_box[3] - true_box[1])
            union = pred_area + true_area - intersection
            
            # Calculate IoU
            iou = intersection / union if union > 0 else 0.0
            best_iou = max(best_iou, iou)
        
        ious.append(best_iou)
    
    return {
        "spatial_iou": np.mean(ious) if len(ious) > 0 else 0.0,
        "spatial_iou_std": np.std(ious) if len(ious) > 0 else 0.0
    }


def latency_to_recall(start_times: List[float], recall_times: List[float]) -> float:
    """
    Compute average latency to recall.
    
    Args:
        start_times: List of query start times.
        recall_times: List of recall completion times.
        
    Returns:
        Average latency in seconds.
    """
    if len(start_times) != len(recall_times) or len(start_times) == 0:
        return 0.0
    
    latencies = [recall - start for start, recall in zip(start_times, recall_times)]
    return np.mean(latencies)


def retrieval_speed(num_queries: int, total_time: float) -> float:
    """
    Compute retrieval speed (queries per second).
    
    Args:
        num_queries: Number of queries processed.
        total_time: Total time taken for retrieval in seconds.
        
    Returns:
        Queries per second.
    """
    if total_time <= 0:
        return 0.0
    return num_queries / total_time


class BenchmarkRunner:
    """Runs benchmarks on episodic memory retrieval models."""
    
    def __init__(self, models: Dict):
        """
        Initialize benchmark runner.
        
        Args:
            models: Dict mapping model names to EpisodicMemoryModel instances.
        """
        self.models = models
        self.results = []
    
    def _extract_ground_truth_indices(self, predicted_moments: List[Dict], 
                                     ground_truth: Dict, 
                                     video_fps: float = 30.0) -> Tuple[List[int], List[Tuple[float, float]]]:
        """
        Extract ground truth indices and intervals for evaluation.
        
        Args:
            predicted_moments: List of predicted moments from model.
            ground_truth: Ground truth dict with 'temporal' key.
            video_fps: FPS of video (for frame-to-time conversion).
            
        Returns:
            Tuple of (ground_truth_indices, ground_truth_intervals).
        """
        gt_indices = []
        gt_intervals = []
        
        if 'temporal' not in ground_truth:
            return gt_indices, gt_intervals
        
        temporal = ground_truth['temporal']
        
        # Extract temporal intervals
        if 'start_time' in temporal and 'end_time' in temporal:
            gt_intervals.append((temporal['start_time'], temporal['end_time']))
        elif 'start_frame' in temporal and 'end_frame' in temporal:
            start_time = temporal['start_frame'] / video_fps if video_fps > 0 else temporal['start_frame'] / 30.0
            end_time = temporal['end_frame'] / video_fps if video_fps > 0 else temporal['end_frame'] / 30.0
            gt_intervals.append((start_time, end_time))
        
        # Find which predicted moments match ground truth (for Recall@K)
        # A moment matches if it overlaps significantly with ground truth
        for i, pred_moment in enumerate(predicted_moments):
            pred_start = pred_moment.get('start_time', 0)
            pred_end = pred_moment.get('end_time', 0)
            
            for gt_start, gt_end in gt_intervals:
                # Check for overlap
                overlap_start = max(pred_start, gt_start)
                overlap_end = min(pred_end, gt_end)
                overlap = max(0.0, overlap_end - overlap_start)
                
                # Consider it a match if overlap > 50% of ground truth duration
                gt_duration = gt_end - gt_start
                if gt_duration > 0 and overlap / gt_duration > 0.5:
                    gt_indices.append(i)
                    break
        
        return gt_indices, gt_intervals
    
    def _extract_predicted_intervals(self, predicted_moments: List[Dict]) -> List[Tuple[float, float]]:
        """Extract temporal intervals from predicted moments."""
        intervals = []
        for moment in predicted_moments:
            start = moment.get('start_time', 0)
            end = moment.get('end_time', 0)
            if end > start:
                intervals.append((start, end))
        return intervals
    
    def _extract_spatial_boxes(self, predicted_moments: List[Dict], 
                              ground_truth: Dict) -> Tuple[List[List[float]], List[List[float]]]:
        """Extract spatial bounding boxes from predictions and ground truth."""
        pred_boxes = []
        true_boxes = []
        
        # Extract from ground truth
        if 'spatial' in ground_truth and 'bbox' in ground_truth['spatial']:
            true_boxes.append(ground_truth['spatial']['bbox'])
        
        # Extract from predictions (if available)
        # Note: Most models don't predict spatial boxes, so this may be empty
        for moment in predicted_moments:
            if 'bbox' in moment:
                pred_boxes.append(moment['bbox'])
        
        return pred_boxes, true_boxes
    
    def run_benchmark(self, video_path: str, query: str, 
                     ground_truth: Optional[Dict] = None,
                     top_k: int = 5) -> Dict:
        """
        Run all models on a query and collect results.
        
        Args:
            video_path: Path to video file.
            query: Text query.
            ground_truth: Optional ground truth dict with 'temporal' and 'spatial' keys.
            top_k: Number of top moments to retrieve.
            
        Returns:
            Dict with benchmark results for all models.
        """
        from pathlib import Path
        
        if not Path(video_path).exists():
            return None
        
        # Get video FPS for frame-time conversion
        import cv2
        cap = cv2.VideoCapture(video_path)
        video_fps = cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 30.0
        if cap.isOpened():
            cap.release()
        
        benchmark_results = {
            "video": video_path,
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "model_results": {}
        }
        
        for model_name, model in self.models.items():
            query_start_time = time.time()
            
            try:
                # Retrieve moments
                predicted_moments = model.retrieve_moments(video_path, query, top_k=top_k)
                query_end_time = time.time()
                
                latency = query_end_time - query_start_time
                
                # Initialize result dict
                result = {
                    "model_name": model_name,
                    "num_predictions": len(predicted_moments),
                    "latency_seconds": latency,
                    "predictions": predicted_moments,
                    "success": True
                }
                
                # Compute metrics if ground truth is available
                if ground_truth:
                    # Extract ground truth information
                    gt_indices, gt_intervals = self._extract_ground_truth_indices(
                        predicted_moments, ground_truth, video_fps
                    )
                    pred_intervals = self._extract_predicted_intervals(predicted_moments)
                    
                    # Recall@K metrics
                    pred_indices = list(range(len(predicted_moments)))
                    result["recall_at_1"] = recall_at_k(gt_indices, pred_indices, k=1)
                    result["recall_at_5"] = recall_at_k(gt_indices, pred_indices, k=5)
                    
                    # Temporal IoU
                    if len(gt_intervals) > 0 and len(pred_intervals) > 0:
                        result["temporal_iou"] = temporal_iou(pred_intervals, gt_intervals)
                    else:
                        result["temporal_iou"] = 0.0
                    
                    # Spatial localization metrics
                    pred_boxes, true_boxes = self._extract_spatial_boxes(predicted_moments, ground_truth)
                    if len(pred_boxes) > 0 and len(true_boxes) > 0:
                        spatial_metrics = spatial_localization_metrics(pred_boxes, true_boxes)
                        result.update(spatial_metrics)
                    else:
                        result["spatial_iou"] = None
                
                benchmark_results["model_results"][model_name] = result
                
            except Exception as e:
                query_end_time = time.time()
                latency = query_end_time - query_start_time
                
                benchmark_results["model_results"][model_name] = {
                    "model_name": model_name,
                    "error": str(e),
                    "latency_seconds": latency,
                    "success": False
                }
        
        self.results.append(benchmark_results)
        return benchmark_results
    
    def run_on_dataset(self, queries: List[Dict]) -> None:
        """
        Run benchmark on all queries from dataset.
        
        Args:
            queries: List of query dicts with 'video_path', 'query', 'ground_truth' keys.
        """
        from pathlib import Path
        
        skipped = 0
        total_queries = len(queries)
        
        for i, query_info in enumerate(queries):
            video_path = query_info.get("video_path")
            query = query_info.get("query", "")
            ground_truth = query_info.get("ground_truth", {})
            
            if not video_path or not Path(video_path).exists():
                skipped += 1
                continue
            
            if (i + 1) % 10 == 0:
                print(f"Processing query {i+1}/{total_queries}...")
            
            result = self.run_benchmark(video_path, query, ground_truth=ground_truth)
            if result is None:
                skipped += 1
        
        if skipped > 0:
            print(f"Skipped {skipped} query/queries (file not found or invalid)")
    
    def save_results(self, output_path: str = "Memory/results/benchmark_results.json") -> None:
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
            video = result.get("video", "")
            query = result.get("query", "")
            
            for model_name, model_result in result.get("model_results", {}).items():
                row = {
                    "video": video,
                    "query": query,
                    "model": model_name,
                    "success": model_result.get("success", False),
                    "num_predictions": model_result.get("num_predictions", 0),
                    "latency_seconds": model_result.get("latency_seconds", None)
                }
                
                # Add accuracy metrics
                if "recall_at_1" in model_result:
                    row["recall_at_1"] = model_result["recall_at_1"]
                if "recall_at_5" in model_result:
                    row["recall_at_5"] = model_result["recall_at_5"]
                if "temporal_iou" in model_result:
                    row["temporal_iou"] = model_result["temporal_iou"]
                if "spatial_iou" in model_result:
                    row["spatial_iou"] = model_result["spatial_iou"]
                
                report_data.append(row)
        
        return pd.DataFrame(report_data)
    
    def save_report(self, output_path: str = "Memory/results/benchmark_report.csv") -> None:
        """
        Save benchmark report to CSV.
        
        Args:
            output_path: Path to save CSV file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df = self.generate_report()
        df.to_csv(output_path, index=False)
    
    def compute_summary_statistics(self) -> Dict:
        """
        Compute summary statistics across all results.
        
        Returns:
            Dict with aggregated metrics per model.
        """
        df = self.generate_report()
        
        if len(df) == 0:
            return {}
        
        summary = {}
        
        for model_name in df['model'].unique():
            model_df = df[df['model'] == model_name]
            model_summary = {
                "num_queries": len(model_df),
                "success_rate": model_df['success'].mean() if 'success' in model_df.columns else 0.0
            }
            
            # Latency metrics
            if 'latency_seconds' in model_df.columns:
                latencies = model_df['latency_seconds'].dropna()
                if len(latencies) > 0:
                    model_summary["avg_latency"] = latencies.mean()
                    model_summary["std_latency"] = latencies.std()
                    model_summary["min_latency"] = latencies.min()
                    model_summary["max_latency"] = latencies.max()
            
            # Retrieval speed
            total_time = model_df['latency_seconds'].sum() if 'latency_seconds' in model_df.columns else 0
            if total_time > 0:
                model_summary["retrieval_speed_qps"] = retrieval_speed(len(model_df), total_time)
            
            # Accuracy metrics
            for metric in ['recall_at_1', 'recall_at_5', 'temporal_iou', 'spatial_iou']:
                if metric in model_df.columns:
                    values = model_df[metric].dropna()
                    if len(values) > 0:
                        model_summary[f"avg_{metric}"] = values.mean()
                        model_summary[f"std_{metric}"] = values.std()
            
            summary[model_name] = model_summary
        
        return summary





