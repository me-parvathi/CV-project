"""Benchmarking framework for routing models."""

from typing import List, Dict, Any
import pandas as pd


def compute_intent_accuracy(
    predictions: List[Dict[str, Any]], 
    ground_truth: List[Dict[str, Any]]
) -> float:
    """
    Compute intent classification accuracy.
    
    Args:
        predictions: List of prediction dicts with "intent" key
        ground_truth: List of ground truth dicts with "intent" key
        
    Returns:
        Accuracy score (0.0 to 1.0)
    """
    if len(predictions) != len(ground_truth):
        raise ValueError("Predictions and ground truth must have same length")
    
    correct = sum(
        1 for pred, gt in zip(predictions, ground_truth)
        if pred["intent"] == gt["intent"]
    )
    
    return correct / len(ground_truth) if len(ground_truth) > 0 else 0.0


def compute_entity_f1(
    predictions: List[Dict[str, Any]], 
    ground_truth: List[Dict[str, Any]]
) -> Dict[str, float]:
    """
    Compute entity-level F1 score (micro-averaged).
    
    Args:
        predictions: List of prediction dicts with "entities" key
        ground_truth: List of ground truth dicts with "entities" key
        
    Returns:
        Dict with "precision", "recall", "f1" keys
    """
    if len(predictions) != len(ground_truth):
        raise ValueError("Predictions and ground truth must have same length")
    
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    for pred, gt in zip(predictions, ground_truth):
        pred_entities = pred.get("entities", [])
        gt_entities = gt.get("entities", [])
        
        # Create sets of entity tuples (start, end, entity_type) for matching
        pred_set = set()
        for entity in pred_entities:
            pred_set.add((
                entity.get("start", 0),
                entity.get("end", 0),
                entity.get("entity_type", "")
            ))
        
        gt_set = set()
        for entity in gt_entities:
            gt_set.add((
                entity.get("start", 0),
                entity.get("end", 0),
                entity.get("entity_type", "")
            ))
        
        # Count matches
        true_positives += len(pred_set & gt_set)
        false_positives += len(pred_set - gt_set)
        false_negatives += len(gt_set - pred_set)
    
    # Compute metrics
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def compute_routing_accuracy(
    predictions: List[Dict[str, Any]], 
    ground_truth: List[Dict[str, Any]]
) -> float:
    """
    Compute end-to-end routing accuracy (intent + all entities correct).
    
    Args:
        predictions: List of prediction dicts with "intent" and "entities" keys
        ground_truth: List of ground truth dicts with "intent" and "entities" keys
        
    Returns:
        Routing accuracy score (0.0 to 1.0)
    """
    if len(predictions) != len(ground_truth):
        raise ValueError("Predictions and ground truth must have same length")
    
    correct = 0
    
    for pred, gt in zip(predictions, ground_truth):
        # Check intent
        if pred["intent"] != gt["intent"]:
            continue
        
        # Check all entities match exactly
        pred_entities = pred.get("entities", [])
        gt_entities = gt.get("entities", [])
        
        # Create sets for comparison
        pred_set = set()
        for entity in pred_entities:
            pred_set.add((
                entity.get("start", 0),
                entity.get("end", 0),
                entity.get("entity_type", "")
            ))
        
        gt_set = set()
        for entity in gt_entities:
            gt_set.add((
                entity.get("start", 0),
                entity.get("end", 0),
                entity.get("entity_type", "")
            ))
        
        # All entities must match
        if pred_set == gt_set:
            correct += 1
    
    return correct / len(ground_truth) if len(ground_truth) > 0 else 0.0


class BenchmarkRunner:
    """Runs benchmarks on routing models."""
    
    def __init__(self):
        self.results = {}
    
    def run_benchmark(
        self, 
        models: Dict[str, Any], 
        test_data: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Run benchmarks on all models.
        
        Args:
            models: Dict mapping model names to RoutingModel instances
            test_data: List of test examples with "text", "intent", "entities"
            
        Returns:
            Dict mapping model names to metric dicts
        """
        results = {}
        
        for model_name, model in models.items():
            print(f"Evaluating {model_name}...")
            
            # Get predictions
            predictions = []
            for example in test_data:
                try:
                    pred = model.predict(example["text"])
                    predictions.append(pred)
                except Exception as e:
                    print(f"  Warning: Error predicting for {model_name}: {e}")
                    # Add dummy prediction
                    predictions.append({"intent": "task_guidance", "entities": []})
            
            # Prepare ground truth
            ground_truth = [
                {"intent": ex["intent"], "entities": ex["entities"]}
                for ex in test_data
            ]
            
            # Compute metrics
            intent_acc = compute_intent_accuracy(predictions, ground_truth)
            entity_metrics = compute_entity_f1(predictions, ground_truth)
            routing_acc = compute_routing_accuracy(predictions, ground_truth)
            
            results[model_name] = {
                "intent_accuracy": intent_acc,
                "entity_f1": entity_metrics["f1"],
                "entity_precision": entity_metrics["precision"],
                "entity_recall": entity_metrics["recall"],
                "routing_accuracy": routing_acc
            }
        
        self.results = results
        return results
    
    def generate_summary_table(self) -> pd.DataFrame:
        """
        Generate summary table of results.
        
        Returns:
            DataFrame with model results
        """
        if not self.results:
            return pd.DataFrame()
        
        rows = []
        for model_name, metrics in self.results.items():
            rows.append({
                "Model": model_name,
                "Intent Accuracy": metrics["intent_accuracy"],
                "Entity F1": metrics["entity_f1"],
                "Routing Accuracy": metrics["routing_accuracy"]
            })
        
        df = pd.DataFrame(rows)
        return df
    
    def print_summary(self) -> None:
        """Print formatted summary table."""
        df = self.generate_summary_table()
        
        if df.empty:
            print("No results to display.")
            return
        
        print("\nBenchmark Results Summary:")
        print("=" * 60)
        
        # Format table
        print(f"{'Model':<30} | {'Intent Acc':<12} | {'Entity F1':<12} | {'Routing Acc':<12}")
        print("-" * 60)
        
        for _, row in df.iterrows():
            model_name = row["Model"]
            intent_acc = f"{row['Intent Accuracy']:.3f}"
            entity_f1 = f"{row['Entity F1']:.3f}"
            routing_acc = f"{row['Routing Accuracy']:.3f}"
            
            print(f"{model_name:<30} | {intent_acc:<12} | {entity_f1:<12} | {routing_acc:<12}")
        
        print("=" * 60)

