"""Dataset loaders and preprocessing for SNIPS benchmarking."""

import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
from sklearn.model_selection import train_test_split


def map_intent_to_system(snips_intent: str) -> str:
    """
    Map SNIPS intent to system intent.
    
    Args:
        snips_intent: Original SNIPS intent name
        
    Returns:
        System intent: memory_retrieval, safety_check, or task_guidance
    """
    memory_intents = {"SearchCreativeWork", "SearchScreeningEvent"}
    safety_intents = {"GetWeather", "RateBook"}
    task_intents = {"BookRestaurant", "PlayMusic", "AddToPlaylist"}
    
    if snips_intent in memory_intents:
        return "memory_retrieval"
    elif snips_intent in safety_intents:
        return "safety_check"
    elif snips_intent in task_intents:
        return "task_guidance"
    else:
        raise ValueError(f"Unknown SNIPS intent: {snips_intent}")


def parse_snips_example(example: Dict[str, Any], intent: str) -> Dict[str, Any]:
    """
    Parse a single SNIPS example into unified format.
    
    Args:
        example: SNIPS example dict with "data" field
        intent: SNIPS intent name
        
    Returns:
        Dict with text, intent, entities, original_intent
    """
    data = example.get("data", [])
    
    # Build full text and track character positions
    full_text = ""
    entities = []
    char_pos = 0
    
    for segment in data:
        text = segment.get("text", "")
        entity_type = segment.get("entity")
        
        start_pos = char_pos
        full_text += text
        end_pos = char_pos + len(text)
        
        if entity_type:
            entities.append({
                "text": text,
                "start": start_pos,
                "end": end_pos,
                "entity_type": entity_type
            })
        
        char_pos = end_pos
    
    system_intent = map_intent_to_system(intent)
    
    return {
        "text": full_text,
        "intent": system_intent,
        "entities": entities,
        "original_intent": intent
    }


def load_snips_dataset(dataset_path: str) -> List[Dict[str, Any]]:
    """
    Load all SNIPS JSON files from dataset directory.
    
    Args:
        dataset_path: Path to SNIPS dataset root (2017-06-custom-intent-engines folder)
        
    Returns:
        List of parsed examples in unified format
    """
    dataset_path = Path(dataset_path)
    all_examples = []
    
    # Intent folders to process
    intent_folders = [
        "AddToPlaylist",
        "BookRestaurant",
        "GetWeather",
        "PlayMusic",
        "RateBook",
        "SearchCreativeWork",
        "SearchScreeningEvent"
    ]
    
    for intent_folder in intent_folders:
        intent_path = dataset_path / intent_folder
        
        if not intent_path.exists():
            continue
        
        # Load train and validate files
        train_file = intent_path / f"train_{intent_folder}.json"
        validate_file = intent_path / f"validate_{intent_folder}.json"
        
        for json_file in [train_file, validate_file]:
            if json_file.exists():
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    # SNIPS format: {"IntentName": [examples...]}
                    intent_name = list(data.keys())[0]
                    examples = data[intent_name]
                    
                    for example in examples:
                        parsed = parse_snips_example(example, intent_name)
                        all_examples.append(parsed)
    
    return all_examples


def preprocess_data(raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Preprocess raw SNIPS data (already parsed).
    
    This function can be used for additional preprocessing if needed.
    Currently, data is already in the correct format from parse_snips_example.
    
    Args:
        raw_data: List of parsed examples
        
    Returns:
        Preprocessed data (same format)
    """
    return raw_data


def create_splits(
    data: List[Dict[str, Any]], 
    test_size: float = 0.2, 
    random_state: int = 42
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Create deterministic train/test splits.
    
    Args:
        data: List of examples
        test_size: Proportion of data for test set
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (train_data, test_data)
    """
    train_data, test_data = train_test_split(
        data,
        test_size=test_size,
        random_state=random_state,
        shuffle=True
    )
    
    return train_data, test_data

