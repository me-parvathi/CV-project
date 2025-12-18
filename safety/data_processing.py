"""Dataset loaders and preprocessing for SNIPS benchmarking."""

import json
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from sklearn.model_selection import train_test_split
from datasets import load_dataset


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


def load_snips_dataset(dataset_path: Optional[str] = None, use_huggingface: bool = True) -> List[Dict[str, Any]]:
    """
    Load SNIPS dataset from HuggingFace or local directory.
    
    Args:
        dataset_path: Path to SNIPS dataset root (2017-06-custom-intent-engines folder).
                     Only used if use_huggingface=False.
        use_huggingface: If True, load from HuggingFace. If False, load from local files.
        
    Returns:
        List of parsed examples in unified format
    """
    if use_huggingface:
        return _load_snips_from_huggingface()
    else:
        return _load_snips_from_local(dataset_path)


def _load_snips_from_huggingface() -> List[Dict[str, Any]]:
    """
    Load SNIPS dataset from HuggingFace.
    
    Returns:
        List of parsed examples in unified format
    """
    try:
        # Load SNIPS dataset from HuggingFace
        # Try multiple possible dataset names
        dataset_names = [
            "snips_built_in_intents",
            "AutoIntent/snips",
            "snips",
        ]
        
        dataset = None
        for name in dataset_names:
            try:
                dataset = load_dataset(name)
                print(f"Successfully loaded dataset: {name}")
                break
            except Exception as e:
                continue
        
        if dataset is None:
            # Fallback: try loading the custom intents version
            try:
                # Load individual intent datasets
                all_examples = []
                intent_datasets = {
                    "AddToPlaylist": "snips_custom_intent_AddToPlaylist",
                    "BookRestaurant": "snips_custom_intent_BookRestaurant",
                    "GetWeather": "snips_custom_intent_GetWeather",
                    "PlayMusic": "snips_custom_intent_PlayMusic",
                    "RateBook": "snips_custom_intent_RateBook",
                    "SearchCreativeWork": "snips_custom_intent_SearchCreativeWork",
                    "SearchScreeningEvent": "snips_custom_intent_SearchScreeningEvent",
                }
                
                for intent_name, dataset_name in intent_datasets.items():
                    try:
                        intent_dataset = load_dataset(dataset_name)
                        # Process each split (train, validation, test)
                        for split_name in intent_dataset.keys():
                            for example in intent_dataset[split_name]:
                                parsed = _parse_huggingface_example(example, intent_name)
                                if parsed:
                                    all_examples.append(parsed)
                    except Exception as e:
                        print(f"Warning: Could not load {dataset_name}: {e}")
                        continue
                
                if all_examples:
                    return all_examples
            except Exception as e:
                pass
            
            raise ValueError(f"Could not load SNIPS dataset from HuggingFace. Tried: {dataset_names}")
        
        # Process the loaded dataset
        all_examples = []
        
        # Handle different dataset structures
        if isinstance(dataset, dict):
            # Dataset has multiple splits
            for split_name, split_data in dataset.items():
                for example in split_data:
                    # Extract intent and entities from HuggingFace format
                    parsed = _parse_huggingface_example(example)
                    if parsed:
                        all_examples.append(parsed)
        else:
            # Single dataset object
            for example in dataset:
                parsed = _parse_huggingface_example(example)
                if parsed:
                    all_examples.append(parsed)
        
        return all_examples
        
    except Exception as e:
        print(f"Error loading from HuggingFace: {e}")
        print("Falling back to local file loading...")
        raise


def _parse_huggingface_example(example: Dict[str, Any], intent_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Parse a HuggingFace dataset example into unified format.
    
    Args:
        example: Example from HuggingFace dataset
        intent_name: Optional intent name if not in example
        
    Returns:
        Parsed example dict or None if parsing fails
    """
    try:
        # Extract text
        text = example.get("text", example.get("utterance", example.get("query", "")))
        if not text:
            return None
        
        # Extract intent
        intent = intent_name or example.get("intent", example.get("intent_name", ""))
        if not intent:
            return None
        
        # Extract entities
        entities = []
        
        # Try different entity formats
        if "entities" in example:
            # Format: [{"text": "...", "entity": "...", "start": int, "end": int}]
            for entity in example["entities"]:
                entities.append({
                    "text": entity.get("text", entity.get("value", "")),
                    "start": entity.get("start", 0),
                    "end": entity.get("end", 0),
                    "entity_type": entity.get("entity", entity.get("entity_type", entity.get("slot_name", "")))
                })
        elif "slots" in example:
            # Format: [{"slot_name": "...", "value": "...", "start": int, "end": int}]
            for slot in example["slots"]:
                entities.append({
                    "text": slot.get("value", ""),
                    "start": slot.get("start", 0),
                    "end": slot.get("end", 0),
                    "entity_type": slot.get("slot_name", slot.get("entity", ""))
                })
        elif "tags" in example:
            # BIO tagging format - need to convert to spans
            tags = example["tags"]
            words = text.split() if isinstance(tags, list) else []
            if len(tags) == len(words):
                current_entity = None
                char_pos = 0
                for word, tag in zip(words, tags):
                    word_start = text.find(word, char_pos)
                    if word_start == -1:
                        word_start = char_pos
                    word_end = word_start + len(word)
                    
                    if tag.startswith("B-") or (tag.startswith("I-") and current_entity is None):
                        # Start new entity
                        if current_entity:
                            entities.append(current_entity)
                        entity_type = tag.split("-", 1)[1] if "-" in tag else tag
                        current_entity = {
                            "text": word,
                            "start": word_start,
                            "end": word_end,
                            "entity_type": entity_type
                        }
                    elif tag.startswith("I-") and current_entity:
                        # Extend entity
                        current_entity["end"] = word_end
                        current_entity["text"] = text[current_entity["start"]:word_end]
                    else:
                        # O tag or end of entity
                        if current_entity:
                            entities.append(current_entity)
                            current_entity = None
                    
                    char_pos = word_end
                
                if current_entity:
                    entities.append(current_entity)
        
        # Map intent to system intent
        system_intent = map_intent_to_system(intent)
        
        return {
            "text": text,
            "intent": system_intent,
            "entities": entities,
            "original_intent": intent
        }
    except Exception as e:
        print(f"Warning: Failed to parse example: {e}")
        return None


def _load_snips_from_local(dataset_path: str) -> List[Dict[str, Any]]:
    """
    Load SNIPS dataset from local JSON files (fallback method).
    
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

