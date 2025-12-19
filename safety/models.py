"""Routing model definitions for intent classification and NER."""

from abc import ABC, abstractmethod
import re
import pickle
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
from torch.utils.data import Dataset
import random


# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)


class RoutingModel(ABC):
    """Abstract base class for routing models."""
    
    @abstractmethod
    def predict(self, query: str) -> Dict[str, Any]:
        """
        Predict intent and entities for a query.
        
        Args:
            query: Input text query
            
        Returns:
            Dict with "intent" and "entities" keys.
            Entities is a list of dicts with "text", "start", "end", "entity_type"
        """
        pass
    
    @abstractmethod
    def train(self, train_data: List[Dict[str, Any]]) -> None:
        """
        Train the model on training data.
        
        Args:
            train_data: List of examples with "text", "intent", "entities"
        """
        pass


class KeywordBaseline(RoutingModel):
    """Rule-based keyword matching baseline."""
    
    def __init__(self):
        self.memory_keywords = {"search", "find", "retrieve", "look", "show", "display", "get"}
        self.safety_keywords = {"check", "weather", "rate", "rating", "evaluate", "assess"}
        self.task_keywords = {"book", "play", "add", "reserve", "schedule", "create"}
        
        # Simple entity patterns
        self.date_pattern = re.compile(r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{1,2}\s+\w+\s+\d{4}|\w+\s+\d{1,2},?\s+\d{4})\b', re.IGNORECASE)
        self.time_pattern = re.compile(r'\b(\d{1,2}:\d{2}\s*(?:am|pm)?|\d{1,2}\s*(?:am|pm))\b', re.IGNORECASE)
        self.location_pattern = re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b')
    
    def predict(self, query: str) -> Dict[str, Any]:
        """Predict using keyword matching."""
        query_lower = query.lower()
        
        # Intent classification
        memory_score = sum(1 for kw in self.memory_keywords if kw in query_lower)
        safety_score = sum(1 for kw in self.safety_keywords if kw in query_lower)
        task_score = sum(1 for kw in self.task_keywords if kw in query_lower)
        
        if memory_score >= safety_score and memory_score >= task_score:
            intent = "memory_retrieval"
        elif safety_score >= task_score:
            intent = "safety_check"
        else:
            intent = "task_guidance"
        
        # Simple entity extraction
        entities = []
        
        # Extract dates
        for match in self.date_pattern.finditer(query):
            entities.append({
                "text": match.group(0),
                "start": match.start(),
                "end": match.end(),
                "entity_type": "date"
            })
        
        # Extract times
        for match in self.time_pattern.finditer(query):
            entities.append({
                "text": match.group(0),
                "start": match.start(),
                "end": match.end(),
                "entity_type": "time"
            })
        
        # Extract potential locations (capitalized words)
        for match in self.location_pattern.finditer(query):
            # Skip if it's a common word or already matched
            text = match.group(0)
            if len(text.split()) <= 3 and text not in ["I", "The", "A", "An"]:
                # Check if not already in entities
                if not any(e["start"] <= match.start() < e["end"] for e in entities):
                    entities.append({
                        "text": text,
                        "start": match.start(),
                        "end": match.end(),
                        "entity_type": "location"
                    })
        
        return {"intent": intent, "entities": entities}
    
    def train(self, train_data: List[Dict[str, Any]]) -> None:
        """No training needed for rule-based model."""
        pass


class TFIDFLogisticRegression(RoutingModel):
    """TF-IDF + Logistic Regression for intent classification only."""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        self.classifier = LogisticRegression(random_state=42, max_iter=1000)
        self.intent_to_idx = {}
        self.idx_to_intent = {}
        self.is_trained = False
    
    def train(self, train_data: List[Dict[str, Any]]) -> None:
        """Train TF-IDF + Logistic Regression model."""
        texts = [ex["text"] for ex in train_data]
        intents = [ex["intent"] for ex in train_data]
        
        # Create intent mappings
        unique_intents = sorted(set(intents))
        self.intent_to_idx = {intent: idx for idx, intent in enumerate(unique_intents)}
        self.idx_to_intent = {idx: intent for intent, idx in self.intent_to_idx.items()}
        
        # Vectorize texts
        X = self.vectorizer.fit_transform(texts)
        y = [self.intent_to_idx[intent] for intent in intents]
        
        # Train classifier
        self.classifier.fit(X, y)
        self.is_trained = True
    
    def predict(self, query: str) -> Dict[str, Any]:
        """Predict intent (no NER)."""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        # Vectorize query
        X = self.vectorizer.transform([query])
        
        # Predict intent
        intent_idx = self.classifier.predict(X)[0]
        intent = self.idx_to_intent[intent_idx]
        
        # No entity extraction
        return {"intent": intent, "entities": []}


class DistilBERTModel(RoutingModel):
    """DistilBERT for intent classification and NER."""
    
    def __init__(self, model_name: str = "distilbert-base-uncased", device: Optional[str] = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.intent_tokenizer = None
        self.intent_model = None
        self.ner_tokenizer = None
        self.ner_model = None
        self.intent_to_id = {}
        self.id_to_intent = {}
        self.entity_to_id = {}
        self.id_to_entity = {}
        self.is_trained = False
    
    def train(self, train_data: List[Dict[str, Any]]) -> None:
        """Train DistilBERT models for intent and NER."""
        # Prepare intent data
        texts = [ex["text"] for ex in train_data]
        intents = [ex["intent"] for ex in train_data]
        
        unique_intents = sorted(set(intents))
        self.intent_to_id = {intent: idx for idx, intent in enumerate(unique_intents)}
        self.id_to_intent = {idx: intent for intent, idx in self.intent_to_id.items()}
        
        # Initialize intent model
        self.intent_tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.intent_model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=len(unique_intents)
        ).to(self.device)
        
        # Train intent classifier (simplified - just fine-tune on a small sample)
        # For full training, we'd use Trainer, but for minimal implementation:
        self.intent_model.train()
        optimizer = torch.optim.AdamW(self.intent_model.parameters(), lr=2e-5)
        
        # Simple training loop (minimal epochs for speed)
        for epoch in range(2):
            for i in range(0, len(texts), 32):  # Batch size 32
                batch_texts = texts[i:i+32]
                batch_intents = intents[i:i+32]
                
                inputs = self.intent_tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors="pt"
                ).to(self.device)
                
                labels = torch.tensor([self.intent_to_id[intent] for intent in batch_intents]).to(self.device)
                
                outputs = self.intent_model(**inputs, labels=labels)
                loss = outputs.loss
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        self.intent_model.eval()
        
        # Prepare NER data
        all_entity_types = set()
        for ex in train_data:
            for entity in ex["entities"]:
                all_entity_types.add(entity["entity_type"])
        
        # Add O (outside) label
        all_entity_types = sorted(all_entity_types)
        self.entity_to_id = {"O": 0}
        for i, entity_type in enumerate(all_entity_types, 1):
            self.entity_to_id[entity_type] = i
        self.id_to_entity = {v: k for k, v in self.entity_to_id.items()}
        
        # Initialize NER model
        self.ner_tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.ner_model = AutoModelForTokenClassification.from_pretrained(
            self.model_name,
            num_labels=len(self.entity_to_id)
        ).to(self.device)
        
        # Train NER model (simplified)
        self.ner_model.train()
        ner_optimizer = torch.optim.AdamW(self.ner_model.parameters(), lr=2e-5)
        
        # Create NER training data
        ner_examples = []
        for ex in train_data:
            text = ex["text"]
            entities = ex["entities"]
            
            # Tokenize with offset mapping
            tokenized = self.ner_tokenizer(
                text,
                return_offsets_mapping=True,
                add_special_tokens=True,
                truncation=True,
                max_length=128
            )
            
            # Create token-level tags (BIO format)
            token_tags = [0] * len(tokenized["input_ids"])  # 0 = O label
            
            # Map entities to tokens using offsets
            for entity in entities:
                entity_start = entity["start"]
                entity_end = entity["end"]
                entity_type = entity["entity_type"]
                entity_label_id = self.entity_to_id.get(entity_type, 0)
                
                if entity_label_id == 0:
                    continue
                
                # Find tokens that overlap with this entity
                first_token_idx = None
                for token_idx, (char_start, char_end) in enumerate(tokenized["offset_mapping"]):
                    # Skip special tokens (offset is (0, 0))
                    if char_start == 0 and char_end == 0:
                        continue
                    
                    # Check if token overlaps with entity
                    if char_start < entity_end and char_end > entity_start:
                        if first_token_idx is None:
                            first_token_idx = token_idx
                            token_tags[token_idx] = entity_label_id  # B- tag
                        else:
                            token_tags[token_idx] = entity_label_id  # I- tag (same ID for simplicity)
            
            ner_examples.append((text, token_tags))
        
        # Simple NER training
        for epoch in range(2):
            for i in range(0, len(ner_examples), 16):  # Smaller batch size
                batch_texts = [ex[0] for ex in ner_examples[i:i+16]]
                batch_tags = [ex[1] for ex in ner_examples[i:i+16]]
                
                inputs = self.ner_tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors="pt"
                ).to(self.device)
                
                # Pad tags to match input length
                max_len = inputs["input_ids"].shape[1]
                padded_tags = []
                for tags in batch_tags:
                    # Truncate or pad to max_len
                    if len(tags) > max_len:
                        padded = tags[:max_len]
                    else:
                        padded = tags + [0] * (max_len - len(tags))
                    padded_tags.append(padded)
                
                labels = torch.tensor(padded_tags).to(self.device)
                
                outputs = self.ner_model(**inputs, labels=labels)
                loss = outputs.loss
                
                ner_optimizer.zero_grad()
                loss.backward()
                ner_optimizer.step()
        
        self.ner_model.eval()
        self.is_trained = True
    
    def predict(self, query: str) -> Dict[str, Any]:
        """Predict intent and entities."""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        # Predict intent
        intent_inputs = self.intent_tokenizer(
            query,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            intent_outputs = self.intent_model(**intent_inputs)
            intent_logits = intent_outputs.logits
            intent_idx = intent_logits.argmax(-1).item()
            intent = self.id_to_intent[intent_idx]
        
        # Predict entities
        ner_inputs = self.ner_tokenizer(
            query,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
            return_offsets_mapping=True
        ).to(self.device)
        
        offsets = ner_inputs.pop("offset_mapping")[0].cpu().numpy()
        
        with torch.no_grad():
            ner_outputs = self.ner_model(**ner_inputs)
            ner_logits = ner_outputs.logits
            ner_predictions = ner_logits.argmax(-1)[0].cpu().numpy()
        
        # Extract entities from predictions
        entities = []
        current_entity = None
        
        for i, (pred_id, offset) in enumerate(zip(ner_predictions, offsets)):
            char_start, char_end = offset
            
            # Skip special tokens (offset is (0, 0))
            if char_start == 0 and char_end == 0:
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
                continue
            
            if pred_id == 0:  # O label
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
            else:
                entity_type = self.id_to_entity[pred_id]
                
                if current_entity and current_entity["entity_type"] == entity_type:
                    # Extend entity (consecutive tokens of same type)
                    current_entity["end"] = char_end
                    current_entity["text"] = query[current_entity["start"]:char_end]
                else:
                    # Start new entity
                    if current_entity:
                        entities.append(current_entity)
                    current_entity = {
                        "text": query[char_start:char_end],
                        "start": char_start,
                        "end": char_end,
                        "entity_type": entity_type
                    }
        
        if current_entity:
            entities.append(current_entity)
        
        return {"intent": intent, "entities": entities}


class BERTMiniModel(RoutingModel):
    """BERT-mini for intent classification and NER."""
    
    def __init__(self, model_name: str = "prajjwal1/bert-mini", device: Optional[str] = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.intent_tokenizer = None
        self.intent_model = None
        self.ner_tokenizer = None
        self.ner_model = None
        self.intent_to_id = {}
        self.id_to_intent = {}
        self.entity_to_id = {}
        self.id_to_entity = {}
        self.is_trained = False
    
    def train(self, train_data: List[Dict[str, Any]]) -> None:
        """Train BERT-mini models for intent and NER."""
        # Same implementation as DistilBERT
        distilbert = DistilBERTModel(model_name=self.model_name, device=self.device)
        distilbert.train(train_data)
        
        # Copy trained models
        self.intent_tokenizer = distilbert.intent_tokenizer
        self.intent_model = distilbert.intent_model
        self.ner_tokenizer = distilbert.ner_tokenizer
        self.ner_model = distilbert.ner_model
        self.intent_to_id = distilbert.intent_to_id
        self.id_to_intent = distilbert.id_to_intent
        self.entity_to_id = distilbert.entity_to_id
        self.id_to_entity = distilbert.id_to_entity
        self.is_trained = True
    
    def predict(self, query: str) -> Dict[str, Any]:
        """Predict intent and entities."""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        # Predict intent
        intent_inputs = self.intent_tokenizer(
            query,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            intent_outputs = self.intent_model(**intent_inputs)
            intent_logits = intent_outputs.logits
            intent_idx = intent_logits.argmax(-1).item()
            intent = self.id_to_intent[intent_idx]
        
        # Predict entities
        ner_inputs = self.ner_tokenizer(
            query,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
            return_offsets_mapping=True
        ).to(self.device)
        
        offsets = ner_inputs.pop("offset_mapping")[0].cpu().numpy()
        
        with torch.no_grad():
            ner_outputs = self.ner_model(**ner_inputs)
            ner_logits = ner_outputs.logits
            ner_predictions = ner_logits.argmax(-1)[0].cpu().numpy()
        
        # Extract entities from predictions
        entities = []
        current_entity = None
        
        for i, (pred_id, offset) in enumerate(zip(ner_predictions, offsets)):
            char_start, char_end = offset
            
            # Skip special tokens (offset is (0, 0))
            if char_start == 0 and char_end == 0:
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
                continue
            
            if pred_id == 0:  # O label
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
            else:
                entity_type = self.id_to_entity[pred_id]
                
                if current_entity and current_entity["entity_type"] == entity_type:
                    # Extend entity (consecutive tokens of same type)
                    current_entity["end"] = char_end
                    current_entity["text"] = query[current_entity["start"]:char_end]
                else:
                    # Start new entity
                    if current_entity:
                        entities.append(current_entity)
                    current_entity = {
                        "text": query[char_start:char_end],
                        "start": char_start,
                        "end": char_end,
                        "entity_type": entity_type
                    }
        
        if current_entity:
            entities.append(current_entity)
        
        return {"intent": intent, "entities": entities}

