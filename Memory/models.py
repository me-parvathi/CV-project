"""Episodic memory retrieval model definitions."""

from abc import ABC, abstractmethod
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import cv2
from PIL import Image


class EpisodicMemoryModel(ABC):
    """Abstract base class for episodic memory retrieval models."""
    
    @abstractmethod
    def retrieve_moments(self, video_path: str, query: str, top_k: int = 5) -> List[Dict]:
        """
        Retrieve relevant moments from video given a text query.
        
        Args:
            video_path: Path to video file.
            query: Text query describing the moment to retrieve.
            top_k: Number of top moments to return.
            
        Returns:
            List of dicts, each containing:
            - start_time: float (seconds)
            - end_time: float (seconds)
            - start_frame: int
            - end_frame: int
            - score: float (confidence/similarity score)
            - model_name: str
        """
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """Return the name of the model."""
        pass


class MomentRetrievalModel(EpisodicMemoryModel):
    """Moment-DETR model for video moment retrieval."""
    
    def __init__(self, device: str = "cuda"):
        """
        Initialize Moment-DETR model.
        
        Args:
            device: Device to run model on ('cuda' or 'cpu').
        """
        self.device = device
        self.model_name = "Moment-DETR"
        try:
            from transformers import AutoProcessor, AutoModelForVideoGrounding
            
            # Try to load Moment-DETR model
            # Note: The exact model name may vary, using a common one
            model_id = "facebook/moment_detr"
            try:
                self.processor = AutoProcessor.from_pretrained(model_id)
                self.model = AutoModelForVideoGrounding.from_pretrained(model_id).to(device)
                self.model.eval()
                self.available = True
            except Exception as e:
                # Fallback: use CLIP-based approach if Moment-DETR not available
                print(f"Warning: Could not load Moment-DETR ({model_id}): {e}")
                print("Falling back to CLIP-based moment retrieval...")
                self.available = False
                self.error = str(e)
                # Initialize CLIP as fallback
                from transformers import CLIPProcessor, CLIPModel
                self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
                self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
                self.clip_model.eval()
                self.use_clip_fallback = True
        except ImportError:
            self.available = False
            self.error = "transformers library not available"
            self.use_clip_fallback = False
    
    def get_model_name(self) -> str:
        return self.model_name
    
    def _extract_frames(self, video_path: str, num_frames: int = 16) -> List[np.ndarray]:
        """Extract frames from video."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        if total_frames == 0:
            cap.release()
            return []
        
        # Sample frames uniformly
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        frames = []
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
        
        cap.release()
        return frames
    
    def _clip_based_retrieval(self, video_path: str, query: str, top_k: int = 5) -> List[Dict]:
        """Fallback CLIP-based retrieval if Moment-DETR is not available."""
        frames = self._extract_frames(video_path, num_frames=32)
        if len(frames) == 0:
            return []
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else len(frames)
        duration = total_frames / fps if fps > 0 else len(frames) / 30.0
        if cap.isOpened():
            cap.release()
        
        # Encode query
        query_inputs = self.clip_processor(text=[query], return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            query_embedding = self.clip_model.get_text_features(**query_inputs)
            query_embedding = query_embedding / query_embedding.norm(dim=-1, keepdim=True)
        
        # Encode frames
        frame_images = [Image.fromarray(frame) for frame in frames]
        frame_inputs = self.clip_processor(images=frame_images, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            frame_embeddings = self.clip_model.get_image_features(**frame_inputs)
            frame_embeddings = frame_embeddings / frame_embeddings.norm(dim=-1, keepdim=True)
        
        # Compute similarities
        similarities = (frame_embeddings @ query_embedding.T).squeeze().cpu().numpy()
        
        # Group frames into moments (sliding window approach)
        window_size = max(4, len(frames) // 8)  # Adaptive window size
        moments = []
        
        for i in range(0, len(frames) - window_size + 1, window_size // 2):
            window_similarities = similarities[i:i+window_size]
            avg_score = np.mean(window_similarities)
            
            start_frame_idx = i
            end_frame_idx = min(i + window_size, len(frames) - 1)
            
            # Convert to actual frame numbers and times
            start_frame = int((start_frame_idx / len(frames)) * total_frames)
            end_frame = int((end_frame_idx / len(frames)) * total_frames)
            start_time = start_frame / fps if fps > 0 else (start_frame_idx / len(frames)) * duration
            end_time = end_frame / fps if fps > 0 else (end_frame_idx / len(frames)) * duration
            
            moments.append({
                "start_time": start_time,
                "end_time": end_time,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "score": float(avg_score),
                "model_name": self.model_name
            })
        
        # Sort by score and return top_k
        moments.sort(key=lambda x: x["score"], reverse=True)
        return moments[:top_k]
    
    def retrieve_moments(self, video_path: str, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve moments using Moment-DETR or CLIP fallback."""
        if not Path(video_path).exists():
            return []
        
        if not self.available or (hasattr(self, 'use_clip_fallback') and self.use_clip_fallback):
            return self._clip_based_retrieval(video_path, query, top_k)
        
        try:
            # Extract frames
            frames = self._extract_frames(video_path, num_frames=16)
            if len(frames) == 0:
                return []
            
            # Get video metadata
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else len(frames)
            if cap.isOpened():
                cap.release()
            
            # Prepare inputs for Moment-DETR
            frame_images = [Image.fromarray(frame) for frame in frames]
            inputs = self.processor(
                text=[query],
                videos=[frame_images],
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Extract predictions
            # Moment-DETR typically returns temporal boundaries
            # Note: Actual output format may vary based on model version
            if hasattr(outputs, 'pred_boxes') or hasattr(outputs, 'logits'):
                # Process outputs to get temporal segments
                # This is a simplified version - actual implementation depends on model output format
                moments = []
                
                # If model returns logits or scores, extract top segments
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits.cpu().numpy()
                    # Extract top segments (simplified)
                    for i in range(min(top_k, len(logits))):
                        # This is a placeholder - actual parsing depends on model output
                        start_time = i * (total_frames / fps / len(logits))
                        end_time = (i + 1) * (total_frames / fps / len(logits))
                        score = float(logits[i]) if isinstance(logits[i], (int, float, np.number)) else 0.5
                        
                        moments.append({
                            "start_time": start_time,
                            "end_time": end_time,
                            "start_frame": int(start_time * fps),
                            "end_frame": int(end_time * fps),
                            "score": score,
                            "model_name": self.model_name
                        })
                
                return moments[:top_k] if moments else self._clip_based_retrieval(video_path, query, top_k)
            else:
                # Fallback to CLIP if output format is unexpected
                return self._clip_based_retrieval(video_path, query, top_k)
        
        except Exception as e:
            print(f"Error in Moment-DETR retrieval: {e}")
            return self._clip_based_retrieval(video_path, query, top_k)


class CLIPModel(EpisodicMemoryModel):
    """CLIP model for video-text retrieval."""
    
    def __init__(self, device: str = "cuda", model_name: str = "openai/clip-vit-base-patch32"):
        """
        Initialize CLIP model.
        
        Args:
            device: Device to run model on ('cuda' or 'cpu').
            model_name: HuggingFace model identifier for CLIP.
        """
        self.device = device
        self.model_name = "CLIP"
        self.hf_model_name = model_name
        try:
            from transformers import CLIPProcessor, CLIPModel as HFCLIPModel
            
            self.processor = CLIPProcessor.from_pretrained(model_name)
            self.model = HFCLIPModel.from_pretrained(model_name).to(device)
            self.model.eval()
            self.available = True
        except Exception as e:
            self.available = False
            self.error = str(e)
    
    def get_model_name(self) -> str:
        return self.model_name
    
    def _extract_frames(self, video_path: str, num_frames: int = 32) -> Tuple[List[np.ndarray], float, int]:
        """Extract frames from video and return metadata."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return [], 30.0, 0
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        
        if total_frames == 0:
            cap.release()
            return [], fps, 0
        
        # Sample frames uniformly
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        frames = []
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
        
        cap.release()
        return frames, fps, total_frames
    
    def retrieve_moments(self, video_path: str, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve moments using CLIP similarity."""
        if not self.available:
            raise RuntimeError(f"CLIP model unavailable: {self.error}")
        
        if not Path(video_path).exists():
            return []
        
        # Extract frames
        frames, fps, total_frames = self._extract_frames(video_path, num_frames=32)
        if len(frames) == 0:
            return []
        
        duration = total_frames / fps if fps > 0 else len(frames) / 30.0
        
        # Encode query
        query_inputs = self.processor(text=[query], return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            query_embedding = self.model.get_text_features(**query_inputs)
            query_embedding = query_embedding / query_embedding.norm(dim=-1, keepdim=True)
        
        # Encode frames
        frame_images = [Image.fromarray(frame) for frame in frames]
        frame_inputs = self.processor(images=frame_images, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            frame_embeddings = self.model.get_image_features(**frame_inputs)
            frame_embeddings = frame_embeddings / frame_embeddings.norm(dim=-1, keepdim=True)
        
        # Compute similarities
        similarities = (frame_embeddings @ query_embedding.T).squeeze().cpu().numpy()
        
        # Group frames into moments using sliding window
        window_size = max(4, len(frames) // 8)
        moments = []
        
        for i in range(0, len(frames) - window_size + 1, max(1, window_size // 2)):
            window_similarities = similarities[i:i+window_size]
            avg_score = np.mean(window_similarities)
            max_score = np.max(window_similarities)
            
            # Use weighted score
            score = 0.7 * avg_score + 0.3 * max_score
            
            start_frame_idx = i
            end_frame_idx = min(i + window_size, len(frames) - 1)
            
            # Convert to actual frame numbers and times
            start_frame = int((start_frame_idx / len(frames)) * total_frames)
            end_frame = int((end_frame_idx / len(frames)) * total_frames)
            start_time = start_frame / fps if fps > 0 else (start_frame_idx / len(frames)) * duration
            end_time = end_frame / fps if fps > 0 else (end_frame_idx / len(frames)) * duration
            
            moments.append({
                "start_time": start_time,
                "end_time": end_time,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "score": float(score),
                "model_name": self.model_name
            })
        
        # Sort by score and return top_k
        moments.sort(key=lambda x: x["score"], reverse=True)
        return moments[:top_k]


class FAISSIndex(EpisodicMemoryModel):
    """FAISS-based retrieval using pre-computed embeddings."""
    
    def __init__(self, device: str = "cuda", embedding_model: str = "openai/clip-vit-base-patch32"):
        """
        Initialize FAISS index.
        
        Args:
            device: Device to run embedding model on ('cuda' or 'cpu').
            embedding_model: Model to use for generating embeddings (default: CLIP).
        """
        self.device = device
        self.model_name = "FAISS"
        self.embedding_model_name = embedding_model
        try:
            import faiss
            from transformers import CLIPProcessor, CLIPModel as HFCLIPModel
            
            self.faiss = faiss
            self.processor = CLIPProcessor.from_pretrained(embedding_model)
            self.embedding_model = HFCLIPModel.from_pretrained(embedding_model).to(device)
            self.embedding_model.eval()
            
            # Initialize empty index (will be built when videos are indexed)
            self.index = None
            self.video_metadata = []  # Store metadata for each indexed video segment
            self.dimension = 512  # CLIP embedding dimension
            self.available = True
        except ImportError as e:
            self.available = False
            self.error = f"Required libraries not available: {e}"
        except Exception as e:
            self.available = False
            self.error = str(e)
    
    def get_model_name(self) -> str:
        return self.model_name
    
    def _extract_frames(self, video_path: str, num_frames: int = 32) -> Tuple[List[np.ndarray], float, int]:
        """Extract frames from video."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return [], 30.0, 0
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        
        if total_frames == 0:
            cap.release()
            return [], fps, 0
        
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        frames = []
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
        
        cap.release()
        return frames, fps, total_frames
    
    def _encode_frames(self, frames: List[np.ndarray]) -> np.ndarray:
        """Encode frames to embeddings."""
        frame_images = [Image.fromarray(frame) for frame in frames]
        inputs = self.processor(images=frame_images, return_tensors="pt", padding=True).to(self.device)
        
        with torch.no_grad():
            embeddings = self.embedding_model.get_image_features(**inputs)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        
        return embeddings.cpu().numpy()
    
    def build_index(self, video_paths: List[str], num_frames_per_video: int = 32) -> None:
        """
        Build FAISS index from video collection.
        
        Args:
            video_paths: List of video file paths to index.
            num_frames_per_video: Number of frames to extract per video.
        """
        if not self.available:
            raise RuntimeError(f"FAISS model unavailable: {self.error}")
        
        print(f"Building FAISS index for {len(video_paths)} videos...")
        
        all_embeddings = []
        self.video_metadata = []
        
        for video_path in video_paths:
            if not Path(video_path).exists():
                continue
            
            frames, fps, total_frames = self._extract_frames(video_path, num_frames_per_video)
            if len(frames) == 0:
                continue
            
            # Encode frames
            embeddings = self._encode_frames(frames)
            
            # Store metadata for each frame segment
            for i, embedding in enumerate(embeddings):
                frame_idx = int((i / len(frames)) * total_frames) if len(frames) > 0 else i
                start_frame = frame_idx
                end_frame = min(frame_idx + (total_frames // len(frames)), total_frames)
                duration = total_frames / fps if fps > 0 else len(frames) / 30.0
                start_time = start_frame / fps if fps > 0 else (i / len(frames)) * duration
                end_time = end_frame / fps if fps > 0 else ((i + 1) / len(frames)) * duration
                
                self.video_metadata.append({
                    "video_path": video_path,
                    "start_time": start_time,
                    "end_time": end_time,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "frame_index": i
                })
                all_embeddings.append(embedding)
        
        if len(all_embeddings) == 0:
            print("Warning: No embeddings generated. Index not built.")
            return
        
        # Build FAISS index
        embeddings_array = np.array(all_embeddings).astype('float32')
        self.dimension = embeddings_array.shape[1]
        
        # Use L2 distance index
        self.index = self.faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings_array)
        
        print(f"FAISS index built with {len(all_embeddings)} embeddings")
    
    def retrieve_moments(self, video_path: str, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve moments using FAISS index."""
        if not self.available:
            raise RuntimeError(f"FAISS model unavailable: {self.error}")
        
        if self.index is None:
            # Build index on-the-fly for this video
            self.build_index([video_path])
        
        if self.index is None or len(self.video_metadata) == 0:
            return []
        
        # Encode query
        query_inputs = self.processor(text=[query], return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            query_embedding = self.embedding_model.get_text_features(**query_inputs)
            query_embedding = query_embedding / query_embedding.norm(dim=-1, keepdim=True)
        
        query_vector = query_embedding.cpu().numpy().astype('float32')
        
        # Search in FAISS index
        # Filter to only search in segments from the target video
        target_video_metadata = [m for m in self.video_metadata if m["video_path"] == video_path]
        if len(target_video_metadata) == 0:
            return []
        
        # Get indices of target video segments
        target_indices = [i for i, m in enumerate(self.video_metadata) if m["video_path"] == video_path]
        
        # Search in full index, then filter
        k = min(top_k * 2, len(target_indices))  # Get more results to filter
        distances, indices = self.index.search(query_vector, k)
        
        # Filter to target video and convert distances to similarities
        moments = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx in target_indices:
                metadata = self.video_metadata[idx]
                # Convert L2 distance to similarity (inverse distance, normalized)
                similarity = 1.0 / (1.0 + dist)
                
                moments.append({
                    "start_time": metadata["start_time"],
                    "end_time": metadata["end_time"],
                    "start_frame": metadata["start_frame"],
                    "end_frame": metadata["end_frame"],
                    "score": float(similarity),
                    "model_name": self.model_name
                })
        
        # Sort by score and return top_k
        moments.sort(key=lambda x: x["score"], reverse=True)
        return moments[:top_k]

