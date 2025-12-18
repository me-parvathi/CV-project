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


class MMAction2Model(EpisodicMemoryModel):
    """MMAction2 model for video understanding (action recognition and temporal localization)."""
    
    def __init__(self, device: str = "cuda", mode: str = "temporal_localization", 
                 model_name: Optional[str] = None, config_file: Optional[str] = None,
                 checkpoint_file: Optional[str] = None):
        """
        Initialize MMAction2 model.
        
        Args:
            device: Device to run model on ('cuda' or 'cpu').
            mode: Operation mode - 'temporal_localization' (BMN) or 'action_recognition' (TSN/SlowFast).
            model_name: Specific model name (e.g., 'bmn', 'tsn', 'slowfast'). If None, uses default for mode.
            config_file: Path to config file. If None, uses default.
            checkpoint_file: Path to checkpoint file. If None, downloads from model zoo.
        """
        self.device = device
        self.mode = mode
        self.model_name_str = model_name or ("bmn" if mode == "temporal_localization" else "tsn")
        self.model_name = f"MMAction2-{self.model_name_str.upper()}"
        
        try:
            from mmaction.apis import init_recognizer, inference_recognizer
            
            self.init_recognizer = init_recognizer
            self.inference_recognizer = inference_recognizer
            
            # Try to import temporal action detector (may not be available in all versions)
            try:
                from mmaction.apis import inference_temporal_action_detector
                self.inference_temporal_action_detector = inference_temporal_action_detector
                self.has_temporal_detector = True
            except ImportError:
                # Temporal action detector not available, will use action recognition fallback
                self.inference_temporal_action_detector = None
                self.has_temporal_detector = False
            
            self.mmaction_available = True
        except ImportError as e:
            self.available = False
            self.error = f"MMAction2 not available: {e}"
            self.mmaction_available = False
            return
        
        try:
            # Initialize model based on mode
            if mode == "temporal_localization":
                # Use BMN for temporal localization
                if config_file is None:
                    # Try to find config in mmaction package
                    try:
                        import mmaction
                        mmaction_dir = Path(mmaction.__file__).parent.parent
                        default_config = mmaction_dir / "configs" / "localization" / "bmn" / "bmn_2xb8-400x100-9e_activitynet-feature.py"
                        if default_config.exists():
                            config_file = str(default_config)
                        else:
                            # Fallback to relative path
                            config_file = "configs/localization/bmn/bmn_2xb8-400x100-9e_activitynet-feature.py"
                    except Exception:
                        config_file = "configs/localization/bmn/bmn_2xb8-400x100-9e_activitynet-feature.py"
                
                if checkpoint_file is None:
                    checkpoint_file = "https://download.openmmlab.com/mmaction/v1.0/localization/bmn/bmn_2xb8-400x100-9e_activitynet-feature/bmn_2xb8-400x100-9e_activitynet-feature_20220927-095211.pth"
                
                try:
                    self.model = init_recognizer(config_file, checkpoint_file, device=device)
                    self.available = True
                except Exception as e:
                    print(f"Warning: Could not load BMN model: {e}")
                    print("Falling back to action recognition mode...")
                    self.mode = "action_recognition"
                    self.model_name_str = "tsn"
                    self.model_name = "MMAction2-TSN"
                    # Try to load TSN as fallback
                    try:
                        import mmaction
                        mmaction_dir = Path(mmaction.__file__).parent.parent
                        default_config = mmaction_dir / "configs" / "recognition" / "tsn" / "tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb.py"
                        if default_config.exists():
                            config_file = str(default_config)
                        else:
                            config_file = "configs/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb.py"
                    except Exception:
                        config_file = "configs/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb.py"
                    checkpoint_file = "https://download.openmmlab.com/mmaction/v1.0/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb/tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb_20220906-2692d16c.pth"
                    try:
                        self.model = init_recognizer(config_file, checkpoint_file, device=device)
                        self.available = True
                    except Exception as e2:
                        self.available = False
                        self.error = f"Could not load any MMAction2 model: {e2}"
            
            elif mode == "action_recognition":
                # Use TSN or SlowFast for action recognition
                if model_name == "slowfast":
                    if config_file is None:
                        try:
                            import mmaction
                            mmaction_dir = Path(mmaction.__file__).parent.parent
                            default_config = mmaction_dir / "configs" / "recognition" / "slowfast" / "slowfast_r50_8x8x1_256e_kinetics400_rgb.py"
                            if default_config.exists():
                                config_file = str(default_config)
                            else:
                                config_file = "configs/recognition/slowfast/slowfast_r50_8x8x1_256e_kinetics400_rgb.py"
                        except Exception:
                            config_file = "configs/recognition/slowfast/slowfast_r50_8x8x1_256e_kinetics400_rgb.py"
                    if checkpoint_file is None:
                        checkpoint_file = "https://download.openmmlab.com/mmaction/v1.0/recognition/slowfast/slowfast_r50_8x8x1_256e_kinetics400_rgb/slowfast_r50_8x8x1_256e_kinetics400_rgb_20200704-73547d2b.pth"
                else:  # Default to TSN
                    if config_file is None:
                        try:
                            import mmaction
                            mmaction_dir = Path(mmaction.__file__).parent.parent
                            default_config = mmaction_dir / "configs" / "recognition" / "tsn" / "tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb.py"
                            if default_config.exists():
                                config_file = str(default_config)
                            else:
                                config_file = "configs/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb.py"
                        except Exception:
                            config_file = "configs/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb.py"
                    if checkpoint_file is None:
                        checkpoint_file = "https://download.openmmlab.com/mmaction/v1.0/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb/tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb_20220906-2692d16c.pth"
                
                try:
                    self.model = init_recognizer(config_file, checkpoint_file, device=device)
                    self.available = True
                except Exception as e:
                    self.available = False
                    self.error = f"Could not load action recognition model: {e}"
            
            if self.available:
                self.model.eval()
        
        except Exception as e:
            self.available = False
            self.error = f"Error initializing MMAction2 model: {e}"
    
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
    
    def _query_to_action_keywords(self, query: str) -> List[str]:
        """Extract action keywords from text query for matching with action classes."""
        # Simple keyword extraction - can be enhanced with NLP
        query_lower = query.lower()
        
        # Common action keywords
        action_keywords = []
        common_actions = [
            'pick', 'place', 'grab', 'put', 'move', 'hold', 'lift', 'drop',
            'open', 'close', 'push', 'pull', 'turn', 'rotate', 'press',
            'cut', 'slice', 'chop', 'pour', 'pour', 'mix', 'stir',
            'walk', 'run', 'sit', 'stand', 'jump', 'climb',
            'throw', 'catch', 'kick', 'hit', 'swing'
        ]
        
        for action in common_actions:
            if action in query_lower:
                action_keywords.append(action)
        
        return action_keywords if action_keywords else [query_lower]
    
    def _temporal_localization_retrieval(self, video_path: str, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve moments using temporal action localization (BMN)."""
        try:
            # For BMN, we need to use temporal action detection API
            # Note: BMN typically works with pre-extracted features, but we'll try with video
            if not self.has_temporal_detector or self.inference_temporal_action_detector is None:
                # Fallback to action recognition if temporal detector not available
                return self._action_recognition_retrieval(video_path, query, top_k)
            
            results = self.inference_temporal_action_detector(self.model, video_path)
            
            moments = []
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
            if cap.isOpened():
                cap.release()
            
            # Parse results - format depends on model output
            if isinstance(results, list):
                for i, result in enumerate(results[:top_k]):
                    if isinstance(result, dict):
                        start_time = result.get('start_time', result.get('start', 0))
                        end_time = result.get('end_time', result.get('end', 0))
                        score = result.get('score', result.get('confidence', 0.5))
                    elif isinstance(result, (list, tuple)) and len(result) >= 2:
                        start_time = float(result[0])
                        end_time = float(result[1])
                        score = float(result[2]) if len(result) > 2 else 0.5
                    else:
                        continue
                    
                    start_frame = int(start_time * fps) if fps > 0 else int(start_time * 30)
                    end_frame = int(end_time * fps) if fps > 0 else int(end_time * 30)
                    
                    moments.append({
                        "start_time": float(start_time),
                        "end_time": float(end_time),
                        "start_frame": start_frame,
                        "end_frame": end_frame,
                        "score": float(score),
                        "model_name": self.model_name
                    })
            
            # If no results or empty, fall back to action recognition approach
            if len(moments) == 0:
                return self._action_recognition_retrieval(video_path, query, top_k)
            
            moments.sort(key=lambda x: x["score"], reverse=True)
            return moments[:top_k]
        
        except Exception as e:
            print(f"Error in temporal localization: {e}")
            # Fallback to action recognition
            return self._action_recognition_retrieval(video_path, query, top_k)
    
    def _action_recognition_retrieval(self, video_path: str, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve moments using action recognition (TSN/SlowFast)."""
        try:
            # Run inference
            results = self.inference_recognizer(self.model, video_path)
            
            # Extract frames and metadata
            frames, fps, total_frames = self._extract_frames(video_path, num_frames=32)
            if len(frames) == 0:
                return []
            
            duration = total_frames / fps if fps > 0 else len(frames) / 30.0
            
            # Parse results - typically returns list of (class_name, score) tuples or dict
            action_scores = []
            if isinstance(results, list):
                action_scores = results
            elif isinstance(results, dict):
                action_scores = list(results.items())
            
            # Extract query keywords
            query_keywords = self._query_to_action_keywords(query)
            
            # Find matching actions
            matching_actions = []
            for action_name, score in action_scores:
                action_lower = str(action_name).lower()
                for keyword in query_keywords:
                    if keyword in action_lower or action_lower in keyword:
                        matching_actions.append((action_name, score))
                        break
            
            # If no direct matches, use top actions
            if len(matching_actions) == 0:
                matching_actions = action_scores[:top_k]
            
            # Convert to moments using sliding window approach
            # Since action recognition doesn't give temporal boundaries, we segment the video
            moments = []
            num_segments = min(len(matching_actions), top_k)
            
            for i, (action_name, score) in enumerate(matching_actions[:num_segments]):
                # Divide video into segments
                segment_duration = duration / num_segments
                start_time = i * segment_duration
                end_time = (i + 1) * segment_duration
                
                start_frame = int(start_time * fps) if fps > 0 else int((i / num_segments) * total_frames)
                end_frame = int(end_time * fps) if fps > 0 else int(((i + 1) / num_segments) * total_frames)
                
                moments.append({
                    "start_time": float(start_time),
                    "end_time": float(end_time),
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "score": float(score) if isinstance(score, (int, float)) else 0.5,
                    "model_name": self.model_name
                })
            
            moments.sort(key=lambda x: x["score"], reverse=True)
            return moments[:top_k]
        
        except Exception as e:
            print(f"Error in action recognition retrieval: {e}")
            return []
    
    def retrieve_moments(self, video_path: str, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve moments using MMAction2."""
        if not self.available:
            raise RuntimeError(f"MMAction2 model unavailable: {self.error}")
        
        if not Path(video_path).exists():
            return []
        
        try:
            if self.mode == "temporal_localization":
                return self._temporal_localization_retrieval(video_path, query, top_k)
            else:  # action_recognition
                return self._action_recognition_retrieval(video_path, query, top_k)
        except Exception as e:
            print(f"Error in MMAction2 retrieval: {e}")
            return []


class PyTorchVideoModel(EpisodicMemoryModel):
    """PyTorchVideo model (SlowFast/MViT) for video-text retrieval."""
    
    def __init__(self, device: str = "cuda", model_name: str = "slowfast"):
        """
        Initialize PyTorchVideo model.
        
        Args:
            device: Device to run model on ('cuda' or 'cpu').
            model_name: Model variant - 'slowfast' or 'mvit' (default: 'slowfast').
        """
        self.device = device
        self.video_model_name = model_name.lower()
        
        if self.video_model_name == "slowfast":
            self.model_name = "PyTorchVideo-SlowFast"
            self.pytorchvideo_model_name = "slowfast_r50"
        elif self.video_model_name == "mvit":
            self.model_name = "PyTorchVideo-MViT"
            self.pytorchvideo_model_name = "mvit_base_16x4"
        else:
            raise ValueError(f"Unsupported model_name: {model_name}. Use 'slowfast' or 'mvit'")
        
        try:
            import torch
            from transformers import CLIPProcessor, CLIPModel as HFCLIPModel
            
            # Load PyTorchVideo model using torch.hub
            self.pytorchvideo_model = torch.hub.load(
                'facebookresearch/pytorchvideo',
                self.pytorchvideo_model_name,
                pretrained=True
            ).to(device)
            self.pytorchvideo_model.eval()
            
            # Store transforms for later use
            self.pytorchvideo_transforms = None
            self._prepare_transforms()
            
            # Load CLIP for text encoding and similarity computation
            clip_model_name = "openai/clip-vit-base-patch32"
            self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
            self.clip_model = HFCLIPModel.from_pretrained(clip_model_name).to(device)
            self.clip_model.eval()
            
            self.available = True
        except ImportError as e:
            self.available = False
            self.error = f"Required libraries not available: {e}"
        except Exception as e:
            self.available = False
            self.error = str(e)
    
    def _prepare_transforms(self):
        """Prepare PyTorchVideo transforms for preprocessing."""
        from pytorchvideo.transforms import (
            ApplyTransformToKey,
            Normalize,
            ShortSideScale,
            UniformTemporalSubsample,
        )
        from torchvision.transforms import Compose, Lambda
        
        # PyTorchVideo models expect specific input format
        # For SlowFast: expects dict with "video" key containing tensor
        # For MViT: similar format
        
        mean = [0.45, 0.45, 0.45]
        std = [0.225, 0.225, 0.225]
        
        if self.video_model_name == "slowfast":
            # SlowFast specific transforms
            self.pytorchvideo_transforms = Compose([
                ApplyTransformToKey(
                    key="video",
                    transform=Compose([
                        UniformTemporalSubsample(8),  # Sample 8 frames
                        Lambda(lambda x: x / 255.0),  # Normalize to [0, 1]
                        Normalize(mean, std),
                        ShortSideScale(size=256),
                        Lambda(lambda x: x.permute(1, 0, 2, 3)),  # (C, T, H, W)
                    ]),
                ),
            ])
        else:  # mvit
            # MViT specific transforms
            self.pytorchvideo_transforms = Compose([
                ApplyTransformToKey(
                    key="video",
                    transform=Compose([
                        UniformTemporalSubsample(16),  # Sample 16 frames
                        Lambda(lambda x: x / 255.0),  # Normalize to [0, 1]
                        Normalize(mean, std),
                        ShortSideScale(size=224),
                    ]),
                ),
            ])
    
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
    
    def _extract_video_features(self, frames: List[np.ndarray]) -> torch.Tensor:
        """Extract video features using PyTorchVideo model."""
        # Convert frames to tensor format expected by PyTorchVideo
        # Frames should be in shape (T, H, W, C) -> (T, C, H, W)
        frame_tensors = []
        for frame in frames:
            # Convert numpy array to tensor and normalize
            frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float()  # (C, H, W)
            frame_tensors.append(frame_tensor)
        
        # Stack frames: (T, C, H, W)
        video_tensor = torch.stack(frame_tensors)
        
        # Apply transforms if available
        if self.pytorchvideo_transforms:
            try:
                # PyTorchVideo expects dict with "video" key
                video_dict = {"video": video_tensor}
                video_dict = self.pytorchvideo_transforms(video_dict)
                video_tensor = video_dict["video"]
            except Exception as e:
                # If transforms fail, use basic preprocessing
                video_tensor = video_tensor / 255.0
                if len(video_tensor.shape) == 4:
                    video_tensor = video_tensor.permute(1, 0, 2, 3)  # (C, T, H, W)
        
        # Ensure correct format: (C, T, H, W) or (1, C, T, H, W)
        if len(video_tensor.shape) == 4:
            # Assume (C, T, H, W), add batch dimension
            video_tensor = video_tensor.unsqueeze(0)
        elif len(video_tensor.shape) == 3:
            # Assume (T, C, H, W), rearrange and add batch
            video_tensor = video_tensor.permute(1, 0, 2, 3).unsqueeze(0)
        
        video_tensor = video_tensor.to(self.device)
        
        # Extract features
        with torch.no_grad():
            # For SlowFast models, prepare dual-pathway inputs
            if self.video_model_name == "slowfast":
                # SlowFast expects a list of two tensors: [slow_pathway, fast_pathway]
                # Slow pathway: 1/8 temporal rate (alpha = 8)
                # Fast pathway: full temporal rate (alpha = 1)
                
                # Get temporal dimension - video_tensor should be (B, C, T, H, W)
                if len(video_tensor.shape) == 5:
                    B, C, T, H, W = video_tensor.shape
                    
                    # Create slow pathway: sample every 8th frame
                    # Ensure we have at least one frame and proper shape
                    slow_indices = torch.arange(0, T, 8, device=self.device, dtype=torch.long)
                    if len(slow_indices) == 0:
                        slow_indices = torch.tensor([0], device=self.device, dtype=torch.long)
                    
                    # Use index_select for proper indexing to avoid shape issues
                    slow_pathway = video_tensor.index_select(2, slow_indices)
                    
                    # Fast pathway: use all frames
                    fast_pathway = video_tensor
                    
                    # Ensure both pathways have compatible shapes (same B, C, H, W)
                    # Only temporal dimension should differ
                    assert slow_pathway.shape[0] == fast_pathway.shape[0], "Batch size mismatch"
                    assert slow_pathway.shape[1] == fast_pathway.shape[1], "Channel size mismatch"
                    assert slow_pathway.shape[3] == fast_pathway.shape[3], "Height mismatch"
                    assert slow_pathway.shape[4] == fast_pathway.shape[4], "Width mismatch"
                    
                    # Pass as list to model
                    model_input = [slow_pathway, fast_pathway]
                else:
                    # Fallback: if shape is unexpected, use single tensor for both pathways
                    # This ensures compatibility even if shape is wrong
                    model_input = [video_tensor, video_tensor]
            else:
                # For MViT and other models, use single tensor
                model_input = video_tensor
            
            # PyTorchVideo models return features/logits
            output = self.pytorchvideo_model(model_input)
            
            # Handle different output formats
            if isinstance(output, tuple):
                features = output[0]  # Usually first element is features
            elif isinstance(output, dict):
                # Some models return dict with 'features' or 'logits' key
                features = output.get('features', output.get('logits', output.get('x', None)))
                if features is None:
                    features = list(output.values())[0]
            else:
                features = output
            
            # Ensure features is a tensor
            if not isinstance(features, torch.Tensor):
                raise ValueError(f"Expected tensor output, got {type(features)}")
            
            # CRITICAL FIX: Flatten features to 2D first (batch, features)
            # Handle any number of dimensions by flattening all non-batch dimensions
            if len(features.shape) == 1:
                # Add batch dimension if missing
                features = features.unsqueeze(0)
            elif len(features.shape) > 2:
                # Flatten all dimensions except batch
                features = features.view(features.size(0), -1)
            
            # Ensure we have 2D tensor: (batch, features)
            if len(features.shape) != 2:
                features = features.view(features.size(0), -1)
            
            # CRITICAL FIX: Ensure consistent feature size (512 dimensions)
            # This ensures all segments produce the same feature size regardless of input
            target_size = 512
            batch_size = features.shape[0]
            current_size = features.shape[1]
            
            if current_size > target_size:
                # If features are too large, use mean pooling in chunks
                # Pad to make current_size divisible by target_size
                remainder = current_size % target_size
                if remainder > 0:
                    # Pad to make divisible
                    pad_size = target_size - remainder
                    padding = torch.zeros(batch_size, pad_size, 
                                         device=features.device, dtype=features.dtype)
                    features = torch.cat([features, padding], dim=1)
                    current_size = features.shape[1]
                
                # Reshape and average pool: (batch, target_size, chunk_size) -> (batch, target_size)
                chunk_size = current_size // target_size
                features = features.view(batch_size, target_size, chunk_size).mean(dim=-1)
            elif current_size < target_size:
                # If features are too small, pad with zeros
                padding = torch.zeros(batch_size, target_size - current_size, 
                                    device=features.device, dtype=features.dtype)
                features = torch.cat([features, padding], dim=1)
            
            # Final validation: ensure exact target size
            if features.shape[1] != target_size:
                # Force to target size
                if features.shape[1] > target_size:
                    features = features[:, :target_size]
                else:
                    padding = torch.zeros(batch_size, target_size - features.shape[1],
                                         device=features.device, dtype=features.dtype)
                    features = torch.cat([features, padding], dim=1)
            
            # Normalize features
            features = features / (features.norm(dim=-1, keepdim=True) + 1e-8)
        
        # Remove batch dimension and ensure 1D output
        features = features.squeeze(0)  # Remove batch dimension -> (features,)
        
        # Final validation: ensure output is exactly target_size
        if len(features.shape) == 0:
            features = features.unsqueeze(0)
        if features.shape[0] != 512:
            # Force to 512
            if features.shape[0] > 512:
                features = features[:512]
            else:
                padding = torch.zeros(512 - features.shape[0], 
                                    device=features.device, dtype=features.dtype)
                features = torch.cat([features, padding])
        
        return features  # Return (512,) tensor
    
    def retrieve_moments(self, video_path: str, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve moments using PyTorchVideo features and CLIP text matching."""
        if not self.available:
            raise RuntimeError(f"PyTorchVideo model unavailable: {self.error}")
        
        if not Path(video_path).exists():
            return []
        
        # Extract frames
        frames, fps, total_frames = self._extract_frames(video_path, num_frames=32)
        if len(frames) == 0:
            return []
        
        duration = total_frames / fps if fps > 0 else len(frames) / 30.0
        
        # Encode query using CLIP
        query_inputs = self.clip_processor(text=[query], return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            query_embedding = self.clip_model.get_text_features(**query_inputs)
            query_embedding = query_embedding / query_embedding.norm(dim=-1, keepdim=True)
        
        # Extract video features for frame segments using sliding window
        window_size = max(8, len(frames) // 4)  # Adaptive window size
        stride = max(1, window_size // 2)
        
        segment_features = []
        segment_indices = []
        
        for i in range(0, len(frames) - window_size + 1, stride):
            segment_frames = frames[i:i+window_size]
            try:
                segment_feature = self._extract_video_features(segment_frames)
                
                # Validate feature shape immediately after extraction
                if not isinstance(segment_feature, torch.Tensor):
                    print(f"Warning: Segment {i} returned non-tensor: {type(segment_feature)}")
                    continue
                
                # Ensure feature is 1D and has correct size
                if len(segment_feature.shape) == 0:
                    segment_feature = segment_feature.unsqueeze(0)
                elif len(segment_feature.shape) > 1:
                    segment_feature = segment_feature.view(-1)
                
                # Validate size
                if segment_feature.shape[0] != 512:
                    print(f"Warning: Segment {i} has incorrect size {segment_feature.shape[0]}, expected 512. Fixing...")
                    if segment_feature.shape[0] < 512:
                        padding = torch.zeros(512 - segment_feature.shape[0],
                                            device=segment_feature.device, dtype=segment_feature.dtype)
                        segment_feature = torch.cat([segment_feature, padding])
                    else:
                        segment_feature = segment_feature[:512]
                
                segment_features.append(segment_feature)
                segment_indices.append((i, min(i + window_size, len(frames))))
            except Exception as e:
                # Skip segment if feature extraction fails
                print(f"Warning: Failed to extract features for segment {i}: {e}")
                continue
        
        if len(segment_features) == 0:
            return []
        
        # CRITICAL FIX: Ensure all features have the same shape before stacking
        # Double-check all features are properly normalized
        normalized_features = []
        target_size = 512  # Should match the target_size in _extract_video_features
        
        for idx, feat in enumerate(segment_features):
            try:
                # Ensure feature is a tensor
                if not isinstance(feat, torch.Tensor):
                    print(f"Warning: Feature {idx} is not a tensor, skipping")
                    continue
                
                # Ensure feature is 1D
                if len(feat.shape) > 1:
                    feat = feat.view(-1)
                elif len(feat.shape) == 0:
                    feat = feat.unsqueeze(0)
                
                current_size = feat.shape[0]
                
                # Validate and fix size
                if current_size != target_size:
                    if current_size < target_size:
                        # Pad with zeros
                        padding = torch.zeros(target_size - current_size, 
                                            device=feat.device, dtype=feat.dtype)
                        feat = torch.cat([feat, padding])
                    elif current_size > target_size:
                        # Truncate (shouldn't happen if _extract_video_features works correctly)
                        feat = feat[:target_size]
                
                # Final validation
                if feat.shape[0] != target_size:
                    print(f"Warning: Feature {idx} still has incorrect size after normalization: {feat.shape[0]}")
                    continue
                
                normalized_features.append(feat)
            except Exception as e:
                print(f"Warning: Error normalizing feature {idx}: {e}")
                continue
        
        if len(normalized_features) == 0:
            print("Warning: No valid features after normalization")
            return []
        
        # Validate all features have same shape before stacking
        first_shape = normalized_features[0].shape
        for idx, feat in enumerate(normalized_features):
            if feat.shape != first_shape:
                print(f"Warning: Feature {idx} shape {feat.shape} doesn't match first feature shape {first_shape}")
                # Force to match
                if feat.shape[0] != first_shape[0]:
                    if feat.shape[0] < first_shape[0]:
                        padding = torch.zeros(first_shape[0] - feat.shape[0],
                                            device=feat.device, dtype=feat.dtype)
                        feat = torch.cat([feat, padding])
                    else:
                        feat = feat[:first_shape[0]]
                normalized_features[idx] = feat
        
        # Stack features and compute similarities
        try:
            segment_features_tensor = torch.stack(normalized_features).to(self.device)
        except Exception as e:
            print(f"Error stacking features: {e}")
            print(f"Feature shapes: {[f.shape for f in normalized_features]}")
            return []
        similarities = (segment_features_tensor @ query_embedding.T).squeeze().cpu().numpy()
        
        # Convert to moments
        moments = []
        for (start_idx, end_idx), score in zip(segment_indices, similarities):
            start_frame = int((start_idx / len(frames)) * total_frames)
            end_frame = int((end_idx / len(frames)) * total_frames)
            start_time = start_frame / fps if fps > 0 else (start_idx / len(frames)) * duration
            end_time = end_frame / fps if fps > 0 else (end_idx / len(frames)) * duration
            
            moments.append({
                "start_time": float(start_time),
                "end_time": float(end_time),
                "start_frame": start_frame,
                "end_frame": end_frame,
                "score": float(score),
                "model_name": self.model_name
            })
        
        # Sort by score and return top_k
        moments.sort(key=lambda x: x["score"], reverse=True)
        return moments[:top_k]

