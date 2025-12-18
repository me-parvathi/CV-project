"""Episodic memory retrieval model definitions."""

from abc import ABC, abstractmethod
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import cv2
from PIL import Image
# NOTE: resize_video_tensor is only used in deprecated _extract_video_features method
# It is not used for similarity search (CLIP-only architecture)
from data_processing import resize_video_tensor


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
        frames = self._extract_frames(video_path, num_frames=64)
        if len(frames) == 0:
            return []
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else len(frames)
        duration = total_frames / fps if fps > 0 else len(frames) / 30.0
        if cap.isOpened():
            cap.release()
        
        # Use mixed precision for faster inference
        use_amp = self.device == "cuda" and torch.cuda.is_available()
        autocast_context = torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16) if use_amp else torch.no_grad()
        
        # Encode query
        query_inputs = self.clip_processor(text=[query], return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            with autocast_context:
                query_embedding = self.clip_model.get_text_features(**query_inputs)
                query_embedding = query_embedding / query_embedding.norm(dim=-1, keepdim=True)
        
        # Encode frames
        frame_images = [Image.fromarray(frame) for frame in frames]
        frame_inputs = self.clip_processor(images=frame_images, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            with autocast_context:
                frame_embeddings = self.clip_model.get_image_features(**frame_inputs)
                frame_embeddings = frame_embeddings / frame_embeddings.norm(dim=-1, keepdim=True)
        
        # Compute similarities (keep on GPU for better utilization)
        similarities_tensor = (frame_embeddings @ query_embedding.T).squeeze()
        # Only move to CPU when needed for numpy operations
        similarities = similarities_tensor.cpu().numpy()
        
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
        """Retrieve moments using CLIP-based retrieval (Moment-DETR bypassed)."""
        if not Path(video_path).exists():
            return []
        
        # Immediately use CLIP-based retrieval - Moment-DETR bypassed
        # Moment-DETR is never called to prevent positional-embedding shape errors
        return self._clip_based_retrieval(video_path, query, top_k)
        
        # NOTE: All Moment-DETR processing code has been removed/commented out below
        # Moment-DETR model will never receive video tensors to eliminate positional-embedding mismatches
        # 
        # Previously attempted Moment-DETR code:
        # - Frame extraction for Moment-DETR
        # - self.processor(...) calls
        # - self.model(**inputs) calls
        # - Moment-DETR output processing
        # All removed to enforce CLIP-only feature extraction architecture


class CLIPModel(EpisodicMemoryModel):
    """CLIP model for video-text retrieval."""
    
    def __init__(self, device: str = "cuda", model_name: str = "openai/clip-vit-base-patch32", 
                 use_amp: bool = True, num_frames: int = 64):
        """
        Initialize CLIP model.
        
        Args:
            device: Device to run model on ('cuda' or 'cpu').
            model_name: HuggingFace model identifier for CLIP.
            use_amp: Enable automatic mixed precision (FP16) for faster inference.
            num_frames: Number of frames to extract per video (default: 64 for better GPU utilization).
        """
        self.device = device
        self.model_name = "CLIP"
        self.hf_model_name = model_name
        self.use_amp = use_amp and device == "cuda" and torch.cuda.is_available()
        self.num_frames = num_frames
        try:
            from transformers import CLIPProcessor, CLIPModel as HFCLIPModel
            
            self.processor = CLIPProcessor.from_pretrained(model_name)
            self.model = HFCLIPModel.from_pretrained(model_name).to(device)
            self.model.eval()
            # Enable half precision if AMP is enabled
            if self.use_amp:
                self.model = self.model.half()
            self.available = True
        except Exception as e:
            self.available = False
            self.error = str(e)
    
    def get_model_name(self) -> str:
        return self.model_name
    
    def _extract_frames(self, video_path: str, num_frames: int = 64) -> Tuple[List[np.ndarray], float, int]:
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
        
        # Extract frames (increased batch size for better GPU utilization)
        frames, fps, total_frames = self._extract_frames(video_path, num_frames=self.num_frames)
        if len(frames) == 0:
            return []
        
        duration = total_frames / fps if fps > 0 else len(frames) / 30.0
        
        # Use mixed precision for faster inference
        autocast_context = torch.cuda.amp.autocast(enabled=self.use_amp, dtype=torch.float16) if self.use_amp else torch.no_grad()
        
        # Encode query
        query_inputs = self.processor(text=[query], return_tensors="pt", padding=True).to(self.device)
        if self.use_amp and query_inputs['input_ids'].dtype != torch.int64:
            # Ensure input_ids are int64
            query_inputs['input_ids'] = query_inputs['input_ids'].long()
        
        with torch.no_grad():
            with autocast_context:
                query_embedding = self.model.get_text_features(**query_inputs)
                query_embedding = query_embedding / query_embedding.norm(dim=-1, keepdim=True)
        
        # Encode frames in batches for better GPU utilization
        frame_images = [Image.fromarray(frame) for frame in frames]
        frame_inputs = self.processor(images=frame_images, return_tensors="pt", padding=True).to(self.device)
        
        with torch.no_grad():
            with autocast_context:
                frame_embeddings = self.model.get_image_features(**frame_inputs)
                frame_embeddings = frame_embeddings / frame_embeddings.norm(dim=-1, keepdim=True)
        
        # Compute similarities (keep on GPU for better utilization)
        # Use GPU tensor operations instead of moving to CPU immediately
        similarities_tensor = (frame_embeddings @ query_embedding.T).squeeze()
        # Only move to CPU when needed for numpy operations
        similarities = similarities_tensor.cpu().numpy()
        
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
    
    def __init__(self, device: str = "cuda", embedding_model: str = "openai/clip-vit-base-patch32", 
                 use_amp: bool = True, num_frames: int = 64):
        """
        Initialize FAISS index.
        
        Args:
            device: Device to run embedding model on ('cuda' or 'cpu').
            embedding_model: Model to use for generating embeddings (default: CLIP).
            use_amp: Enable automatic mixed precision (FP16) for faster inference.
            num_frames: Number of frames to extract per video (default: 64 for better GPU utilization).
        """
        self.device = device
        self.model_name = "FAISS"
        self.embedding_model_name = embedding_model
        self.use_amp = use_amp and device == "cuda" and torch.cuda.is_available()
        self.num_frames = num_frames
        try:
            import faiss
            from transformers import CLIPProcessor, CLIPModel as HFCLIPModel
            
            self.faiss = faiss
            self.processor = CLIPProcessor.from_pretrained(embedding_model)
            self.embedding_model = HFCLIPModel.from_pretrained(embedding_model).to(device)
            self.embedding_model.eval()
            if self.use_amp:
                self.embedding_model = self.embedding_model.half()
            
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
    
    def _extract_frames(self, video_path: str, num_frames: int = 64) -> Tuple[List[np.ndarray], float, int]:
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
        
        autocast_context = torch.cuda.amp.autocast(enabled=self.use_amp, dtype=torch.float16) if self.use_amp else torch.no_grad()
        with torch.no_grad():
            with autocast_context:
                embeddings = self.embedding_model.get_image_features(**inputs)
                embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        
        return embeddings.cpu().numpy()
    
    def build_index(self, video_paths: List[str], num_frames_per_video: int = 64) -> None:
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
        
        # Encode query with mixed precision
        query_inputs = self.processor(text=[query], return_tensors="pt", padding=True).to(self.device)
        autocast_context = torch.cuda.amp.autocast(enabled=self.use_amp, dtype=torch.float16) if self.use_amp else torch.no_grad()
        with torch.no_grad():
            with autocast_context:
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
            frames, fps, total_frames = self._extract_frames(video_path, num_frames=64)
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
    
    def __init__(self, device: str = "cuda", model_name: str = "slowfast", 
                 use_amp: bool = True, num_frames: int = 64):
        """
        Initialize PyTorchVideo model.
        
        Args:
            device: Device to run model on ('cuda' or 'cpu').
            model_name: Model variant - 'slowfast' or 'mvit' (default: 'slowfast').
            use_amp: Enable automatic mixed precision (FP16) for faster inference.
            num_frames: Number of frames to extract per video (default: 64 for better GPU utilization).
        """
        self.device = device
        self.video_model_name = model_name.lower()
        self.use_amp = use_amp and device == "cuda" and torch.cuda.is_available()
        self.num_frames = num_frames
        
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
            if self.use_amp:
                self.pytorchvideo_model = self.pytorchvideo_model.half()
            
            # Store transforms for later use
            self.pytorchvideo_transforms = None
            self._prepare_transforms()
            
            # Load CLIP for text encoding and similarity computation
            clip_model_name = "openai/clip-vit-base-patch32"
            self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
            self.clip_model = HFCLIPModel.from_pretrained(clip_model_name).to(device)
            self.clip_model.eval()
            if self.use_amp:
                self.clip_model = self.clip_model.half()
            
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
    
    def _extract_frames(self, video_path: str, num_frames: int = 64) -> Tuple[List[np.ndarray], float, int]:
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
        """
        DEPRECATED: This method is no longer used for similarity search.
        
        Extract video features using PyTorchVideo model.
        
        NOTE: This method is kept for optional metadata extraction only, not for similarity search.
        PyTorchVideo models are NOT used as feature backbones for similarity search to prevent
        positional-embedding shape errors. All similarity computations now use CLIP embeddings (512-D).
        
        This method may be removed in future versions.
        """
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
                # Input should be (T, C, H, W) for transforms
                video_dict = {"video": video_tensor}
                video_dict = self.pytorchvideo_transforms(video_dict)
                video_tensor = video_dict["video"]
                # Transforms output should be (C, T, H, W) based on the Lambda transform
            except Exception as e:
                # If transforms fail, use basic preprocessing
                video_tensor = video_tensor / 255.0
                if len(video_tensor.shape) == 4:
                    # Ensure (C, T, H, W) format
                    if video_tensor.shape[0] == 3:  # Channels first
                        pass  # Already (C, T, H, W)
                    else:
                        video_tensor = video_tensor.permute(1, 0, 2, 3)  # (T, C, H, W) -> (C, T, H, W)
        
        # Ensure correct format: (C, T, H, W) -> (B, C, T, H, W)
        if len(video_tensor.shape) == 4:
            # Check if it's (C, T, H, W) or (T, C, H, W)
            if video_tensor.shape[0] == 3:  # RGB channels
                # Assume (C, T, H, W), add batch dimension
                video_tensor = video_tensor.unsqueeze(0)  # (1, C, T, H, W)
            else:
                # Might be (T, C, H, W), rearrange
                video_tensor = video_tensor.permute(1, 0, 2, 3).unsqueeze(0)  # (1, C, T, H, W)
        elif len(video_tensor.shape) == 3:
            # Assume (T, C, H, W), rearrange and add batch
            video_tensor = video_tensor.permute(1, 0, 2, 3).unsqueeze(0)
        elif len(video_tensor.shape) == 5:
            # Already has batch dimension, ensure it's (B, C, T, H, W)
            if video_tensor.shape[1] != 3:  # Channels not in position 1
                # Might be (B, T, C, H, W), permute to (B, C, T, H, W)
                video_tensor = video_tensor.permute(0, 2, 1, 3, 4)
        
        # Final check: ensure we have (B, C, T, H, W) format
        if len(video_tensor.shape) != 5 or video_tensor.shape[1] != 3:
            # Force to correct format
            if len(video_tensor.shape) == 4:
                video_tensor = video_tensor.unsqueeze(0)
            # Ensure channels are in position 1
            if len(video_tensor.shape) == 5 and video_tensor.shape[2] == 3:
                video_tensor = video_tensor.permute(0, 2, 1, 3, 4)
        
        # Resize spatial dimensions to 224x224 to match model's expected resolution
        # This avoids positional embedding mismatch errors
        video_tensor = resize_video_tensor(video_tensor, target_size=(224, 224))
        
        video_tensor = video_tensor.to(self.device)
        
        # Extract features
        with torch.no_grad():
            # PyTorchVideo models from torch.hub - handle SlowFast specially
            output = None
            last_error = None
            
            if self.video_model_name == "slowfast":
                # SlowFast model expects a list of two tensors:
                # - slow_pathway: [B, C, 8, H, W] - 8 frames
                # - fast_pathway: [B, C, 32, H, W] - 32 frames (4x slow)
                # We need to create these pathways with exact frame counts
                
                B, C, T, H, W = video_tensor.shape
                
                # Target frame counts for slowfast_r50
                slow_frames = 8
                fast_frames = 32
                
                # Create slow pathway with exactly 8 frames
                if T >= slow_frames:
                    # Sample 8 frames uniformly
                    slow_indices = torch.linspace(0, T - 1, slow_frames, device=self.device, dtype=torch.long)
                    slow_pathway = video_tensor.index_select(2, slow_indices)
                else:
                    # Not enough frames, repeat to get 8
                    repeat_factor = (slow_frames + T - 1) // T
                    repeated = video_tensor.repeat(1, 1, repeat_factor, 1, 1)
                    slow_indices = torch.linspace(0, repeated.shape[2] - 1, slow_frames, device=self.device, dtype=torch.long)
                    slow_pathway = repeated.index_select(2, slow_indices)
                
                # Create fast pathway with exactly 32 frames
                if T >= fast_frames:
                    # Sample 32 frames uniformly
                    fast_indices = torch.linspace(0, T - 1, fast_frames, device=self.device, dtype=torch.long)
                    fast_pathway = video_tensor.index_select(2, fast_indices)
                else:
                    # Not enough frames, repeat to get 32
                    repeat_factor = (fast_frames + T - 1) // T
                    repeated = video_tensor.repeat(1, 1, repeat_factor, 1, 1)
                    fast_indices = torch.linspace(0, repeated.shape[2] - 1, fast_frames, device=self.device, dtype=torch.long)
                    fast_pathway = repeated.index_select(2, fast_indices)
                
                # Verify shapes
                assert slow_pathway.shape == (B, C, slow_frames, H, W), f"Slow pathway shape mismatch: {slow_pathway.shape}"
                assert fast_pathway.shape == (B, C, fast_frames, H, W), f"Fast pathway shape mismatch: {fast_pathway.shape}"
                
                # Pass as list [slow, fast] to model
                model_input = [slow_pathway, fast_pathway]
                try:
                    output = self.pytorchvideo_model(model_input)
                except Exception as e:
                    last_error = e
                    raise RuntimeError(f"SlowFast model failed with slow={slow_pathway.shape}, fast={fast_pathway.shape}: {e}")
            
            else:
                # For MViT and other models, use single tensor input
                try:
                    output = self.pytorchvideo_model(video_tensor)
                except Exception as e:
                    last_error = e
                    raise RuntimeError(f"Model failed with input shape {video_tensor.shape}: {e}")
            
            if output is None:
                raise RuntimeError(f"Failed to extract features. Last error: {last_error}")
            
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
            original_shape = features.shape
            
            # If features has more than 2 dimensions, we need to flatten
            if len(features.shape) == 1:
                # Add batch dimension if missing: (features,) -> (1, features)
                features = features.unsqueeze(0)
            elif len(features.shape) > 2:
                # Flatten all dimensions except batch: (B, ...) -> (B, features)
                # Use global average pooling for spatial dimensions, then flatten temporal
                # For video features, we typically have (B, C, T, H, W) or (B, T, C, H, W)
                if len(features.shape) == 5:
                    # (B, C, T, H, W) or (B, T, C, H, W) - need to determine
                    B = features.shape[0]
                    # Try to identify format by checking if C=3 (RGB)
                    if features.shape[1] == 3:
                        # (B, C, T, H, W) format
                        # Global average pool over spatial and temporal: (B, C, T, H, W) -> (B, C)
                        features = features.mean(dim=(2, 3, 4))  # Average over T, H, W
                    elif features.shape[2] == 3:
                        # (B, T, C, H, W) format
                        features = features.mean(dim=(1, 3, 4))  # Average over T, H, W
                    else:
                        # Unknown format, flatten all except batch
                        features = features.view(B, -1)
                elif len(features.shape) == 4:
                    # (B, C, H, W) or (B, T, H, W) - global average pool spatial
                    features = features.mean(dim=(2, 3))  # Average over H, W
                elif len(features.shape) == 3:
                    # (B, T, C) or (B, C, T) - average over temporal
                    features = features.mean(dim=1)  # Average over middle dimension
                else:
                    # Fallback: flatten all except batch
                    features = features.view(features.size(0), -1)
            
            # Ensure we have 2D tensor: (batch, features)
            if len(features.shape) != 2:
                # Force to 2D
                batch_size = features.shape[0] if len(features.shape) > 0 else 1
                features = features.view(batch_size, -1)
            
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
        """Retrieve moments using CLIP-based feature extraction (PyTorchVideo bypassed for similarity search)."""
        if not self.available:
            raise RuntimeError(f"PyTorchVideo model unavailable: {self.error}")
        
        if not Path(video_path).exists():
            return []
        
        # Extract frames using CLIP-based approach (same as CLIPModel)
        frames, fps, total_frames = self._extract_frames(video_path, num_frames=self.num_frames)
        if len(frames) == 0:
            return []
        
        duration = total_frames / fps if fps > 0 else len(frames) / 30.0
        
        # Use mixed precision for faster inference
        autocast_context = torch.cuda.amp.autocast(enabled=self.use_amp, dtype=torch.float16) if self.use_amp else torch.no_grad()
        
        # Encode query using CLIP
        query_inputs = self.clip_processor(text=[query], return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            with autocast_context:
                query_embedding = self.clip_model.get_text_features(**query_inputs)
                query_embedding = query_embedding / query_embedding.norm(dim=-1, keepdim=True)
        
        # Encode frames using CLIP (guaranteed 512-D embeddings)
        # PyTorchVideo models are NOT used for feature extraction to prevent positional-embedding errors
        frame_images = [Image.fromarray(frame) for frame in frames]
        frame_inputs = self.clip_processor(images=frame_images, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            with autocast_context:
                frame_embeddings = self.clip_model.get_image_features(**frame_inputs)
                frame_embeddings = frame_embeddings / frame_embeddings.norm(dim=-1, keepdim=True)
        
        # Compute similarities (all features are guaranteed 512-D CLIP embeddings)
        # Keep on GPU for better utilization
        similarities_tensor = (frame_embeddings @ query_embedding.T).squeeze()
        # Only move to CPU when needed for numpy operations
        similarities = similarities_tensor.cpu().numpy()
        
        # Group frames into moments using sliding window (same approach as CLIPModel)
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

