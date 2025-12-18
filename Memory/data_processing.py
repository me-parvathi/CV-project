"""Dataset loaders for MECCANO episodic memory benchmarking."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import re
import pandas as pd
import cv2
import torch
import torch.nn.functional as F


def resize_video_tensor(video: torch.Tensor, target_size: tuple = (224, 224)) -> torch.Tensor:
    """
    Resize video tensor spatial dimensions to target size.
    
    This function resizes the spatial dimensions (H, W) of a video tensor
    to the model's expected resolution (typically 224x224) to avoid
    positional embedding mismatch errors.
    
    Args:
        video: Video tensor of shape [B, C, T, H, W] or [C, T, H, W]
        target_size: Target spatial size as (height, width). Default: (224, 224)
        
    Returns:
        Resized video tensor with same shape except H and W are resized to target_size
    """
    if not isinstance(video, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(video)}")
    
    original_shape = video.shape
    original_ndim = len(original_shape)
    
    # Handle different input shapes
    if original_ndim == 4:
        # Shape: [C, T, H, W] - add batch dimension temporarily
        video = video.unsqueeze(0)
        added_batch = True
    elif original_ndim == 5:
        # Shape: [B, C, T, H, W] - already has batch dimension
        added_batch = False
    else:
        raise ValueError(f"Expected 4D [C, T, H, W] or 5D [B, C, T, H, W] tensor, got shape {original_shape}")
    
    B, C, T, H, W = video.shape
    
    # Reshape to [B*T, C, H, W] for batch processing
    video_reshaped = video.permute(0, 2, 1, 3, 4).contiguous()  # [B, T, C, H, W]
    video_reshaped = video_reshaped.view(B * T, C, H, W)  # [B*T, C, H, W]
    
    # Resize spatial dimensions
    video_resized = F.interpolate(
        video_reshaped,
        size=target_size,
        mode='bilinear',
        align_corners=False
    )  # [B*T, C, target_H, target_W]
    
    # Reshape back to [B, T, C, target_H, target_W]
    video_resized = video_resized.view(B, T, C, target_size[0], target_size[1])
    
    # Permute back to [B, C, T, target_H, target_W]
    video_resized = video_resized.permute(0, 2, 1, 3, 4).contiguous()
    
    # Remove batch dimension if it was added
    if added_batch:
        video_resized = video_resized.squeeze(0)  # [C, T, target_H, target_W]
    
    return video_resized


class DatasetLoader(ABC):
    """Abstract base class for dataset loaders."""
    
    @abstractmethod
    def load_queries(self) -> List[Dict[str, Any]]:
        """
        Load queries and associated video data from dataset.
        
        Returns:
            List of dicts with 'video_path', 'query', 'ground_truth' keys.
        """
        pass


class MECCANOLoader(DatasetLoader):
    """Loader for MECCANO dataset (videos + annotation files)."""
    
    def __init__(self, dataset_path: str, annotation_file: Optional[str] = None,
                 split: Optional[str] = None, load_all_splits: bool = False):
        """
        Initialize MECCANO loader.
        
        Args:
            dataset_path: Path to MECCANO dataset root directory.
            annotation_file: Path to specific annotation file (CSV or JSON). 
                            If None, will search for MECCANO annotation files.
            split: Dataset split to load ('train', 'test', 'val'). If None, loads all splits.
            load_all_splits: If True, loads train/test/val splits automatically.
        """
        self.dataset_path = Path(dataset_path)
        self.annotation_file = Path(annotation_file) if annotation_file else None
        self.split = split
        self.load_all_splits = load_all_splits
        
        # MECCANO-specific annotation file patterns
        self.meccano_annotation_patterns = {
            'actions': {
                'train': '**/MECCANO_train_actions.csv',
                'test': '**/MECCANO_test_actions.csv',
                'val': '**/MECCANO_val_actions.csv'
            },
            'verbal': {
                'train': '**/MECCANO_verb_annotations_train.csv',
                'test': '**/MECCANO_verb_annotations_test.csv',
                'val': '**/MECCANO_verb_annotations_val.csv'
            },
            'bbox': {
                'train': '**/instances_meccano_train.json',
                'test': '**/instances_meccano_test.json',
                'val': '**/instances_meccano_val.json'
            }
        }
        
        # Common annotation file names to search for (fallback)
        self.common_annotation_names = [
            "annotations.json",
            "annotations.csv",
            "queries.json",
            "queries.csv",
            "ground_truth.json",
            "ground_truth.csv"
        ]
    
    def _find_video_directory(self) -> Optional[Path]:
        """Find the video directory in MECCANO structure."""
        # Common MECCANO video directory patterns
        video_patterns = [
            "MECCANO_RGB_Videos/MECCANO_RGB_Videos",
            "MECCANO_RGB_Videos",
            "videos",
            "Videos"
        ]
        
        # Check in dataset path
        for pattern in video_patterns:
            candidate = self.dataset_path / pattern
            if candidate.exists() and candidate.is_dir():
                return candidate
        
        # Check parent directory
        for pattern in video_patterns:
            candidate = self.dataset_path.parent / pattern
            if candidate.exists() and candidate.is_dir():
                return candidate
        
        # Fallback: search for any directory with .mp4 files
        for video_file in self.dataset_path.rglob("*.mp4"):
            return video_file.parent
        
        return None
    
    def _find_annotation_files(self) -> Dict[str, List[Path]]:
        """Find MECCANO annotation files for all splits."""
        annotation_files = {
            'actions': [],
            'verbal': [],
            'bbox': []
        }
        
        # If specific annotation file provided, use it
        if self.annotation_file and self.annotation_file.exists():
            if self.annotation_file.suffix.lower() == '.csv':
                if 'verb' in self.annotation_file.name.lower():
                    annotation_files['verbal'].append(self.annotation_file)
                else:
                    annotation_files['actions'].append(self.annotation_file)
            elif self.annotation_file.suffix.lower() == '.json':
                annotation_files['bbox'].append(self.annotation_file)
            return annotation_files
        
        # Search for MECCANO annotation files
        splits_to_load = ['train', 'test', 'val'] if self.load_all_splits else ([self.split] if self.split else ['train', 'test', 'val'])
        
        for split in splits_to_load:
            # Find action annotations
            found = list(self.dataset_path.parent.glob(self.meccano_annotation_patterns['actions'][split]))
            if found:
                annotation_files['actions'].extend(found)
            
            # Find verbal annotations
            found = list(self.dataset_path.parent.glob(self.meccano_annotation_patterns['verbal'][split]))
            if found:
                annotation_files['verbal'].extend(found)
            
            # Find bounding box annotations
            found = list(self.dataset_path.parent.glob(self.meccano_annotation_patterns['bbox'][split]))
            if found:
                annotation_files['bbox'].extend(found)
        
        # Fallback: search for common annotation names
        if not any(annotation_files.values()):
            for name in self.common_annotation_names:
                candidate = self.dataset_path / name
                if candidate.exists():
                    if candidate.suffix.lower() == '.csv':
                        annotation_files['actions'].append(candidate)
                    else:
                        annotation_files['bbox'].append(candidate)
                    break
        
        return annotation_files
    
    def _find_video_file(self, video_id: str) -> Optional[Path]:
        """Find video file by ID (e.g., '0001' or '0001.mp4')."""
        video_dir = self._find_video_directory()
        if not video_dir:
            return None
        
        # Normalize video ID (remove extension if present, ensure 4-digit format)
        video_id_clean = video_id.replace('.mp4', '').strip()
        # Try to pad with zeros if needed
        if video_id_clean.isdigit():
            video_id_clean = video_id_clean.zfill(4)
        
        # Try different patterns
        patterns = [
            f"{video_id_clean}.mp4",
            f"{video_id}.mp4",
            f"{int(video_id_clean)}.mp4" if video_id_clean.isdigit() else None
        ]
        
        for pattern in patterns:
            if pattern is None:
                continue
            candidate = video_dir / pattern
            if candidate.exists():
                return candidate
        
        # Fallback: search recursively
        for video_file in video_dir.rglob("*.mp4"):
            if video_id_clean in video_file.stem or video_id in video_file.stem:
                return video_file
        
        return None
    
    def _load_annotations_json(self, annotation_path: Path) -> List[Dict]:
        """Load annotations from JSON file."""
        with open(annotation_path, 'r') as f:
            data = json.load(f)
        
        # Handle different JSON structures
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            # Try common keys
            for key in ['annotations', 'queries', 'data', 'items']:
                if key in data:
                    return data[key] if isinstance(data[key], list) else [data[key]]
            # If no list found, return as single item
            return [data]
        else:
            return []
    
    def _load_annotations_csv(self, annotation_path: Path) -> List[Dict]:
        """Load annotations from CSV file."""
        try:
            df = pd.read_csv(annotation_path)
            return df.to_dict('records')
        except Exception as e:
            print(f"Warning: Error loading CSV {annotation_path}: {e}")
            return []
    
    def _parse_meccano_video_id(self, annotation: Dict) -> Optional[str]:
        """Extract video ID from MECCANO annotation."""
        # Try common MECCANO column names
        for key in ['video_id', 'video_name', 'video', 'video_file', 'id', 'name']:
            if key in annotation:
                video_id = str(annotation[key])
                # Remove path if present, keep just filename
                video_id = Path(video_id).stem if video_id else None
                if video_id:
                    return video_id
        return None
    
    def _merge_annotations(self, actions: List[Dict], verbal: List[Dict], 
                          bbox_data: Dict) -> List[Dict]:
        """Merge annotations from different sources."""
        merged = []
        
        # Create lookup dictionaries
        verbal_lookup = {}
        for v in verbal:
            vid = self._parse_meccano_video_id(v)
            if vid:
                if vid not in verbal_lookup:
                    verbal_lookup[vid] = []
                verbal_lookup[vid].append(v)
        
        bbox_lookup = {}
        if bbox_data and 'annotations' in bbox_data:
            for ann in bbox_data['annotations']:
                # COCO format typically has image_id
                img_id = ann.get('image_id') or ann.get('video_id')
                if img_id:
                    img_id = str(img_id)
                    if img_id not in bbox_lookup:
                        bbox_lookup[img_id] = []
                    bbox_lookup[img_id].append(ann)
        
        # Merge action annotations with verbal and bbox
        for action in actions:
            video_id = self._parse_meccano_video_id(action)
            if not video_id:
                continue
            
            merged_item = action.copy()
            
            # Add verbal annotation if available
            if video_id in verbal_lookup:
                verbal_ann = verbal_lookup[video_id][0]  # Take first match
                # Merge text query fields
                for key in ['query', 'text', 'description', 'sentence', 'verb_description']:
                    if key in verbal_ann and key not in merged_item:
                        merged_item[key] = verbal_ann[key]
            
            # Add bbox annotation if available
            if video_id in bbox_lookup:
                bbox_anns = bbox_lookup[video_id]
                # Store bbox annotations
                merged_item['bbox_annotations'] = bbox_anns
            
            merged.append(merged_item)
        
        # Add verbal-only entries (if any)
        for video_id, verbal_list in verbal_lookup.items():
            if video_id not in [self._parse_meccano_video_id(a) for a in actions]:
                for verbal_ann in verbal_list:
                    merged.append(verbal_ann)
        
        return merged
    
    def _parse_temporal_interval(self, annotation: Dict) -> Optional[Dict]:
        """
        Parse temporal interval from annotation.
        
        Supports multiple formats:
        - start_frame, end_frame
        - start_time, end_time (in seconds)
        - start, end (could be frames or times)
        - temporal_segment (list of [start, end])
        """
        temporal = {}
        
        # Helper function to extract frame number from string (handles filenames like '00010.jpg')
        def _extract_frame_number(value):
            """Extract numeric frame number from string, handling filenames."""
            if isinstance(value, (int, float)):
                return int(value)
            value_str = str(value).strip()
            # If it's a filename, extract the numeric part before the extension
            if '.' in value_str:
                # Remove file extension and extract numeric part
                stem = Path(value_str).stem
                # Extract digits from the stem
                match = re.search(r'\d+', stem)
                if match:
                    return int(match.group())
            # Try direct conversion
            try:
                return int(value_str)
            except ValueError:
                # If it's a float string, convert to int
                try:
                    return int(float(value_str))
                except ValueError:
                    raise ValueError(f"Cannot convert '{value}' to frame number")
        
        # Try frame-based
        if 'start_frame' in annotation and 'end_frame' in annotation:
            temporal['start_frame'] = _extract_frame_number(annotation['start_frame'])
            temporal['end_frame'] = _extract_frame_number(annotation['end_frame'])
            temporal['type'] = 'frames'
        # Try time-based
        elif 'start_time' in annotation and 'end_time' in annotation:
            temporal['start_time'] = float(annotation['start_time'])
            temporal['end_time'] = float(annotation['end_time'])
            temporal['type'] = 'time'
        # Try generic start/end
        elif 'start' in annotation and 'end' in annotation:
            start = annotation['start']
            end = annotation['end']
            # Determine if frames or times
            if isinstance(start, (int, float)) and isinstance(end, (int, float)):
                if start < 10000:  # Heuristic: likely frames if small numbers
                    temporal['start_frame'] = int(start)
                    temporal['end_frame'] = int(end)
                    temporal['type'] = 'frames'
                else:
                    temporal['start_time'] = float(start)
                    temporal['end_time'] = float(end)
                    temporal['type'] = 'time'
            else:
                # Try to extract frame numbers from strings
                try:
                    start_frame = _extract_frame_number(start)
                    end_frame = _extract_frame_number(end)
                    temporal['start_frame'] = start_frame
                    temporal['end_frame'] = end_frame
                    temporal['type'] = 'frames'
                except (ValueError, TypeError):
                    # If extraction fails, try as time
                    try:
                        temporal['start_time'] = float(start)
                        temporal['end_time'] = float(end)
                        temporal['type'] = 'time'
                    except (ValueError, TypeError):
                        pass
        # Try temporal_segment
        elif 'temporal_segment' in annotation:
            segment = annotation['temporal_segment']
            if isinstance(segment, list) and len(segment) == 2:
                temporal['start_time'] = float(segment[0])
                temporal['end_time'] = float(segment[1])
                temporal['type'] = 'time'
        
        return temporal if temporal else None
    
    def _parse_spatial_bbox(self, annotation: Dict) -> Optional[Dict]:
        """
        Parse spatial bounding box from annotation.
        
        Supports formats:
        - bbox, bounding_box, box (list of [x, y, w, h] or [x1, y1, x2, y2])
        - x, y, width, height
        - x1, y1, x2, y2
        """
        bbox = None
        
        # Try common bbox keys
        for key in ['bbox', 'bounding_box', 'box', 'spatial_bbox']:
            if key in annotation:
                bbox = annotation[key]
                break
        
        if bbox is None:
            # Try individual coordinates
            if all(k in annotation for k in ['x', 'y', 'width', 'height']):
                bbox = [
                    annotation['x'],
                    annotation['y'],
                    annotation['width'],
                    annotation['height']
                ]
            elif all(k in annotation for k in ['x1', 'y1', 'x2', 'y2']):
                bbox = [
                    annotation['x1'],
                    annotation['y1'],
                    annotation['x2'],
                    annotation['y2']
                ]
        
        if bbox and isinstance(bbox, list) and len(bbox) >= 4:
            return {
                'bbox': [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
                'format': 'xywh' if len(bbox) == 4 and bbox[2] < 1000 and bbox[3] < 1000 else 'xyxy'
            }
        
        return None
    
    def _get_video_fps(self, video_path: Path) -> Optional[float]:
        """Get FPS of video file."""
        try:
            cap = cv2.VideoCapture(str(video_path))
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS)
                cap.release()
                return fps if fps > 0 else None
        except Exception:
            pass
        return None
    
    def _convert_time_to_frames(self, start_time: float, end_time: float, fps: float) -> tuple:
        """Convert time-based intervals to frame-based."""
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        return start_frame, end_frame
    
    def _normalize_temporal(self, temporal: Dict, video_path: Path) -> Dict:
        """Normalize temporal annotation to include both frames and times if possible."""
        if temporal is None:
            return {}
        
        normalized = temporal.copy()
        
        # If we have frames, try to get times
        if 'start_frame' in temporal and 'end_frame' in temporal:
            fps = self._get_video_fps(video_path)
            if fps:
                normalized['start_time'] = temporal['start_frame'] / fps
                normalized['end_time'] = temporal['end_frame'] / fps
        
        # If we have times, try to get frames
        elif 'start_time' in temporal and 'end_time' in temporal:
            fps = self._get_video_fps(video_path)
            if fps:
                start_frame, end_frame = self._convert_time_to_frames(
                    temporal['start_time'], temporal['end_time'], fps
                )
                normalized['start_frame'] = start_frame
                normalized['end_frame'] = end_frame
        
        return normalized
    
    def load_queries(self) -> List[Dict[str, Any]]:
        """
        Load queries and associated video data from MECCANO dataset.
        
        Supports MECCANO dataset structure with:
        - Multiple annotation files (train/test/val splits)
        - Action annotations (CSV)
        - Verbal annotations (CSV) 
        - Bounding box annotations (JSON)
        - Nested video directories
        
        Returns:
            List of dicts with:
            - video_path: str
            - query: str
            - ground_truth: dict with 'temporal' and optionally 'spatial' keys
        """
        queries = []
        
        if not self.dataset_path.exists():
            print(f"Warning: Dataset path does not exist: {self.dataset_path}")
            return queries
        
        # Find annotation files
        annotation_files = self._find_annotation_files()
        
        # Load action annotations
        actions = []
        for action_file in annotation_files['actions']:
            print(f"Loading action annotations from: {action_file}")
            actions.extend(self._load_annotations_csv(action_file))
        
        # Load verbal annotations
        verbal = []
        for verbal_file in annotation_files['verbal']:
            print(f"Loading verbal annotations from: {verbal_file}")
            verbal.extend(self._load_annotations_csv(verbal_file))
        
        # Load bounding box annotations
        bbox_data = {}
        for bbox_file in annotation_files['bbox']:
            print(f"Loading bounding box annotations from: {bbox_file}")
            bbox_loaded = self._load_annotations_json(bbox_file)
            # Handle COCO format JSON (dict with 'annotations' key) or list format
            if isinstance(bbox_loaded, dict):
                bbox_data.update(bbox_loaded)
            elif isinstance(bbox_loaded, list) and len(bbox_loaded) > 0:
                # If list, check if first item is dict with 'annotations' key
                if isinstance(bbox_loaded[0], dict) and 'annotations' in bbox_loaded[0]:
                    bbox_data.update(bbox_loaded[0])
                else:
                    # Store as annotations list
                    bbox_data['annotations'] = bbox_loaded
        
        # Merge annotations
        if actions or verbal:
            annotations = self._merge_annotations(actions, verbal, bbox_data)
        elif bbox_data:
            # If only bbox data, convert to list format
            if isinstance(bbox_data, dict) and 'annotations' in bbox_data:
                annotations = bbox_data['annotations']
            elif isinstance(bbox_data, dict):
                annotations = [bbox_data]
            else:
                annotations = [bbox_data] if bbox_data else []
        else:
            # Fallback: search for videos without annotations
            print(f"Warning: No annotation files found. Searching for videos without annotations...")
            video_dir = self._find_video_directory()
            if video_dir:
                for video_file in video_dir.glob("*.mp4"):
                    queries.append({
                        "video_path": str(video_file),
                        "query": "",
                        "ground_truth": {}
                    })
            else:
                for video_file in self.dataset_path.rglob("*.mp4"):
                    queries.append({
                        "video_path": str(video_file),
                        "query": "",
                        "ground_truth": {}
                    })
            return queries
        
        # Process each annotation
        for annotation in annotations:
            # Extract video ID and find video file
            video_id = self._parse_meccano_video_id(annotation)
            if not video_id:
                # Try direct video path
                for key in ['video_path', 'video_file', 'video']:
                    if key in annotation:
                        video_path_str = str(annotation[key])
                        video_path = Path(video_path_str)
                        if video_path.exists():
                            break
                        # Try relative paths
                        video_path = self.dataset_path / video_path_str
                        if not video_path.exists():
                            found = list(self.dataset_path.rglob(Path(video_path_str).name))
                            if found:
                                video_path = found[0]
                        if video_path.exists():
                            break
                else:
                    continue
            else:
                video_path = self._find_video_file(video_id)
                if not video_path:
                    continue
            
            # Extract query text
            query = ""
            for key in ['query', 'text_query', 'query_text', 'text', 'sentence', 
                       'description', 'verb_description', 'action_description']:
                if key in annotation:
                    query = str(annotation[key])
                    break
            
            # Extract ground truth
            ground_truth = {}
            
            # Parse temporal information
            temporal = self._parse_temporal_interval(annotation)
            if temporal:
                ground_truth['temporal'] = self._normalize_temporal(temporal, video_path)
            
            # Parse spatial information from bbox_annotations if present
            if 'bbox_annotations' in annotation:
                bbox_anns = annotation['bbox_annotations']
                if bbox_anns and len(bbox_anns) > 0:
                    # Take first bbox annotation
                    bbox_ann = bbox_anns[0]
                    # COCO format: bbox is [x, y, width, height]
                    if 'bbox' in bbox_ann:
                        spatial = {
                            'bbox': bbox_ann['bbox'],
                            'format': 'xywh'  # COCO format
                        }
                        ground_truth['spatial'] = spatial
            else:
                # Try parsing spatial from main annotation
                spatial = self._parse_spatial_bbox(annotation)
                if spatial:
                    ground_truth['spatial'] = spatial
            
            queries.append({
                "video_path": str(video_path),
                "query": query,
                "ground_truth": ground_truth,
                "annotation": annotation  # Keep raw annotation for reference
            })
        
        print(f"Loaded {len(queries)} queries from dataset")
        return queries

