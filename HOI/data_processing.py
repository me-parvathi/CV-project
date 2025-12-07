"""Dataset loaders for HOI benchmarking."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any


class DatasetLoader(ABC):
    """Abstract base class for dataset loaders."""
    
    @abstractmethod
    def load_videos(self) -> List[Dict[str, Any]]:
        """
        Load video paths from dataset.
        
        Returns:
            List of dicts with at least 'video_path' key.
        """
        pass


class UCF101Loader(DatasetLoader):
    """Loader for UCF101 dataset (directory-based structure)."""
    
    def __init__(self, dataset_path: str):
        """
        Initialize UCF101 loader.
        
        Args:
            dataset_path: Path to UCF101 dataset root directory.
        """
        self.dataset_path = Path(dataset_path)
    
    def load_videos(self) -> List[Dict[str, Any]]:
        """
        Load videos from UCF101 directory structure.
        
        Expected structure: dataset_path/action_class/video.mp4
        """
        videos = []
        
        if not self.dataset_path.exists():
            return videos
        
        for action_dir in self.dataset_path.iterdir():
            if action_dir.is_dir():
                for video_file in action_dir.glob("*.mp4"):
                    if video_file.exists() and video_file.is_file():
                        videos.append({
                            "video_path": str(video_file),
                            "action_class": action_dir.name,
                            "video_name": video_file.name
                        })
        
        return videos


class EPICKitchensLoader(DatasetLoader):
    """Loader for EPIC-KITCHENS dataset (annotation file-based)."""
    
    def __init__(self, dataset_path: str, annotation_file: str):
        """
        Initialize EPIC-KITCHENS loader.
        
        Args:
            dataset_path: Path to EPIC-KITCHENS dataset root directory.
            annotation_file: Path to annotation CSV/JSON file.
        """
        self.dataset_path = Path(dataset_path)
        self.annotation_file = Path(annotation_file)
    
    def load_videos(self) -> List[Dict[str, Any]]:
        """
        Load videos from EPIC-KITCHENS annotation file.
        
        Placeholder for future implementation.
        """
        # TODO: Implement EPIC-KITCHENS loading logic
        return []

