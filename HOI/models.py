"""HOI model definitions."""

from abc import ABC, abstractmethod
import torch
import cv2
import numpy as np
from PIL import Image
from typing import Dict
from datetime import datetime


class HOIModel(ABC):
    """Abstract base class for HOI models."""
    
    @abstractmethod
    def process_video(self, video_path: str) -> Dict:
        """
        Process a video and return results.
        
        Args:
            video_path: Path to video file.
            
        Returns:
            Dict with inference results.
        """
        pass


class YOLOVideoMAEModel(HOIModel):
    """YOLOv8 + VideoMAE pipeline."""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        try:
            from ultralytics import YOLO
            from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
            
            # Initialize YOLO - device will be specified during inference
            # YOLO from ultralytics doesn't use .to(), but accepts device parameter in predict()
            self.yolo_model = YOLO('yolov8m.pt')
            self.videomae_processor = VideoMAEImageProcessor.from_pretrained(
                "MCG-NJU/videomae-base-finetuned-kinetics"
            )
            self.videomae_model = VideoMAEForVideoClassification.from_pretrained(
                "MCG-NJU/videomae-base-finetuned-kinetics"
            ).to(device)
            self.available = True
        except Exception as e:
            self.available = False
            self.error = str(e)
    
    def process_video(self, video_path: str, sample_frames: int = 16) -> Dict:
        """Process video with YOLOv8 for objects and VideoMAE for actions."""
        if not self.available:
            raise RuntimeError(f"Model unavailable: {self.error}")
        
        from pathlib import Path
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        results = {
            "video_path": video_path,
            "timestamp": datetime.now().isoformat(),
            "objects_detected": [],
            "actions_detected": []
        }
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        frame_indices = np.linspace(0, total_frames - 1, sample_frames, dtype=int)
        frames = []
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
        
        cap.release()
        
        if len(frames) == 0:
            return results
        
        # YOLO Object Detection
        middle_frame = frames[len(frames) // 2]
        # Explicitly specify device for YOLO inference
        # YOLO accepts device as string ("cuda" or "cpu") or int (0 for GPU 0)
        yolo_device = self.device if torch.cuda.is_available() and self.device == "cuda" else "cpu"
        yolo_results = self.yolo_model(middle_frame, device=yolo_device, verbose=False)
        
        objects = []
        for result in yolo_results:
            for box in result.boxes:
                obj = {
                    "class": result.names[int(box.cls)],
                    "confidence": float(box.conf),
                    "bbox": box.xyxy[0].cpu().numpy().tolist()
                }
                objects.append(obj)
        
        results["objects_detected"] = objects
        
        # VideoMAE Action Recognition
        video_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
        video_frames = [Image.fromarray(f) for f in video_frames]
        
        inputs = self.videomae_processor(video_frames, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.videomae_model(**inputs)
            logits = outputs.logits
            predicted_class = logits.argmax(-1).item()
        
        action = self.videomae_model.config.id2label[predicted_class]
        confidence = torch.softmax(logits, dim=-1).max().item()
        
        results["actions_detected"] = [{
            "action": action,
            "confidence": confidence
        }]
        
        results["structured_output"] = {
            "primary_action": action,
            "action_confidence": confidence,
            "objects_in_scene": [obj["class"] for obj in objects]
        }
        
        return results


class LLaVAModel(HOIModel):
    """LLaVA-1.5 pipeline."""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        try:
            from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
            
            model_id = "llava-hf/llava-1.5-7b-hf"
            self.processor = LlavaNextProcessor.from_pretrained(model_id)
            self.model = LlavaNextForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            ).to(device)
            self.available = True
        except Exception as e:
            self.available = False
            self.error = str(e)
    
    def process_video(self, video_path: str, sample_frames: int = 8) -> Dict:
        """Process video with LLaVA."""
        if not self.available:
            raise RuntimeError(f"Model unavailable: {self.error}")
        
        from pathlib import Path
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        results = {
            "video_path": video_path,
            "timestamp": datetime.now().isoformat(),
            "llava_analysis": {}
        }
        
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        frame_indices = np.linspace(0, total_frames - 1, sample_frames, dtype=int)
        frames = []
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame_rgb))
        
        cap.release()
        
        if len(frames) == 0:
            return results
        
        prompts = [
            "What objects can you see in this image? List them.",
            "What action is being performed in this image?",
            "Describe what is happening in this scene in detail.",
            "Is there any safety concern in this image? What objects need attention?"
        ]
        
        analyses = {}
        middle_frame = frames[len(frames) // 2]
        
        for i, prompt in enumerate(prompts):
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            prompt_text = self.processor.apply_chat_template(
                conversation, add_generation_prompt=True
            )
            
            inputs = self.processor(
                images=middle_frame,
                text=prompt_text,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=False
                )
            
            response = self.processor.decode(
                output[0], skip_special_tokens=True
            )
            
            if "ASSISTANT:" in response:
                response = response.split("ASSISTANT:")[-1].strip()
            
            analyses[f"prompt_{i+1}"] = {
                "question": prompt,
                "answer": response
            }
        
        results["llava_analysis"] = analyses
        results["structured_output"] = {
            "objects_mentioned": analyses.get("prompt_1", {}).get("answer", ""),
            "action_description": analyses.get("prompt_2", {}).get("answer", ""),
            "scene_description": analyses.get("prompt_3", {}).get("answer", ""),
            "safety_assessment": analyses.get("prompt_4", {}).get("answer", "")
        }
        
        return results


class Qwen2VLModel(HOIModel):
    """Qwen2-VL pipeline."""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        try:
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
            
            model_id = "Qwen/Qwen2-VL-2B-Instruct"
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.float16
            ).to(device)
            self.processor = AutoProcessor.from_pretrained(model_id)
            self.available = True
        except Exception as e:
            self.available = False
            self.error = str(e)
    
    def process_video(self, video_path: str, sample_frames: int = 8) -> Dict:
        """Process video with Qwen2-VL."""
        if not self.available:
            raise RuntimeError(f"Model unavailable: {self.error}")
        
        from pathlib import Path
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        results = {
            "video_path": video_path,
            "timestamp": datetime.now().isoformat(),
            "qwen_analysis": {}
        }
        
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        frame_indices = np.linspace(0, total_frames - 1, sample_frames, dtype=int)
        frames = []
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame_rgb))
        
        cap.release()
        
        if len(frames) == 0:
            return results
        
        prompts = [
            "List all objects visible in this image.",
            "What hand-object interaction is happening? Describe the action.",
            "Describe the scene and what activity is being performed.",
            "From a safety perspective for an elderly person, what should we monitor here?"
        ]
        
        analyses = {}
        middle_frame = frames[len(frames) // 2]
        
        for i, prompt in enumerate(prompts):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": middle_frame},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            inputs = self.processor(
                text=[text],
                images=[middle_frame],
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=128
                )
            
            response = self.processor.batch_decode(
                output_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]
            
            if prompt in response:
                response = response.split(prompt)[-1].strip()
            
            analyses[f"prompt_{i+1}"] = {
                "question": prompt,
                "answer": response
            }
        
        results["qwen_analysis"] = analyses
        results["structured_output"] = {
            "objects_detected": analyses.get("prompt_1", {}).get("answer", ""),
            "action_detected": analyses.get("prompt_2", {}).get("answer", ""),
            "scene_context": analyses.get("prompt_3", {}).get("answer", ""),
            "safety_monitoring": analyses.get("prompt_4", {}).get("answer", "")
        }
        
        return results

