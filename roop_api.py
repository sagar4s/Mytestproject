# roop_api.py - Complete API for roop-unleashed
import os
import cv2
import numpy as np
import torch
import threading
from typing import List, Optional, Union, Dict, Any, Callable, Tuple
from datetime import datetime
from pathlib import Path
import tempfile
import shutil
from PIL import Image

# Import from your actual roop module structure
from roop.core import live_swap, batch_process_regular, batch_process_with_options
from roop.ProcessOptions import ProcessOptions
from roop.ProcessMgr import ProcessMgr
from roop.face_util import extract_face_images
from roop.ProcessEntry import ProcessEntry
from roop.FaceSet import FaceSet
from roop.StreamWriter import StreamWriter
from roop.virtualcam import start_virtual_cam, stop_virtual_cam, cam_active
from roop.vr_util import get_perspective
from roop.utilities import (
    replace_template, detect_fps, get_temp_directory_path, 
    get_temp_frame_paths, create_temp, clean_temp, delete_temp_frames,
    has_image_extension, is_image, is_video, get_destfilename_from_path,
    open_folder, compute_cosine_distance, clean_dir, shuffle_array,
    sort_filenames_ignore_path, sort_rename_frames
)
from roop.util_ffmpeg import (
    extract_frames, create_video, restore_audio, cut_video, 
    join_videos, create_gif_from_video, repair_video
)
import roop.globals

# Import from your actual clip module structure
try:
    from clip.clip import load, available_models
    from clip.simple_tokenizer import SimpleTokenizer
    CLIP_AVAILABLE = True
except ImportError as e:
    CLIP_AVAILABLE = False
    print(f"CLIP modules not available: {e}")

try:
    from clip.clipseg import CLIPDensePredT
    CLIPSEG_AVAILABLE = True
except ImportError as e:
    CLIPSEG_AVAILABLE = False
    print(f"CLIPSeg modules not available: {e}")

class RoopUnleashedAPI:
    def __init__(self):
        self.initialized = False
        self.process_mgr = None
        self.stream_writer = None
        self.processing_callback = None
        self.virtual_cam_thread = None
        self.virtual_cam_active = False
        self.clip_model = None
        self.clip_preprocess = None
        self.clipseg_model = None
        self.clip_tokenizer = None
        
    def initialize(self, 
                   execution_providers: List[str] = None,
                   max_threads: int = 2, 
                   max_memory: int = 16,
                   output_path: str = "./output",
                   video_encoder: str = "libx264",
                   video_quality: int = 14,
                   log_level: str = "error",
                   output_image_format: str = "png",
                   output_video_format: str = "mp4",
                   face_distance_threshold: float = 0.65,
                   blend_ratio: float = 0.5,
                   subsample_size: int = 128,
                   enable_vr_mode: bool = False,
                   load_clip_models: bool = True,
                   load_clipseg_model: bool = True,
                   clip_model_name: str = "ViT-B/32") -> bool:
        
        if execution_providers is None:
            execution_providers = self._auto_detect_providers()
            
        roop.globals.execution_providers = execution_providers
        roop.globals.execution_threads = max_threads
        roop.globals.max_memory = max_memory
        roop.globals.video_encoder = video_encoder
        roop.globals.video_quality = video_quality
        roop.globals.output_path = output_path
        roop.globals.log_level = log_level
        roop.globals.distance_threshold = face_distance_threshold
        roop.globals.blend_ratio = blend_ratio
        roop.globals.subsample_size = subsample_size
        roop.globals.vr_mode = enable_vr_mode
        
        # Set output formats
        if not hasattr(roop.globals, 'CFG'):
            class Config:
                pass
            roop.globals.CFG = Config()
        roop.globals.CFG.output_image_format = output_image_format
        roop.globals.CFG.output_video_format = output_video_format
        
        roop.globals.INPUT_FACESETS = []
        roop.globals.TARGET_FACES = []
        
        # Initialize face analysis
        roop.globals.g_desired_face_analysis = [
            "landmark_3d_68", "landmark_2d_106", "detection", "recognition"
        ]
        
        # Create output directory
        os.makedirs(output_path, exist_ok=True)
        
        # Load CLIP models if requested
        if load_clip_models and CLIP_AVAILABLE:
            self._load_clip_models(clip_model_name)
        
        # Load CLIPSeg model if requested
        if load_clipseg_model and CLIPSEG_AVAILABLE:
            self._load_clipseg_model()
        
        self.initialized = True
        print(f"✓ Roop Unleashed initialized with {execution_providers}")
        return True
    
    def _auto_detect_providers(self) -> List[str]:
        try:
            import onnxruntime
            available = onnxruntime.get_available_providers()
            if 'CUDAExecutionProvider' in available:
                return ['CUDAExecutionProvider']
            elif 'DmlExecutionProvider' in available:
                return ['DmlExecutionProvider']
            elif 'ROCMExecutionProvider' in available:
                return ['ROCMExecutionProvider']
            else:
                return ['CPUExecutionProvider']
        except:
            return ['CPUExecutionProvider']
    
    def _load_clip_models(self, model_name: str = "ViT-B/32"):
        try:
            print(f"Loading CLIP model: {model_name}")
            self.clip_model, self.clip_preprocess = load(model_name, device="cpu")
            self.clip_tokenizer = SimpleTokenizer()
            print(f"✓ CLIP model {model_name} loaded successfully")
        except Exception as e:
            print(f"Failed to load CLIP models: {e}")
            self.clip_model = None
            self.clip_preprocess = None
            self.clip_tokenizer = None
    
    def _load_clipseg_model(self):
        try:
            print("Loading CLIPSeg model...")
            self.clipseg_model = CLIPDensePredT(version='ViT-B/32', reduce_dim=64, complex_trans_conv=True)
            self.clipseg_model.eval()
            print("✓ CLIPSeg model loaded successfully")
        except Exception as e:
            print(f"Failed to load CLIPSeg model: {e}")
            self.clipseg_model = None

    def load_source_faces(self, 
                         source_image_paths: Union[str, List[str]], 
                         face_index: int = 0,
                         extra_padding: float = -1.0) -> int:
        if isinstance(source_image_paths, str):
            source_image_paths = [source_image_paths]
            
        roop.globals.INPUT_FACESETS = []
        loaded_count = 0
        
        for source_path in source_image_paths:
            if not os.path.exists(source_path):
                print(f"Source image not found: {source_path}")
                continue
                
            faces_data = extract_face_images(source_path, (False, 0), extra_padding)
            if faces_data and len(faces_data) > face_index:
                face_set = FaceSet()
                face_set.faces = [faces_data[face_index][0]]
                face_set.ref_images = [faces_data[face_index][1]]
                roop.globals.INPUT_FACESETS.append(face_set)
                loaded_count += 1
                print(f"✓ Loaded source face from {os.path.basename(source_path)}")
            else:
                print(f"No face found in {os.path.basename(source_path)}")
        
        return loaded_count

    def swap_single_image(self,
                         source_image_path: str,
                         target_image_path: str,
                         output_path: str = None,
                         swap_model: str = "inswapper_128_fp16",
                         enhancer: str = None,
                         swap_mode: str = "all",
                         face_restorer: str = None,
                         face_restorer_visibility: float = 1.0,
                         blending_ratio: float = 0.65,
                         **kwargs) -> bool:
        
        if not self.initialized:
            print("API not initialized. Call initialize() first.")
            return False
        
        if not os.path.exists(source_image_path):
            print(f"Source image not found: {source_image_path}")
            return False
            
        if not os.path.exists(target_image_path):
            print(f"Target image not found: {target_image_path}")
            return False
        
        # Load source faces
        source_faces_loaded = self.load_source_faces([source_image_path])
        if source_faces_loaded == 0:
            print("No source faces found")
            return False
        
        # Generate output path if not provided
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            target_filename = Path(target_image_path).stem
            output_path = f"result_{target_filename}_{timestamp}.png"
            output_path = os.path.join(roop.globals.output_path, output_path)
        
        # Create ProcessOptions
        options = ProcessOptions(
            execution_providers=roop.globals.execution_providers,
            execution_threads=roop.globals.execution_threads,
            max_memory=roop.globals.max_memory,
            face_swap_model=swap_model,
            face_enhancer=enhancer,
            face_restorer=face_restorer,
            face_restorer_visibility=face_restorer_visibility,
            blending_ratio=blending_ratio,
            swap_mode=swap_mode,
            vr_mode=roop.globals.vr_mode
        )
        
        try:
            print(f"Processing image: {os.path.basename(target_image_path)}")
            result = batch_process_regular(
                target_path=target_image_path,
                output_path=output_path,
                process_options=options
            )
            
            if result and os.path.exists(output_path):
                print(f"✓ Successfully created: {output_path}")
                return True
            else:
                print(f"✗ Failed to process: {target_image_path}")
                return False
                
        except Exception as e:
            print(f"Error during face swap: {e}")
            return False

    def batch_process_images(self,
                            source_image_path: str,
                            target_folder: str,
                            output_folder: str = None,
                            swap_model: str = "inswapper_128_fp16",
                            enhancer: str = None,
                            swap_mode: str = "all",
                            **kwargs) -> Dict[str, bool]:
        
        if not self.initialized:
            print("API not initialized. Call initialize() first.")
            return {}
        
        if not os.path.exists(source_image_path):
            print(f"Source image not found: {source_image_path}")
            return {}
            
        if not os.path.exists(target_folder):
            print(f"Target folder not found: {target_folder}")
            return {}
        
        # Load source faces
        source_faces_loaded = self.load_source_faces([source_image_path])
        if source_faces_loaded == 0:
            print("No source faces found")
            return {}
        
        # Set output folder
        if output_folder is None:
            output_folder = os.path.join(roop.globals.output_path, "batch_output")
        os.makedirs(output_folder, exist_ok=True)
        
        # Get all image files
        valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
        image_files = []
        for filename in os.listdir(target_folder):
            if filename.lower().endswith(valid_extensions):
                image_files.append(filename)
        
        if not image_files:
            print("No valid image files found in target folder")
            return {}
        
        results = {}
        
        # Create ProcessOptions
        options = ProcessOptions(
            execution_providers=roop.globals.execution_providers,
            execution_threads=roop.globals.execution_threads,
            max_memory=roop.globals.max_memory,
            face_swap_model=swap_model,
            face_enhancer=enhancer,
            swap_mode=swap_mode,
            vr_mode=roop.globals.vr_mode
        )
        
        print(f"Processing {len(image_files)} images...")
        
        for i, filename in enumerate(image_files):
            target_path = os.path.join(target_folder, filename)
            output_filename = f"swapped_{filename}"
            output_path = os.path.join(output_folder, output_filename)
            
            try:
                result = batch_process_regular(
                    target_path=target_path,
                    output_path=output_path,
                    process_options=options
                )
                
                success = result and os.path.exists(output_path)
                results[target_path] = success
                
                if success:
                    print(f"✓ Processed {i+1}/{len(image_files)}: {filename}")
                else:
                    print(f"✗ Failed {i+1}/{len(image_files)}: {filename}")
                    
                if self.processing_callback:
                    self.processing_callback((i+1, len(image_files)))
                    
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                results[target_path] = False
        
        success_count = sum(results.values())
        print(f"✓ Batch processing complete: {success_count}/{len(image_files)} successful")
        return results

    def process_video(self,
                     source_image_path: str,
                     target_video_path: str,
                     output_path: str = None,
                     swap_model: str = "inswapper_128_fp16",
                     enhancer: str = None,
                     swap_mode: str = "all",
                     keep_frames: bool = False,
                     keep_audio: bool = True,
                     **kwargs) -> bool:
        
        if not self.initialized:
            print("API not initialized. Call initialize() first.")
            return False
        
        if not os.path.exists(source_image_path):
            print(f"Source image not found: {source_image_path}")
            return False
            
        if not os.path.exists(target_video_path):
            print(f"Target video not found: {target_video_path}")
            return False
        
        # Load source faces
        source_faces_loaded = self.load_source_faces([source_image_path])
        if source_faces_loaded == 0:
            print("No source faces found")
            return False
        
        # Generate output path if not provided
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_filename = Path(target_video_path).stem
            output_path = f"video_{video_filename}_{timestamp}.mp4"
            output_path = os.path.join(roop.globals.output_path, output_path)
        
        # Create ProcessOptions
        options = ProcessOptions(
            execution_providers=roop.globals.execution_providers,
            execution_threads=roop.globals.execution_threads,
            max_memory=roop.globals.max_memory,
            face_swap_model=swap_model,
            face_enhancer=enhancer,
            swap_mode=swap_mode,
            vr_mode=roop.globals.vr_mode
        )
        
        try:
            print(f"Processing video: {os.path.basename(target_video_path)}")
            print("This may take a while depending on video length...")
            
            result = batch_process_with_options(
                target_path=target_video_path,
                output_path=output_path,
                process_options=options,
                keep_frames=keep_frames,
                keep_audio=keep_audio
            )
            
            if result and os.path.exists(output_path):
                print(f"✓ Successfully created video: {output_path}")
                return True
            else:
                print(f"✗ Failed to process video: {target_video_path}")
                return False
                
        except Exception as e:
            print(f"Error during video processing: {e}")
            return False

    def set_processing_callback(self, callback: Callable):
        self.processing_callback = callback

    def cleanup(self):
        if self.virtual_cam_active:
            self.stop_virtual_camera()
        if self.stream_writer:
            self.stream_writer.Close()
        self.clip_model = None
        self.clip_preprocess = None
        self.clip_tokenizer = None
        self.clipseg_model = None
        self.initialized = False
        print("✓ Roop Unleashed cleaned up")