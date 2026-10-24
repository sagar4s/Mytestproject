# roop_api.py - WITH FIXED INITIALIZATION
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
        self.output_image_format = "png"  # Fallback storage
        self.output_video_format = "mp4"  # Fallback storage
        
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
                   load_clip_models: bool = False,  # Disable by default for stability
                   load_clipseg_model: bool = False,  # Disable by default for stability
                   clip_model_name: str = "ViT-B/32") -> bool:
        
        try:
            if execution_providers is None:
                execution_providers = self._auto_detect_providers()
                
            print(f"Initializing with providers: {execution_providers}")
            
            # Set basic globals - only set what definitely exists
            roop.globals.execution_providers = execution_providers
            roop.globals.execution_threads = max_threads
            roop.globals.max_memory = max_memory
            roop.globals.output_path = output_path
            
            # Try to set optional globals if they exist
            optional_globals = {
                'video_encoder': video_encoder,
                'video_quality': video_quality, 
                'log_level': log_level,
                'distance_threshold': face_distance_threshold,
                'blend_ratio': blend_ratio,
                'subsample_size': subsample_size,
                'vr_mode': enable_vr_mode
            }
            
            for attr, value in optional_globals.items():
                if hasattr(roop.globals, attr):
                    setattr(roop.globals, attr, value)
                else:
                    print(f"Note: roop.globals.{attr} not available")
            
            # Handle CFG safely - FIXED APPROACH
            self.output_image_format = output_image_format
            self.output_video_format = output_video_format
            
            # Try to set CFG if it exists and is writable
            if hasattr(roop.globals, 'CFG') and roop.globals.CFG is not None:
                try:
                    roop.globals.CFG.output_image_format = output_image_format
                    roop.globals.CFG.output_video_format = output_video_format
                    print("✓ Set output formats in roop.globals.CFG")
                except (AttributeError, TypeError) as e:
                    print(f"Note: Could not set CFG formats: {e}")
                    print("Using internal format storage")
            else:
                print("Note: roop.globals.CFG not available, using internal storage")
            
            # Initialize face sets
            if not hasattr(roop.globals, 'INPUT_FACESETS'):
                roop.globals.INPUT_FACESETS = []
            else:
                roop.globals.INPUT_FACESETS = []
                
            if not hasattr(roop.globals, 'TARGET_FACES'):
                roop.globals.TARGET_FACES = []
            else:
                roop.globals.TARGET_FACES = []
            
            # Initialize face analysis if the attribute exists
            if hasattr(roop.globals, 'g_desired_face_analysis'):
                roop.globals.g_desired_face_analysis = [
                    "landmark_3d_68", "landmark_2d_106", "detection", "recognition"
                ]
            
            # Create output directory
            os.makedirs(output_path, exist_ok=True)
            
            # Load CLIP models only if explicitly requested
            if load_clip_models and CLIP_AVAILABLE:
                self._load_clip_models(clip_model_name)
            else:
                print("CLIP models disabled for stability")
            
            # Load CLIPSeg model only if explicitly requested  
            if load_clipseg_model and CLIPSEG_AVAILABLE:
                self._load_clipseg_model()
            else:
                print("CLIPSeg model disabled for stability")
            
            self.initialized = True
            print(f"✓ Roop Unleashed initialized successfully with {execution_providers}")
            return True
            
        except Exception as e:
            print(f"✗ Failed to initialize Roop Unleashed: {e}")
            return False

    def _auto_detect_providers(self) -> List[str]:
        try:
            import onnxruntime
            available = onnxruntime.get_available_providers()
            if 'CUDAExecutionProvider' in available:
                print("✓ CUDA execution provider available")
                return ['CUDAExecutionProvider']
            elif 'DmlExecutionProvider' in available:
                print("✓ DML execution provider available")
                return ['DmlExecutionProvider']
            else:
                print("✓ Using CPU execution provider")
                return ['CPUExecutionProvider']
        except Exception as e:
            print(f"✓ Using CPU execution provider (fallback): {e}")
            return ['CPUExecutionProvider']

    # ... [Keep all the other methods the same as before] ...
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
            output_path = f"result_{target_filename}_{timestamp}.{self.output_image_format}"
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
            swap_mode=swap_mode
        )
        
        # Add VR mode if available
        if hasattr(roop.globals, 'vr_mode'):
            options.vr_mode = roop.globals.vr_mode
        
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