# run_video_swap.py
import os
from pathlib import Path
from roop_api import RoopUnleashedAPI

def main():
    # Initialize API
    api = RoopUnleashedAPI()
    
    def progress_callback(progress):
        current, total = progress
        print(f"Progress: {current}/{total} ({(current/total)*100:.1f}%)")
    
    api.set_processing_callback(progress_callback)
    
    try:
        # Initialize with GPU for faster video processing
        print("Initializing Roop Unleashed for Video Processing...")
        api.initialize(
            execution_providers=['CUDAExecutionProvider'],  # Use CPU if GPU not available
            max_threads=4,
            max_memory=8,
            output_path="./output",
            face_distance_threshold=0.65,
            blend_ratio=0.65,
            video_encoder="libx264",
            video_quality=18  # Lower number = better quality
        )
        
        # Define your file paths - UPDATE THESE!
        source_face = "source_faces/your_source_face.jpg"  # Person to swap in
        target_video = "target_videos/your_target_video.mp4"  # Video to swap onto
        
        # Create directories if they don't exist
        os.makedirs("source_faces", exist_ok=True)
        os.makedirs("target_videos", exist_ok=True)
        os.makedirs("output", exist_ok=True)
        
        # VIDEO FACE SWAP
        if os.path.exists(source_face) and os.path.exists(target_video):
            print("\n" + "="*50)
            print("RUNNING VIDEO FACE SWAP")
            print("="*50)
            print("This may take a while depending on video length...")
            print("For a 1-minute video, expect 5-15 minutes processing time")
            
            success = api.process_video(
                source_image_path=source_face,
                target_video_path=target_video,
                swap_model="inswapper_128_fp16",  # Use this for videos (faster)
                enhancer="GFPGAN",  # Improve face quality
                swap_mode="all",  # Swap all faces found
                keep_frames=False,  # Don't keep temporary frames
                keep_audio=True  # Keep original audio
            )
            
            if success:
                print("🎉 Video face swap completed successfully!")
                print("Check the 'output' folder for your processed video")
            else:
                print("❌ Video face swap failed")
        else:
            print(f"Please add source face to: {source_face}")
            print(f"Please add target video to: {target_video}")
            print("Supported video formats: MP4, AVI, MOV, MKV")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        api.cleanup()

if __name__ == "__main__":
    main()