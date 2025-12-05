import torch
from transformers import pipeline
from datasets import load_dataset, Image
import tempfile
import shutil
import cv2
import os
import sys
import time

import argparse

SUPPORTED_VIDEO_FORMATS = [".mov", ".mp4", ".avi", ".mkv", ".webm"]

def strip_video(filepath, outpath, fps=1):
    """
    Extract frames from video at specified fps
    Args:
        filepath: Path to video file
        outpath: Output directory for frames
        fps: Frames per second to extract (default 1)
    """
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        sys.exit(1)
    
    video = cv2.VideoCapture(filepath)
    
    video_fps = video.get(cv2.CAP_PROP_FPS)
    frame_interval = int(video_fps / fps)
    
    count = 0
    frame_count = 0
    
    while True:
        success = video.grab()
        if not success:
            break
            
        if frame_count % frame_interval == 0:
            success, frame = video.retrieve()
            if success:
                cv2.imwrite(os.path.join(outpath, f"frame_{count:06d}.jpg"), frame)
                count += 1
                
        frame_count += 1
    
    print(f"Extracted {count} frames")
    video.release()
    return count

def filter_duplicates(embeddings_db, threshold=0.98):
    """
    Remove duplicate frames based on cosine similarity
    """
    if not embeddings_db:
        return {}

    filenames = sorted(list(embeddings_db.keys()))
    unique_db = {}

    # First frame is always kept
    prev_filename = filenames[0]
    unique_db[prev_filename] = embeddings_db[prev_filename]

    duplicates_count = 0

    for i in range(1, len(filenames)):
        curr_filename = filenames[i]

        #embeddings
        emb_1 = embeddings_db[prev_filename]
        emb_2 = embeddings_db[curr_filename]
        cos_sim = torch.nn.functional.cosine_similarity(emb_1, emb_2, dim=0).item()

        if cos_sim < threshold:
            unique_db[curr_filename] = embeddings_db[curr_filename]
            prev_filename = curr_filename
        else:
            duplicates_count += 1

    print(f"Removed {duplicates_count} duplicate frames with at least {threshold} cosine similarity")
    return unique_db

def compute_embeddings(image_dir, pipe, batch_size=16):
    """
    Compute embeddings for all images in image_dir
    Args:
        image_dir: Directory containing images
        pipe: Huggingface pipeline model
        batch_size: Batch size for processing
    """
    embeddings_db = {}
    
    dataset = load_dataset("imagefolder", 
                         data_dir=image_dir, 
                         split="train").cast_column("image", Image(decode=False))

    
    start_time = time.time()
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i + batch_size]
        filenames = [ob["path"] for ob in batch["image"]]
        outputs = pipe(filenames)
        
        for filename, embedding in zip(filenames, outputs):
            embedding_tensor = torch.tensor(embedding)
            embedding_tensor = embedding_tensor / embedding_tensor.norm()
            embeddings_db[filename] = embedding_tensor[0].ravel()
    end_time = time.time()
    
    print(f"Computed {len(embeddings_db)} embeddings in {end_time - start_time} seconds")
    return embeddings_db

def filter_images(embeddings_db, ratio=0.5):
    """
    Filter images based on similarity
    Args:
        embeddings_db: Dictionary of image names to embeddings
        ratio: Percentage of frames to keep
    Returns:
        tuple: (selected_files, shift_frames) where selected_files are the regular filtered frames
        and shift_frames are frames with significant camera movement
    """
    
    embeddings = torch.stack(list(embeddings_db.values()))
    filenames = list(embeddings_db.keys())    
    similarities = torch.mm(embeddings, embeddings.t())
    avg_similarities = similarities.mean(dim=1)
    
    # Find regular distinct frames
    num_keep = int(len(filenames) * ratio)
    _, indices = torch.topk(avg_similarities, num_keep, largest=False)
    
    # Detect camera shifts and other anomalies by finding frames with very low similarity to their neighbors
    window_size = 5
    rolling_similarities = []
    
    for i in range(len(filenames)):
        start_idx = max(0, i - window_size)
        end_idx = min(len(filenames), i + window_size + 1)
        neighbor_similarities = similarities[i, start_idx:end_idx].mean()
        rolling_similarities.append(neighbor_similarities)
    
    rolling_similarities = torch.tensor(rolling_similarities)
    
    # Find frames with significantly lower similarity to their neighbors
    mean_sim = rolling_similarities.mean()
    std_sim = rolling_similarities.std()
    threshold = mean_sim - std_sim # NOTE: tweak here to control sensitivity, 1.std seems "mostly" reasonable
    
    shift_indices = torch.where(rolling_similarities < threshold)[0]
    shift_frames = [filenames[i] for i in shift_indices]
    selected_files = [filenames[i] for i in indices if i not in shift_indices]
    
    print(f"Selected {len(selected_files)}/{len(selected_files) + len(shift_frames)} frames")
    print(f"{len(shift_frames)} camera shifts / extreme outliers") 
    
    return selected_files, shift_frames

def extract_frames(input_path, output_path, fps, ratio, pipe, remove_duplicates, threshold):
    """
    Extract frames from input_path and save to output_path
    Args:
        input_path: Path to input video file or directory of videos
        output_path: Path to output directory
        fps: Frames per second to extract (default 1)
        ratio: Percentage of frames to keep
        pipe: Huggingface pipeline model
    """

    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Extracting frames at {fps} FPS... to {temp_dir}")
        frame_count = strip_video(input_path, temp_dir, fps=fps)
        print(f"Extracted {frame_count} frames")

        print("Computing embeddings...")
        embeddings_db = compute_embeddings(temp_dir, pipe)

        if remove_duplicates:
            print("Filtering duplicate frames...")
            embeddings_db = filter_duplicates(embeddings_db, threshold)
        
        print("Filtering distinct frames...")
        good_frames, shift_frames = filter_images(embeddings_db, ratio)

        os.makedirs(output_path, exist_ok=True)
        os.makedirs(os.path.join(output_path, "outliers"), exist_ok=True)
        for frame in good_frames:
            shutil.copy2(
                frame,
                os.path.join(output_path, os.path.basename(frame))
            )
        for frame in shift_frames:
            shutil.copy2(
                frame,
                os.path.join(output_path, "outliers", os.path.basename(frame))
            )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input path, either video file or directory of videos")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--ratio", type=float, default=0.5, help="Percentage of frames to keep")
    parser.add_argument("--fps", type=float, default=1, help="Initial sampling rate in frames per second")
    parser.add_argument("--device", type=str, default="auto", help="Device to use for inference")
    parser.add_argument("--token", type=str, default=None, help="Huggingface token")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for inference")
    parser.add_argument("--remove_duplicates", action="store_true", default=True, help="Remove duplicate frames based on cosine similarity")
    parser.add_argument("--threshold", type=float, default=0.98, help="Threshold for cosine similarity")

    args = parser.parse_args()

    if args.token is None:
        print("No token provided, attempting to load from environment variable HF_TOKEN")
        token = os.environ["HF_TOKEN"]
    else:
        token = args.token

    from huggingface_hub import login
    login(token = token)

    print("Loading model...")    
    pretrained_model_name = "facebook/dinov3-convnext-tiny-pretrain-lvd1689m"    
    pipe = pipeline(
        model=pretrained_model_name,
        task="image-feature-extraction", 
        batch_size=args.batch_size
    )

    if os.path.isdir(args.input):
        print(f"Processing directory {args.input}")
        for f in os.listdir(args.input):
            ext = os.path.splitext(f)[1]
            if ext in SUPPORTED_VIDEO_FORMATS:
                base_name = os.path.splitext(f)[0]
                extract_frames(os.path.join(args.input, f), 
                               os.path.join(args.output, base_name), 
                               args.fps, 
                               args.ratio,
                               pipe,
                               args.remove_duplicates,
                               args.threshold)
            else:
                print(f"Skipping {f}, unsupported file type")
    else: 
        print(f"Processing video {args.input}")
        extract_frames(args.input,
                       args.output,
                       args.fps,
                       args.ratio,
                       pipe,
                       args.remove_duplicates,
                       args.threshold
                       )
        
    print("Done...")
    
if __name__ == "__main__":
    main()
