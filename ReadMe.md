<p align="center">
  <img width="384" height="384" alt="xtractor" src="https://github.com/user-attachments/assets/f2161d7f-ec1e-4a81-bf16-5409fde08eaa" />
</p>

<h2 align="center">XTRACTOR</h2>

Unique video frame extractor using dinov3 - Building datasets for Computer Vision models.

Uses embeddings & feature extraction to pull only the most relevant frames from a directory of videos / single video.

Also detects camera movement or significant outliers by looking for frames with an extremely low similarity within a sliding window to their direct neighbors. 

# Requirements
Hugging Face API token is required.

Can be passed as an argument or set in the environment variable `HF_TOKEN`.

# Setup
```
pip install -r requirements.txt
```

# Usage
```
usage: xtract.py [-h] --input INPUT --output OUTPUT [--ratio RATIO] [--fps FPS] [--device DEVICE] [--token TOKEN] [--batch_size BATCH_SIZE]

options:
  -h, --help            show this help message and exit
  --input INPUT         Input path, either video file or directory of videos
  --output OUTPUT       Output directory
  --ratio RATIO         Percentage of frames to keep
  --fps FPS             Initial sampling rate in frames per second
  --device DEVICE       Device to use for inference
  --token TOKEN         Huggingface token
  --batch_size BATCH_SIZE
```
