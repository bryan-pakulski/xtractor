<p align="center">
  <img width="384" height="384" alt="xtractor" src="https://github.com/user-attachments/assets/f2161d7f-ec1e-4a81-bf16-5409fde08eaa" />
</p>

<h2 align="center">XTRACTOR</h2>

Unique video frame extractor using dinov3 - Building datasets for Computer Vision models.

Uses embeddings & feature extraction to pull only the most relevant frames from a directory of videos / single video.

Features:
- Extract frames from video at a specified initial sampling rate, from there you can filter out X% significant frames to keep
- Automatically remove duplicate frames based on cosine similarity
- Outliers are filtered and saved in a seperate location, based on similarity to neighboring frames

# Requirements
Hugging Face API token is required.

Can be passed as an argument or set in the environment variable `HF_TOKEN`.

# Setup
```
# If using pyenv for environment management
# This has only been tested on this python version
pyenv install 3.10.18

pip install -r requirements.txt
```

# Usage
```
options:
  -h, --help               show this help message and exit
  --input INPUT            Input path, either video file or directory of videos
  --output OUTPUT          Output directory
  --ratio RATIO            Percentage of frames to keep
  --fps FPS                Initial sampling rate in frames per second
  --device DEVICE          Device to use for inference
  --token TOKEN            Huggingface token
  --batch_size BATCH_SIZE  Batch size for inference
  --remove_duplicates      Remove duplicate frames based on cosine similarity
  --threshold THRESHOLD    Threshold for cosine similarity
```

# Tools
An additional tool is included to help with selecting a distibution of frames from a dataset, it uses dinov2 to get embeddings and
select a count of further neighbours to keep trying to make sure an even distrubution is kept.
