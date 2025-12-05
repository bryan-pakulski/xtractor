import os
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
import numpy as np
import shutil
from tqdm import tqdm
import argparse

# 1. Setup Arguments
parser = argparse.ArgumentParser(description="Select a diverse subset of images.")
parser.add_argument("--source", type=str, required=True, help="Path to input directory")
parser.add_argument("--dest", type=str, required=True, help="Path to output directory")
parser.add_argument("--count", type=int, default=500, help="Number of images to select")
args = parser.parse_args()


# 2. Define a simple Dataset loader
class ImageFolderDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.files = [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
        ]
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.files[idx]


# 3. Embedding Pipeline
def get_embeddings(dataset, batch_size=64):
    # Use a lightweight ResNet18
    model = models.resnet18(pretrained=True)
    # Remove classification layer to get raw feature vectors
    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    embeddings = []
    file_paths = []

    print("Computing embeddings...")
    with torch.no_grad():
        for images, paths in tqdm(loader):
            images = images.to(device)
            # Forward pass returns (Batch, 512, 1, 1), flatten to (Batch, 512)
            emb = model(images).squeeze(-1).squeeze(-1).cpu().numpy()
            embeddings.append(emb)
            file_paths.extend(paths)

    return np.vstack(embeddings), file_paths


# 4. Selection Logic
def main():
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    dataset = ImageFolderDataset(args.source, transform=transform)

    # Check if we have enough images
    if len(dataset) < args.count:
        print(
            f"Requested {args.count} images but only found {len(dataset)}. Copying all."
        )
        selected_indices = range(len(dataset))
    else:
        features, file_paths = get_embeddings(dataset)

        print(f"Clustering into {args.count} representative groups...")
        # K-Means tries to find 'count' centroids that minimize variance
        kmeans = KMeans(n_clusters=args.count, random_state=42, n_init=10)
        kmeans.fit(features)

        # Find the actual image closest to each centroid
        closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, features)
        selected_paths = [file_paths[i] for i in closest]

    # 5. Copy Files
    os.makedirs(args.dest, exist_ok=True)
    print(f"Copying {len(selected_paths)} diverse images to {args.dest}...")

    for src in tqdm(selected_paths):
        shutil.copy(src, os.path.join(args.dest, os.path.basename(src)))


if __name__ == "__main__":
    main()
