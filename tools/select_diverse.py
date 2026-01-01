import os
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFile
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import pairwise_distances_argmin_min
import numpy as np
import shutil
from tqdm import tqdm
import argparse
import gc

# Allow loading truncated images to prevent crashing on bad files
ImageFile.LOAD_TRUNCATED_IMAGES = True

parser = argparse.ArgumentParser(description="Select a diverse subset of images.")
parser.add_argument("--source", type=str, required=True, help="Path to input directory")
parser.add_argument("--dest", type=str, required=True, help="Path to output directory")
parser.add_argument("--count", type=int, default=500, help="Number of images to select")
parser.add_argument(
    "--batch_size", type=int, default=32, help="Batch size for processing"
)
parser.add_argument(
    "--workers", type=int, default=4, help="Number of dataloader workers"
)
args = parser.parse_args()


class ImageFolderDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.files = [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff"))
        ]
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        try:
            with open(path, "rb") as f:
                img = Image.open(f)
                img = img.convert("RGB")

                if self.transform:
                    img = self.transform(img)
                return img, path
        except Exception as e:
            print(f"Error loading {path}: {e}")
            # Return a dummy tensor if file is corrupt to prevent crash
            return torch.zeros(3, 224, 224), path


def get_embeddings(dataset, batch_size=32, workers=4):
    model = models.resnet18(pretrained=True)
    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")
    model.to(device)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
    )

    embeddings = []
    file_paths = []

    print("Computing embeddings...")
    with torch.no_grad():
        for images, paths in tqdm(loader):
            images = images.to(device, non_blocking=True)

            emb = model(images).squeeze(-1).squeeze(-1)

            # Move to CPU immediately to free VRAM
            embeddings.append(emb.cpu().numpy())
            file_paths.extend(paths)

            del images, emb

    # Clean up model to free VRAM before clustering
    del model
    torch.cuda.empty_cache()
    gc.collect()

    return np.vstack(embeddings), file_paths


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

    if len(dataset) < args.count:
        print(
            f"Requested {args.count} images but only found {len(dataset)}. Copying all."
        )
        selected_paths = dataset.files
    else:
        features, file_paths = get_embeddings(dataset, args.batch_size, args.workers)

        print(
            f"Clustering {len(features)} items into {args.count} representative groups..."
        )

        kmeans = MiniBatchKMeans(
            n_clusters=args.count, random_state=42, batch_size=2048, n_init=10
        )
        kmeans.fit(features)

        print("Finding closest images to centroids...")
        closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, features)
        selected_paths = [file_paths[i] for i in closest]

    os.makedirs(args.dest, exist_ok=True)
    print(f"Copying {len(selected_paths)} diverse images to {args.dest}...")

    for src in tqdm(selected_paths):
        try:
            shutil.copy(src, os.path.join(args.dest, os.path.basename(src)))
        except OSError as e:
            print(f"Failed to copy {src}: {e}")


if __name__ == "__main__":
    main()
