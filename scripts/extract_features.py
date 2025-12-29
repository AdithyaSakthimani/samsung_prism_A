import os
import torch
import clip
from PIL import Image
from tqdm import tqdm

# Paths
FRAME_DIR = "../data/frames"
FEATURE_DIR = "../data/features"

os.makedirs(FEATURE_DIR, exist_ok=True)

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load CLIP
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()

# Loop over each video folder
for video_name in os.listdir(FRAME_DIR):
    video_path = os.path.join(FRAME_DIR, video_name)

    if not os.path.isdir(video_path):
        continue

    print(f"Processing video: {video_name}")

    frame_files = sorted(os.listdir(video_path))
    embeddings = []

    for frame in tqdm(frame_files):
        frame_path = os.path.join(video_path, frame)

        image = preprocess(Image.open(frame_path)).unsqueeze(0).to(device)

        with torch.no_grad():
            emb = model.encode_image(image)
            emb = emb / emb.norm(dim=-1, keepdim=True)

        embeddings.append(emb.cpu())

    # Stack frames → [T, 512]
    video_embedding = torch.cat(embeddings, dim=0)

    # Save
    save_path = os.path.join(FEATURE_DIR, f"{video_name.replace(' ', '_')}.pt")
    torch.save(video_embedding, save_path)

    print(f"Saved: {save_path}, shape: {video_embedding.shape}")

