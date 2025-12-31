import os
import torch
import clip
from PIL import Image
from tqdm import tqdm

def process_video(frames_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()

    frame_files = sorted(os.listdir(frames_dir))
    features = []

    with torch.no_grad():
        for frame in tqdm(frame_files):
            img_path = os.path.join(frames_dir, frame)
            image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
            emb = model.encode_image(image)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            features.append(emb.cpu())

    features = torch.cat(features, dim=0)
    return features  # shape: [num_frames, 512]
