import torch
import clip
from PIL import Image

# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Load and preprocess image
image = preprocess(Image.open("test_img.jpg")).unsqueeze(0).to(device)

# Generate embedding
with torch.no_grad():
    image_embedding = model.encode_image(image)

# Normalize (important!)
image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)

print(image_embedding.shape)  # [1, 512]
