import torch

from PIL import Image
import matplotlib.pyplot as plt
import os
import sys


# Ajouter le dossier `src` au chemin d'import
sys.path.append(os.path.abspath("./src"))

from src.model import CLIPClassifier, get_processor



# 🔹 Charger le modèle entraîné
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 2  # Normal (0) et Anomaly (1)
model = CLIPClassifier(num_classes=num_classes).to(device)

# Charger les poids entraînés
model_load_path = "./clip_model.pth"
if not os.path.exists(model_load_path):
    raise FileNotFoundError(f"❌ Le fichier {model_load_path} n'existe pas ! Lance `train.py` d'abord.")

model.load_state_dict(torch.load(model_load_path, map_location=device))
model.eval()
print("✅ Modèle chargé avec succès !")

# 🔹 Charger le processor CLIP
processor = get_processor()

# 🔹 Tester 2 images situées à la racine du projet
test_images = ["./001.JPG", "./0004.JPG"]  # Assure-toi que ces images existent

fig, axes = plt.subplots(1, len(test_images), figsize=(10, 5))

for i, img_path in enumerate(test_images):
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"❌ L'image {img_path} est introuvable !")

    # Charger l'image
    image = Image.open(img_path).convert("RGB")

    # Prétraiter avec CLIP
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].squeeze(0).to(device)

    # Prédire la classe
    with torch.no_grad():
        logits = model(pixel_values.unsqueeze(0))
        predicted_label = torch.argmax(logits, dim=1).item()

    # Affichage
    axes[i].imshow(image)
    axes[i].set_title(f"Prédit : {'Normal' if predicted_label == 0 else 'Anomaly'}", fontsize=12)
    axes[i].axis("off")

plt.show()
