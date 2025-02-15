import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from codecarbon import EmissionsTracker
import pandas as pd
from model import CLIPClassifier, get_processor
from utils import load_dataset_images, CustomImageDataset
import os

# 🔹 Démarrer le suivi des émissions carbone avec CodeCarbon
tracker = EmissionsTracker(project_name="IM-IAD_CLIP", output_dir=".")

# ✅ Démarrage de la mesure des émissions
tracker.start()

# 🔹 Configuration du dataset
data_dir = "./data/Images"  # Assure-toi que ce chemin est correct !
batch_size = 8
num_epochs = 5
learning_rate = 1e-3

# 🔹 Chargement du dataset d'entraînement
image_paths, labels, class_to_idx = load_dataset_images(data_dir)

if len(image_paths) == 0:
    raise ValueError("❌ Aucune image trouvée ! Vérifie que le dossier ./data/Images contient des sous-dossiers avec des images.")

processor = get_processor()
dataset = CustomImageDataset(image_paths, labels, processor)
train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 🔹 Vérification des premières images chargées
print(f"🔍 Exemple d'images chargées : {image_paths[:5]}")
print(f"🔍 Labels associés : {labels[:5]}")

# 🔹 Charger le modèle CLIP
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = len(class_to_idx)
model = CLIPClassifier(num_classes=num_classes).to(device)
optimizer = torch.optim.Adam(model.classifier.parameters(), lr=learning_rate)

# 🔹 Entraînement du modèle
model.train()
for epoch in range(num_epochs):
    total_loss = 0
    for pixel_values, labels in train_dataloader:
        pixel_values = pixel_values.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(pixel_values)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {total_loss/len(train_dataloader):.4f}")

# ✅ Arrêter CodeCarbon et récupérer les émissions CO₂
tracker.stop()

# 🔹 Lire les émissions enregistrées par CodeCarbon
emissions_df = pd.read_csv("emissions.csv")

# ✅ Calcul de l'impact environnemental
eq_kg_co2 = emissions_df["emissions"].sum()  # Total des émissions en kg CO₂
max_eq_kg_co2 = 1  # 1kwh en moyenne mondiale qui vaut 0.056 kg/co2  On prend notre modèle comme référence pour le moment

# Accuracy obtenue après test (remplace par la vraie valeur après l'évaluation)
accuracy = 0.9427  

# Appliquer la formule de l'impact environnemental
impact_env = accuracy * (1 - (eq_kg_co2 / max_eq_kg_co2))

print(f"🌍 Impact Environnemental Calculé : {impact_env:.4f}")

# 🔹 Sauvegarde du modèle après entraînement
model_save_path = "./clip_model.pth"
torch.save(model.state_dict(), model_save_path)
print(f"✅ Modèle sauvegardé sous {model_save_path}")
