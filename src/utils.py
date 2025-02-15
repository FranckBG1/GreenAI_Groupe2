import os
from torch.utils.data import Dataset
from PIL import Image

class CustomImageDataset(Dataset):
    def __init__(self, image_paths, labels, processor, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.processor = processor
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # 🔹 Prétraitement avec CLIP
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)
        label = self.labels[idx]

        return pixel_values, label


def load_dataset_images(data_dir):
    """Charge les images et leurs labels depuis le dossier structuré."""
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"❌ Le dossier {data_dir} n'existe pas ! Vérifie le chemin.")

    # 🔹 Correction des noms de classes et labels associés
    class_to_idx = {"Normal": 0, "Anomaly": 1}  # Normal = 0, Anomaly = 1
    images = []
    labels = []

    # 🔹 Vérification des sous-dossiers
    sous_dossiers = os.listdir(data_dir)
    print(f"📂 Dossiers trouvés : {sous_dossiers}")

    for cls, idx in class_to_idx.items():
        cls_dir = os.path.join(data_dir, cls)
        if not os.path.isdir(cls_dir):
            print(f"⚠️ Attention : le dossier '{cls}' est introuvable dans {data_dir}. Vérifie la structure !")
            continue

        fichiers = [f for f in os.listdir(cls_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        print(f"📂 Classe '{cls}' ({idx}) : {len(fichiers)} images chargées.")

        for file in fichiers:
            images.append(os.path.join(cls_dir, file))
            labels.append(idx)

    print(f"✅ Nombre total d'images chargées : {len(images)}")
    
    if len(images) == 0:
        raise ValueError("❌ Aucune image trouvée ! Vérifie que les fichiers sont dans les bons dossiers.")

    return images, labels, class_to_idx
