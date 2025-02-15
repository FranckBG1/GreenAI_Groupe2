import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor

class CLIPClassifier(nn.Module):
    def __init__(self, num_classes):
        super(CLIPClassifier, self).__init__()
        # Charger CLIP pré-entraîné
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        # Geler les poids de CLIP (optionnel)
        for param in self.clip.parameters():
            param.requires_grad = False

        # Taille correcte des features image
        self.image_feature_dim = 512  # CLIP ViT-B/32 retourne des features de taille 512

        # Ajouter une couche de classification
        self.classifier = nn.Linear(self.image_feature_dim, num_classes)
        
    def forward(self, pixel_values):
        # Extraire les features de l'image via CLIP
        image_features = self.clip.get_image_features(pixel_values=pixel_values)
        # Classification
        logits = self.classifier(image_features)
        return logits

def get_processor():
    # Pour prétraiter les images
    return CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
