import os
import tqdm
import torch
import random
import numpy as np
from PIL import Image
import albumentations as A
from torchvision import transforms
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def test_side_classificator(model, dataloader, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)

    txt_path = os.path.join(output_dir, "side_metrics.txt")
    all_preds = []
    all_labels = []

    with open(txt_path, "a", encoding="utf-8") as f:
        total_samples = 0
        wrong_predictions = 0

        for image_paths, classes, _, _ in tqdm.tqdm(dataloader, desc="Testing ... "):
            features = model.reliefF(image_paths, 'side')

            if features is None:
                raise KeyError('Lack of features')

            side_labels = [1 if cls == 'revers' else 0 for cls in classes[0]['side']]
            features_scaled = model.side_scaler.transform(features)
            side_preds = model.side_classifier.predict(features_scaled)

            batch_errors = sum(p != l for p, l in zip(side_preds, side_labels))
            wrong_predictions += batch_errors
            total_samples += len(side_labels)

            all_preds.extend(side_preds)
            all_labels.extend(side_labels)

        error_rate = wrong_predictions / total_samples * 100
        f.write(f"Total wrong predictions: {wrong_predictions}/{total_samples} Accuracy: {100 - error_rate:.2f}%")

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['avers', 'revers'])
    disp.plot(cmap=plt.cm.Blues)

    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()


class ImageAugmentation:
    def __init__(self):
        self.transform = A.Compose([
            A.Rotate(limit=180, p=1.0), 
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=10, p=1.0), 
            A.RandomBrightnessContrast(p=0.5), 
            A.GaussNoise(p=0.5),  
            A.GaussianBlur(blur_limit=(3, 5), p=0.3),  
        ])

        self.pre_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def generate_augmented_images(self, img, n_augments: int = 10):
        augmented = []
        
        for _ in range(n_augments):
            augmented_image = self.transform(image=np.array(img))['image']
            
            img_pil = Image.fromarray(augmented_image)
            img_tensor = self.pre_transform(img_pil)
            
            augmented.append(img_tensor)
        
        augmented_tensor = torch.stack(augmented)
        return augmented_tensor

