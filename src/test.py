import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from scipy.spatial.distance import cdist
import os
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from model import ARCIKELM


# feature_model = models.resnet50(pretrained=True)
# feature_model = torch.nn.Sequential(*list(feature_model.children())[:-1])
# classifier = ARCIKELM(C=1.0, kernel='rbf', gamma='scale')

        
def extract_features( image_path, model):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
    ])
    
    try:
        image = Image.open(image_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0)
        
        with torch.no_grad():
            features = model(input_tensor)
        
        return features.squeeze().numpy()
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        return None





import os

def main():

    # classifier.initialize(features, np.array(valid_labels))

    # classifier.sequential_learning(features, np.array(valid_labels))
        

    # features = extract_features(image_path)

    base_model = models.resnet50(pretrained=True)
    base_model.fc = torch.nn.Identity()  # Usuń klasyfikator, aby uzyskać cechy

    model = ARCIKELM(C=1.0, kernel='rbf', gamma='scale')

    image_path = "C:/Users/jakub/Desktop/PULP/STUDIA/Praca mgr/DATASET_COINS/avers/CHF/CHF_0,1/CHFc10_0003.jpg" # class 1
    features = extract_features(image_path, base_model)
    initial_features = features.reshape(1, -1)
    image_path = "C:/Users/jakub/Desktop/PULP/STUDIA/Praca mgr/DATASET_COINS/avers/SEK/SEK_5/sek5_00001_a.jpg" # class 2
    features = extract_features(image_path, base_model)
    initial_features_2 = features.reshape(1, -1)
    initial_features = np.vstack((initial_features, initial_features_2))

    model.initialize(initial_features, np.array([[0], [1]]))

    image_path = "C:/Users/jakub/Desktop/PULP/STUDIA/Praca mgr/DATASET_COINS/avers/GBP/GBP_1/GBP1_0003.jpg" # class 3
    features = extract_features(image_path, base_model)
    initial_features = features.reshape(1, -1)

    model.add_new_class(initial_features, np.array([2]))

    image_path = "C:/Users/jakub/Desktop/PULP/STUDIA/Praca mgr/DATASET_COINS/avers/GBP/GBP_0,2/GBPc20_0003.jpg" # class 4
    features = extract_features(image_path, base_model)
    initial_features = features.reshape(1, -1)
    model.add_new_class(initial_features, np.array([3]))

    
    image_path = "C:/Users/jakub/Desktop/PULP/STUDIA/Praca mgr/DATASET_COINS/avers/SEK/SEK_5/sek5_00087_a.jpg" # class 2
    features = extract_features(image_path, base_model)
    initial_features = features.reshape(1, -1)
    predictions = model.sequential_learning(initial_features, np.array([1]))

    image_path = "C:/Users/jakub/Desktop/PULP/STUDIA/Praca mgr/DATASET_COINS/avers/CHF/CHF_0,1/CHFc10_0003.jpg" # class 1 the same image
    features = extract_features(image_path, base_model)
    initial_features = features.reshape(1, -1)
    predictions = model.predict(initial_features)
    print(predictions, 'TRUE: 0')

    image_path = "C:/Users/jakub/Desktop/PULP/STUDIA/Praca mgr/DATASET_COINS/avers/SEK/SEK_5/sek5_00087_a.jpg" # class 2
    features = extract_features(image_path, base_model)
    initial_features = features.reshape(1, -1)
    predictions = model.predict(initial_features)
    print(predictions, 'TRUE: 1')

    

if __name__ == "__main__":
    main()
