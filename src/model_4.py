import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import pickle
from skrebate import ReliefF
import joblib
import os
import torchvision.transforms as T
from PIL import Image
import numpy as np
import random
from utils import ImageAugmentation


class FeatureExtractorReliefF:
    '''
    A feature extractor class using a pretrained model and ReliefF for feature selection.
    
    Parameters:
    model (torch.nn.Module): The pretrained neural network model (e.g., ResNet50).
    n_features_to_select (int): The number of features to select using ReliefF.
    device (str): The device to run the model on, e.g., 'cpu' or 'cuda'.
    
    Returns:
    None
    '''
    def __init__(self, model, n_features_to_select=100, device='cuda'):
        self.n_features_to_select = n_features_to_select
        self.device = torch.device(device if torch.cuda.is_available() and device=='cuda' else 'cpu')
        self.relief = None
        self.selected_features_idx_side = None
        self.selected_features_idx_class = None

        self.model = model
        self.model.eval()
        

    def extract_features(self, batch_tensors, batch_size=32):
        '''
        Extract features from images using a pretrained model (e.g., ResNet50).
        
        Parameters:
        image_paths (list): List of paths to the images.
        batch_size (int): The batch size to process images.
        
        Returns:
        np.ndarray: Extracted features from all images.
        '''
        features = []
        with torch.no_grad():
            for i in range(len(batch_tensors)):
                # batch = torch.stack(batch_tensors).to(self.device)
                batch = batch_tensors[i].to(self.device)
                batch_features = self.model(batch.unsqueeze(0))
                features.extend(batch_features.squeeze((-1, -2)).cpu().numpy())
        
        return np.array(features)

    def fit(self, batch_tensors, labels, version, batch_size=32):
        '''
        Train ReliefF feature selection model.
        
        Parameters:
        image_paths (list): List of image paths.
        labels (list): Corresponding labels for the images.
        batch_size (int): The batch size to process images.
        
        Returns:
        self: The trained FeatureExtractorReliefF object.
        '''
        features = self.extract_features(batch_tensors, batch_size)
        
        self.relief = ReliefF(n_features_to_select=self.n_features_to_select, n_neighbors=10)
        self.relief.fit(features, labels)
        if version == 'side':
            self.selected_features_idx_side = self.relief.top_features_[:self.n_features_to_select]
        elif version == 'class':
            self.selected_features_idx_class = self.relief.top_features_[:self.n_features_to_select]
        else:
            KeyError('Wrong chossen version')
        
        return self

    def __call__(self, batch_tensors, version, batch_size=32):
        '''
        Apply the trained ReliefF model to extract and select features.
        
        Parameters:
        image_paths (list): List of image paths.
        batch_size (int): The batch size to process images.
        
        Returns:
        np.ndarray: Selected features for all images.
        '''
        if self.relief is None:
            raise ValueError("Model is not trained. Use the fit() method first.")
        
        features = self.extract_features(batch_tensors, batch_size)
        if len(features.shape) > 1:
            if version == 'side':
                selected_features = features[:,self.selected_features_idx_side]
            elif version == 'class':
                selected_features = features[:,self.selected_features_idx_class]
            else:
                KeyError('Wrong chossen version')
        else: 
            if version == 'side':
                selected_features = features[self.selected_features_idx_side]
            elif version == 'class':
                selected_features = features[self.selected_features_idx_class]
            else:
                KeyError('Wrong chossen version')

        return selected_features

    def save_relief(self, path, version):
        '''
        Save the trained ReliefF model to a file.
        
        Parameters:
        path (str): Path to save the trained model.
        
        Returns:
        None
        '''
        if self.relief is None:
            raise ValueError("Model is not trained. Use the fit() method first.")
        if version == 'side':
            selected_features = self.selected_features_idx_side
            path = f'{path}/reliefF_side.pkl'
        elif version == 'class':
            selected_features = self.selected_features_idx_class
            path = f'{path}/reliefF_class.pkl'
        with open(path, 'wb') as f:
            pickle.dump({
                'relief': self.relief,
                'selected_features_idx': selected_features
            }, f)

    def load_relief(self, version, path):
        '''
        Load a trained ReliefF model from a file.
        
        Parameters:
        path (str): Path to the trained model file.
        
        Returns:
        None
        '''
        if version == 'side':
            path = f'{path}/reliefF_side.pkl'
        elif version == 'class':
            path = f'{path}/reliefF_class.pkl'
        
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.relief = data['relief']
            if version == 'side':
                self.selected_features_idx_side = data['selected_features_idx']
            elif version == 'class':
                self.selected_features_idx_class = data['selected_features_idx']
            else:
                KeyError('Wrong chossen version')

class ARCIKELM:
    '''
    ARCIKELM: A custom implementation of a classification model with sequential learning and fuzzy membership.
    
    Parameters:
    C (float): Regularization parameter for the model.
    kernel (str): Kernel type to use for the classifier ('rbf' in this case).
    gamma (str or float): Parameter for the RBF kernel.
    
    Returns:
    None
    '''
    def __init__(self, C=1.0, kernel='rbf', gamma='scale', max_negative_samples=50):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.X_L = None  # Kernel matrix vectors
        self.beta = None  # Output weights
        self.n_classes = 0
        self.neuron_activation = None  # Neuron eligibility matrix
        self.class_counts = {}  # Track samples per class
        self.max_negative_samples = max_negative_samples

    def _compute_kernel(self, X, Y=None):
        '''
        Compute kernel matrix for the input samples.
        
        Parameters:
        X (np.ndarray): Input feature matrix.
        Y (np.ndarray): (Optional) Another feature matrix for cross-kernel calculation.
        
        Returns:
        np.ndarray: Kernel matrix.
        '''
        if Y is None:
            Y = X
            
        if self.gamma == 'scale':
            self.gamma = 1.0 / (X.shape[1] * X.var()) if X.var() != 0 else 1.0
            
        if self.kernel == 'rbf':
            return np.exp(-self.gamma * cdist(X, Y, 'sqeuclidean'))
        else:
            raise ValueError(f"Unsupported kernel: {self.kernel}")

    def _compute_membership(self, x):
        '''
        Compute fuzzy membership values for a given input sample.
        
        Parameters:
        x (np.ndarray): Input sample.
        
        Returns:
        np.ndarray: Membership values.
        '''
        if self.X_L is None:
            return 0
        
        distances = cdist(x.reshape(1, -1), self.X_L, 'sqeuclidean')
        return np.exp(-self.gamma * distances).flatten()

    def initialize(self, X, y):
        '''
        Initialize the network with the base classes and set the model parameters.
        
        Parameters:
        X (np.ndarray): Feature matrix for the base classes.
        y (np.ndarray): Labels for the base classes.
        
        Returns:
        None
        '''

        self.X_L = X
        unique_classes = np.unique(y)
        self.n_classes = len(unique_classes)

        self.classifiers = {}   

        T = np.zeros((len(y), self.n_classes))
        for i, label in enumerate(y):
            T[i, label] = 1
        self.T_e = T

        for class_number in range(self.n_classes):
            positive_mask = self.T_e[:, class_number] == 1
            positive_indices = np.where(positive_mask)[0]

            negative_mask = ~positive_mask
            negative_indices = np.where(negative_mask)[0]

            if len(negative_indices) > self.max_negative_samples:
                sampled_negative_indices = np.random.choice(negative_indices, self.max_negative_samples, replace=False)
            else:
                sampled_negative_indices = negative_indices

            selected_indices = np.concatenate([positive_indices, sampled_negative_indices])

            X_selected = self.X_L[selected_indices]
            y_selected = np.zeros(len(selected_indices))
            y_selected[:len(positive_indices)] = 1  

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_selected)

            clf = SVC(kernel='rbf', probability=True)
            clf.fit(X_scaled, y_selected)

            self.classifiers[class_number] = {
                "classifier": clf,
                "scaler": scaler
            }


    def add_new_class(self, X_new, y_new, i):
        '''
        Add a new class to the network and update the kernel matrix, labels, 
        neuron activations and the one-vs-rest classifiers.

        Parameters:
        - X_new (np.ndarray): New feature samples for the new class.
        - y_new (int): The label of the new class (e.g., 3 if it’s the 4th class).
        - i (int): Iteration or sample index.

        Returns:
        - None
        '''
        new_class = int(y_new)
        self.class_counts[new_class] = X_new.shape[0]

        self.X_L = np.vstack([self.X_L, X_new])

        Tn = np.zeros((self.X_L.shape[0], new_class + 1))
        Tn[:self.T_e.shape[0], :self.T_e.shape[1]] = self.T_e
        Tn[-X_new.shape[0]:, -1] = 1
        self.T_e = Tn

        self.n_classes += 1

        neuron_activation = np.zeros(len(self.X_L))
        neuron_activation[:] = i
        neuron_activation[:-X_new.shape[0]] = self.neuron_activation
        self.neuron_activation = neuron_activation

        positive_indices = np.where(self.T_e[:, new_class] == 1)[0]
        negative_indices = np.where(self.T_e[:, new_class] == 0)[0]

        if len(negative_indices) > self.max_negative_samples:
            sampled_negative_indices = np.random.choice(negative_indices, self.max_negative_samples, replace=False)
        else:
            sampled_negative_indices = negative_indices

        selected_indices = np.concatenate([positive_indices, sampled_negative_indices])
        X_selected = self.X_L[selected_indices]
        y_selected = np.zeros(len(selected_indices))
        y_selected[:len(positive_indices)] = 1 

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_selected)

        clf = SVC(kernel='rbf', probability=True)
        clf.fit(X_scaled, y_selected)

        self.classifiers[new_class] = {
            "classifier": clf,
            "scaler": scaler
        }

    def _update_G(self, K_new):
        '''
        Update the G matrix for sequential learning.
        
        Parameters:
        K_new (np.ndarray): Kernel matrix for the new sample.
        
        Returns:
        np.ndarray: Updated G matrix.
        '''
        I = np.eye(len(K_new))
        return np.linalg.inv(I/self.C + K_new.T @ K_new)

    def _update_beta(self, K_new, T_new):
        '''
        Update the output weights (beta) for sequential learning.
        
        Parameters:
        K_new (np.ndarray): Kernel matrix for the new sample.
        T_new (np.ndarray): Target matrix for the new sample.
        
        Returns:
        np.ndarray: Updated beta weights.
        '''
        return self.beta + self.G @ K_new.T @ (T_new - K_new @ self.beta)

    def _remove_redundant_neurons(self, i, activation_threshold=100):
        '''
        Remove neurons that have not been activated recently.
        
        Parameters:
        activation_threshold (int): The minimum number of activations required to keep a neuron.
        
        Returns:
        None
        '''
        if len(self.neuron_activation) == 0:
            return
            
        # Identify inactive neurons
        inactive_neurons = np.where( i - self.neuron_activation > activation_threshold)[0]
        
        # Remove neurons representing dominant classes with few occurrences
        minority_threshold = np.average(list(self.class_counts.values()))
        neurons_to_remove = []

        for neuron_idx in inactive_neurons:
            # Find which class the neuron corresponds to
            if self.class_counts[np.argmax(self.beta[neuron_idx])] > minority_threshold:
                neurons_to_remove.append(neuron_idx)
                
        if neurons_to_remove:
            # Remove redundant neurons
            self.X_L = np.delete(self.X_L, neurons_to_remove, axis=0)
            self.beta = np.delete(self.beta, neurons_to_remove, axis=0)
            self.neuron_activation = np.delete(self.neuron_activation, neurons_to_remove)
            self.T_e = np.delete(self.T_e, neurons_to_remove, axis=0)

    def predict(self, X):
# TODO: 
# - check this code 
        for class_number in range(self.n_classes):
            model_data = self.classifiers[class_number]
            scaler = model_data["scaler"]
            clf = model_data["classifier"]

            X_scaled = scaler.transform(X)
            prob = clf.predict_proba(X_scaled)  
            predicted_class = np.argmax(prob, axis=0)[0]
            predicted_probability = np.max(prob)
            if predicted_class == 1:
                return predicted_class, predicted_probability

        return predicted_class, 0



class CoinClassifier:
    '''
    A class for classifying coins using a multi-step approach that involves feature extraction, side classification, and obverse/reverse classification.
    
    Methods:
    - `prepare_reliefF`: Prepare the ReliefF feature extraction model.
    - `prepare_features_batch`: Extract features for a batch of images.
    - `train_side_classifier`: Train a classifier for determining the side (obverse or reverse) of the coin.
    - `train_coin_classifiers`: Train classifiers for identifying obverse or reverse coin details.
    - `predict`: Predict the type of coin and its side.
    - `get_side_confidence`: Retrieve the probability confidence for the side classification.
    '''
    def __init__(self, n_features_to_select=100, name='resnet50'):
        '''
        Initializes the CoinClassifier with pre-trained models and SVM.
        '''

        model_dict = {
        'resnet50': models.resnet50,
        'regnet': models.regnet_y_400mf,
        'efficientnet': models.efficientnet_b0,
        'densenet': models.densenet121,
        }

        try:
            self.feature_model = model_dict[name](pretrained=True)
            self.feature_model = torch.nn.Sequential(*list(self.feature_model.children())[:-1]) 
        except:
            raise ValueError(f"Invalid model name '{name}'. Choose from: {list(model_dict.keys())}")
 
        self.reliefF = FeatureExtractorReliefF(self.feature_model, n_features_to_select=n_features_to_select)
        
        self.side_classifier = SVC(kernel='rbf', probability=True)
        self.side_scaler = StandardScaler()  
        
        self.obverse_classifier = ARCIKELM(C=1.0, kernel='rbf', gamma='scale')
        self.reverse_classifier = ARCIKELM(C=1.0, kernel='rbf', gamma='scale')
        
        self.augmenter = ImageAugmentation()

        self.side_classifier_initialized = False
        self.obverse_classifier_initialized = False
        self.reverse_classifier_initialized = False
    
        self.augment = T.Compose([
            T.RandomAffine(degrees=10, translate=(0.05, 0.05)),
            T.ToTensor(),
            T.RandomErasing(p=0.3, scale=(0.02, 0.08), ratio=(0.5, 1.5), value='random')
            ])
        

    def prepare_reliefF(self, batch_tensors, labels, batch_size, version, path=None):
        '''
        Prepare the ReliefF feature extractor.
        
        Parameters:
        image_batch (list of str): List of paths to images for feature extraction.
        batch_size (int): Number of images to process per batch.
        path (str, optional): Path to load the pre-trained ReliefF model if available.
        
        Returns:
        None
        '''
        if path is None:
            self.reliefF.fit(batch_tensors, labels, version, batch_size)  
        else:
            self.reliefF.load_relief(path) 

    def prepare_features_batch(self, batch_tensors, version):
        '''
        Prepare feature vectors for a batch of images.
        
        Parameters:
        image_paths (list of str): List of image paths to extract features from.
        
        Returns:
        features_list (np.ndarray): Extracted features for each valid image.
        valid_paths (list of str): List of image paths that produced valid features.
        '''
        features = self.reliefF(batch_tensors, version, batch_size=1)
                
        return np.array(features)

    def train_side_classifier(self, paths, labels):
        '''
        Train the side classifier (obverse/reverse) using SVM.
        
        Parameters:
        paths (list of str): Image paths to train the classifier.
        labels (list of dict): Labels for the images, containing the "side" key for classification.
        
        Returns:
        None
        '''
        features = self.prepare_features_batch(paths, 'side')
        
        # Prepare labels: 'revers' -> 1, 'avers' -> 0
        labels = [1 if label == 'revers' else 0 for label in labels[0]['side']]
        
        X = np.vstack(features)
        y = np.array(labels)
        
        X_scaled = self.side_scaler.fit_transform(X)
        
        self.side_classifier.fit(X_scaled, y)
        self.side_classifier_initialized = True

    def save_side_classifier(self, path):
        os.makedirs(path, exist_ok=True) 
        joblib.dump(self.side_classifier, path +'side_classifier.pkl')
        joblib.dump(self.side_scaler, path + 'side_scaler.pkl')

    def load_side_classifier(self, path):
        self.side_classifier = joblib.load(path +  'side_classifier.pkl')
        self.side_scaler = joblib.load(path + 'side_scaler.pkl')

        self.side_classifier_initialized = True 

    def train_coin_classifiers(self, batch_tensors, labels, is_avers=True, features = None):
        '''
        Train either the obverse or reverse classifier for coin classification.
        
        Parameters:
        image_paths (list of str): List of image paths to train on.
        labels (list of dict): Labels for the images.
        is_avers (bool): If True, train the obverse classifier; otherwise train the reverse classifier.
        
        Returns:
        valid_paths (list of str): List of image paths that had valid features.
        '''
        if features is None:
            features = self.prepare_features_batch(batch_tensors, 'class')
        
        # Select classifier based on obverse or reverse
        classifier = self.obverse_classifier if is_avers else self.reverse_classifier
        initialized = self.obverse_classifier_initialized if is_avers else self.reverse_classifier_initialized
        
        if not initialized:
            # Initialize the classifier if not already initialized
            classifier.initialize(features, np.array(labels))
            if is_avers:
                self.obverse_classifier_initialized = True
            else:
                self.reverse_classifier_initialized = True
        else:   
            # Perform sequential learning if the classifier has been initialized
            classifier.sequential_learning(features, np.array(labels))
            
        return 0

    def predict(self, image_path, image, i):
        '''
        Predict the coin type and side (obverse or reverse) for a given image.
        
        Parameters:
        image_path (str): Path to the image to classify.
        
        Returns:
        tuple: A tuple with the side ("obverse", "reverse", or "uncertain") and coin prediction (if available).
        '''
        features = self.reliefF(image_path, 'side')
        if features is None:
            return None, None
            
        features_scaled = self.side_scaler.transform(features.reshape(1, -1))
        
        side_pred = self.side_classifier.predict(features_scaled)[0]        
        
        classifier = self.obverse_classifier if side_pred == 0 else self.reverse_classifier
        features = self.reliefF(image_path, 'class')


        coin_pred, probability = classifier.predict(features.reshape(1, -1))
        if probability < 0.5: 
            is_averse = True if side_pred == 0 else False
            new_tensor_images = self.augmenter.generate_augmented_images(image, 9)
            new_features = self.reliefF(new_tensor_images, 'class')
            self.add_classifier_class(None, is_averse, np.vstack([features, new_features]), i=i)
            coin_pred = classifier.n_classes 

        return "obverse" if side_pred == 0 else "reverse", coin_pred, probability


    def add_classifier_class(self, batch_tensors, is_avers, features=None, i=0):
        if features is None:
            features = self.prepare_features_batch(batch_tensors, 'class')
        
        classifier = self.obverse_classifier if is_avers else self.reverse_classifier
        label = classifier.n_classes
        classifier.add_new_class(features, label, i)
        

    def get_side_confidence(self, image_path):
        '''
        Get the confidence probabilities for side classification (obverse/reverse).
        
        Parameters:
        image_path (str): Path to the image to classify.
        
        Returns:
        dict: Dictionary containing the probabilities for obverse and reverse.
        '''
        features = self.reliefF(image_path, 'side')
        if features is None:
            return None
            
        features_scaled = self.side_scaler.transform(features.reshape(1, -1))
        probabilities = self.side_classifier.predict_proba(features_scaled)[0]
        
        return {
            'obverse': probabilities[0],
            'reverse': probabilities[1]
        }
    
   