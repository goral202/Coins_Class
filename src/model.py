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
        self.selected_features_idx = None

        self.model = model
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def extract_features(self, image_paths, batch_size=32):
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
            for i in range(0, len(image_paths), batch_size):
                batch_paths = image_paths[i:i + batch_size]
                batch_tensors = []
                
                for img_path in batch_paths:
                    img = Image.open(img_path).convert('RGB')
                    img_tensor = self.transform(img)
                    batch_tensors.append(img_tensor)
                
                batch = torch.stack(batch_tensors).to(self.device)
                batch_features = self.model(batch)
                features.extend(batch_features.squeeze().cpu().numpy())
        
        return np.array(features)

    def fit(self, image_paths, labels, batch_size=32):
        '''
        Train ReliefF feature selection model.
        
        Parameters:
        image_paths (list): List of image paths.
        labels (list): Corresponding labels for the images.
        batch_size (int): The batch size to process images.
        
        Returns:
        self: The trained FeatureExtractorReliefF object.
        '''
        features = self.extract_features(image_paths, batch_size)
        
        self.relief = ReliefF(n_features_to_select=self.n_features_to_select, n_neighbors=10)
        self.relief.fit(features, labels)
        
        self.selected_features_idx = self.relief.top_features_[:self.n_features_to_select]
        
        return self

    def __call__(self, image_paths, batch_size=32):
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
        
        features = self.extract_features(image_paths, batch_size)
        
        selected_features = features[:, self.selected_features_idx]
        
        return selected_features

    def save_relief(self, path):
        '''
        Save the trained ReliefF model to a file.
        
        Parameters:
        path (str): Path to save the trained model.
        
        Returns:
        None
        '''
        if self.relief is None:
            raise ValueError("Model is not trained. Use the fit() method first.")
        
        with open(path, 'wb') as f:
            pickle.dump({
                'relief': self.relief,
                'selected_features_idx': self.selected_features_idx
            }, f)

    def load_relief(self, path):
        '''
        Load a trained ReliefF model from a file.
        
        Parameters:
        path (str): Path to the trained model file.
        
        Returns:
        None
        '''
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.relief = data['relief']
            self.selected_features_idx = data['selected_features_idx']


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
    def __init__(self, C=1.0, kernel='rbf', gamma='scale'):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.X_L = None  # Kernel matrix vectors
        self.beta = None  # Output weights
        self.n_classes = 0
        self.neuron_activation = None  # Neuron eligibility matrix
        self.class_counts = {}  # Track samples per class

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
        unique_classes = np.unique(y)
        self.n_classes = len(unique_classes)
        
        # Set the initial set of samples for the kernel matrix
        self.X_L = X

        # Compute the kernel matrix
        K = self._compute_kernel(X, self.X_L)
        
        # Prepare target matrix with one-hot encoding
        T = np.zeros((len(y), self.n_classes))
        for i, label in enumerate(y):
            T[i, label] = 1
            
        # Calculate the initial output weights (beta)
        I = np.eye(len(K))
        self.G = np.linalg.inv(I/self.C + K.T @ K)
        self.beta = self.G @ K.T @ T
        
        # Initialize the neuron activation matrix
        self.neuron_activation = np.zeros(len(self.X_L))
        
        # Track the count of samples per class
        for label in y:
            self.class_counts[label[0]] = self.class_counts.get(label[0], 0) + 1

    def add_new_class(self, X_new, y_new):
        '''
        Add a new class to the network and update the kernel matrix and weights.
        
        Parameters:
        X_new (np.ndarray): New feature samples for the new class.
        y_new (np.ndarray): New labels for the new class.
        
        Returns:
        None
        '''
        new_class = np.max(y_new)
        self.class_counts[new_class] = len(y_new)
        
        # Add a transformation matrix to account for the new class
        M = np.zeros((self.beta.shape[1], self.beta.shape[1] + 1))
        np.fill_diagonal(M, 1)
        self.beta = self.beta @ M
        
        # Add new hidden neurons (using a subset of new samples)
        n_new_samples = int(0.1 * len(X_new))  # 10% of new samples
        indices = np.random.choice(len(X_new), n_new_samples, replace=False)
        self.X_L = np.vstack([self.X_L, X_new])

        # Update kernel matrix and output weights (beta)
        K_new = self._compute_kernel(X_new, self.X_L)
        
        # Prepare target matrix for the new class
        T_new = np.zeros((len(y_new), self.beta.shape[1]))
        T_new[:, -1] = 1
        
        # Update G matrix and beta using the new kernel
        K_total = self._compute_kernel(np.vstack([X_new]), self.X_L)
        I = np.eye(len(K_total))
        self.G = np.linalg.inv(I/self.C + K_total.T @ K_total)
        self.beta = self.G @ K_total.T @ np.vstack([T_new])
        
        # Update neuron activation matrix and increment the number of classes
        self.neuron_activation = np.zeros(len(self.X_L))
        self.n_classes += 1

    def sequential_learning(self, X, y, threshold=0.1):
        '''
        Perform sequential learning by adding new neurons as needed for unseen samples.
        
        Parameters:
        X (np.ndarray): New feature samples.
        y (np.ndarray): New labels for the samples.
        threshold (float): Membership threshold to decide when to add a new neuron.
        
        Returns:
        None
        '''
        for i, (x_i, y_i) in enumerate(zip(X, y)):
            # Compute membership values for the current sample
            membership = self._compute_membership(x_i)
            max_membership = np.max(membership)
            
            if max_membership < threshold:
                # Add a new hidden neuron if membership is below the threshold
                self.X_L = np.vstack([self.X_L, x_i])
                
                # Update kernel matrix and output weights (beta)
                K_new = self._compute_kernel(x_i.reshape(1, -1), self.X_L)
                
                # Prepare target matrix for the new sample
                T_new = np.zeros((1, self.n_classes))
                T_new[0, y_i] = 1
                
                # Update the G matrix and beta using sequential learning formulas
                K_total = self._compute_kernel(np.vstack([x_i]), self.X_L)
                self.G = self._update_G(K_total)
                self.beta = self._update_beta(K_total, T_new)
                
                # Update neuron activation matrix
                self.neuron_activation = np.append(self.neuron_activation, i)
            else:
                # Update existing neurons without adding a new one
                K_i = self._compute_kernel(x_i.reshape(1, -1), self.X_L)
                T_i = np.zeros((1, self.n_classes))
                T_i[0, y_i] = 1
                
                # Adjust beta with the new sample
                self.beta = self.beta + self.G @ K_i.T @ (T_i - K_i @ self.beta)
            
            # Update class counts
            self.class_counts[y_i] = self.class_counts.get(y_i, 0) + 1
            
            # Check for and remove redundant neurons
            self._remove_redundant_neurons()

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

    def _remove_redundant_neurons(self, activation_threshold=100):
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
        inactive_neurons = np.where(self.neuron_activation < activation_threshold)[0]
        
        # Remove neurons representing dominant classes with few occurrences
        minority_threshold = np.median(list(self.class_counts.values()))
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

    def predict(self, X):
        '''
        Predict the class labels for the input samples.
        
        Parameters:
        X (np.ndarray): Input feature matrix.
        
        Returns:
        np.ndarray: Predicted class labels for each sample.
        '''
        K = self._compute_kernel(X, self.X_L)
        predictions = K @ self.beta
        print(predictions)
        return np.argmax(predictions, axis=1)
    

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
    def __init__(self):
        '''
        Initializes the CoinClassifier with pre-trained models and SVM.
        '''
        self.feature_model = models.resnet50(pretrained=True)
        self.feature_model = torch.nn.Sequential(*list(self.feature_model.children())[:-1]) 
        
        self.reliefF = FeatureExtractorReliefF(self.feature_model)
        
        self.side_classifier = SVC(kernel='rbf', probability=True)
        self.side_scaler = StandardScaler()  
        
        self.obverse_classifier = ARCIKELM(C=1.0, kernel='rbf', gamma='scale')
        self.reverse_classifier = ARCIKELM(C=1.0, kernel='rbf', gamma='scale')
        
        self.side_classifier_initialized = False
        self.obverse_classifier_initialized = False
        self.reverse_classifier_initialized = False
    
    def prepare_reliefF(self, image_paths, batch_size, path=None):
        '''
        Prepare the ReliefF feature extractor.
        
        Parameters:
        image_paths (list of str): List of paths to images for feature extraction.
        batch_size (int): Number of images to process per batch.
        path (str, optional): Path to load the pre-trained ReliefF model if available.
        
        Returns:
        None
        '''
        if path is None:
            self.reliefF.fit(image_paths, batch_size)  
        else:
            self.reliefF.load_relief(path) 

    def prepare_features_batch(self, image_paths):
        '''
        Prepare feature vectors for a batch of images.
        
        Parameters:
        image_paths (list of str): List of image paths to extract features from.
        
        Returns:
        features_list (np.ndarray): Extracted features for each valid image.
        valid_paths (list of str): List of image paths that produced valid features.
        '''
        features_list = []
        valid_paths = []
        
        for img_path in image_paths:
            features = self.reliefF(img_path)
            if features is not None:
                features_list.append(features)
                valid_paths.append(img_path)
                
        return np.array(features_list), valid_paths

    def train_side_classifier(self, paths, labels):
        '''
        Train the side classifier (obverse/reverse) using SVM.
        
        Parameters:
        paths (list of str): Image paths to train the classifier.
        labels (list of dict): Labels for the images, containing the "side" key for classification.
        
        Returns:
        None
        '''
        features, _ = self.prepare_features_batch(paths)
        
        # Prepare labels: 'revers' -> 1, 'obverse' -> 0
        labels = labels[0]['side']
        labels = [1 if label == 'revers' else 0 for label in labels]
        
        X = np.vstack(features)
        y = np.array(labels)
        
        X_scaled = self.side_scaler.fit_transform(X)
        
        self.side_classifier.fit(X_scaled, y)
        self.side_classifier_initialized = True

    def train_coin_classifiers(self, image_paths, labels, is_obverse=True):
        '''
        Train either the obverse or reverse classifier for coin classification.
        
        Parameters:
        image_paths (list of str): List of image paths to train on.
        labels (list of dict): Labels for the images.
        is_obverse (bool): If True, train the obverse classifier; otherwise train the reverse classifier.
        
        Returns:
        valid_paths (list of str): List of image paths that had valid features.
        '''
        features, valid_paths = self.prepare_features_batch(image_paths)
        
        # Get valid labels for training
        valid_labels = [labels[i] for i, path in enumerate(image_paths) if path in valid_paths]
        
        # Select classifier based on obverse or reverse
        classifier = self.obverse_classifier if is_obverse else self.reverse_classifier
        initialized = self.obverse_classifier_initialized if is_obverse else self.reverse_classifier_initialized
        
        if not initialized:
            # Initialize the classifier if not already initialized
            classifier.initialize(features, np.array(valid_labels))
            if is_obverse:
                self.obverse_classifier_initialized = True
            else:
                self.reverse_classifier_initialized = True
        else:
            # Perform sequential learning if the classifier has been initialized
            classifier.sequential_learning(features, np.array(valid_labels))
            
        return valid_paths

    def predict(self, image_path):
        '''
        Predict the coin type and side (obverse or reverse) for a given image.
        
        Parameters:
        image_path (str): Path to the image to classify.
        
        Returns:
        tuple: A tuple with the side ("obverse", "reverse", or "uncertain") and coin prediction (if available).
        '''
        features = self.reliefF(image_path)
        if features is None:
            return None, None
            
        # Scale features for side classification
        features_scaled = self.side_scaler.transform(features.reshape(1, -1))
        
        # Predict side (obverse or reverse)
        side_pred = self.side_classifier.predict(features_scaled)[0]
        side_proba = self.side_classifier.predict_proba(features_scaled)[0]
        
        confidence_threshold = 0.7
        if max(side_proba) < confidence_threshold:
            return "uncertain", None  # Return uncertain if confidence is too low
        
        # Choose the appropriate classifier based on the side prediction
        classifier = self.obverse_classifier if side_pred == 0 else self.reverse_classifier
        coin_pred = classifier.predict(features.reshape(1, -1))[0]
        
        return "obverse" if side_pred == 0 else "reverse", coin_pred

    def get_side_confidence(self, image_path):
        '''
        Get the confidence probabilities for side classification (obverse/reverse).
        
        Parameters:
        image_path (str): Path to the image to classify.
        
        Returns:
        dict: Dictionary containing the probabilities for obverse and reverse.
        '''
        features = self.reliefF(image_path)
        if features is None:
            return None
            
        features_scaled = self.side_scaler.transform(features.reshape(1, -1))
        probabilities = self.side_classifier.predict_proba(features_scaled)[0]
        
        return {
            'obverse': probabilities[0],
            'reverse': probabilities[1]
        }