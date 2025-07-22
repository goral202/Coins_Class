import yaml
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import cv2


def apply_sobel_opencv(pil_img):
    img_gray = np.array(pil_img.convert('L'))
    
    sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_combined = cv2.magnitude(sobelx, sobely)

    sobel_uint8 = cv2.convertScaleAbs(sobel_combined) 
    sobel_rgb = cv2.cvtColor(sobel_uint8, cv2.COLOR_GRAY2RGB) 
    return Image.fromarray(sobel_rgb)


class CoinDataset(Dataset):
    '''
    Custom dataset for loading coin images and their class labels from a YAML file.
    
    Parameters:
    yaml_file (str): Path to the YAML file containing dataset information. 
                     The YAML file should include image paths and associated class labels.
    transform (callable, optional): Optional transformation to apply to the images (e.g., resizing, normalization).
    
    Returns:
    None
    '''
    def __init__(self, yaml_file, root_path, config_file, stage, side=None):
        """
        Args:
            root_path (str): Root path to the dataset dir
            yaml_file (str): Path to the YAML file containing dataset information.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_path = root_path
        self.class_dict = None
        self.side = side

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)

        if stage not in config["stages"]:
            raise ValueError(f"Stage '{stage}' not found in config file.")
        
        self.classes_list = config["stages"][stage]["denominations"]
        with open(yaml_file, 'r') as file:
            self.data = yaml.safe_load(file)
        
        self.data = self.filter_classes()
        if self.side:
            self.data = self.filter_side()
        _ = self.take_classes()

    def filter_classes(self):
        return [ item for item in self.data if item['classes'][2]['denomination'] in self.classes_list ]    
    
    def filter_side(self):
        return [ item for item in self.data if item['classes'][0]['side'] == self.side ]    

    def take_classes(self):
        unique_classes = set()

        for entry in self.data:
            side = None
            denomination = None

            side = entry["classes"][0]["side"]
            denomination = entry["classes"][2]["denomination"]

            if side and denomination:
                class_label = f"{denomination}"
                unique_classes.add(class_label)

        self.class_dict = {class_label: idx for idx, class_label in enumerate(sorted(unique_classes))}

        return self.class_dict
    

    def __len__(self):
        '''
        Returns the total number of samples in the dataset.
        
        Returns:
        int: The number of samples in the dataset.
        '''
        return len(self.data)

    def __getitem__(self, idx):
        '''
        Retrieve a sample from the dataset based on the index.
        
        Parameters:
        idx (int or slice): Index or slice of the sample(s) to retrieve.
        
        Returns:
        tuple: A tuple containing:
            - image_path (str): Path to the image.
            - classes (list): List of class labels for the image.
        '''
        if isinstance(idx, slice):
            idx = range(*idx.indices(len(self)))
            return [self.__getitem__(i) for i in idx]

        item = self.data[idx]

        image_path = os.path.join(self.root_path, item['image'])

        img = Image.open(image_path).convert('RGB')

        # img = Image.open(image_path)
        # sobel_img = apply_sobel_opencv(img)
        # img_tensor = self.transform(sobel_img)

        img_tensor = self.transform(img)

        classes = item['classes']
        denomination = self.class_dict[f'{classes[2]["denomination"]}']
        return img_tensor, classes, denomination, image_path

if __name__ == "__main__":
    
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    root_dataset_path = config["root_dataset_path"]
    yaml_file_path = config["yaml_file_path"]

    dataset = CoinDataset(yaml_file=yaml_file_path, root_path=root_dataset_path)

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    for image_paths, classes in dataloader:
        print("Batch of image paths:", image_paths)
        print("Batch of classes:", classes)
