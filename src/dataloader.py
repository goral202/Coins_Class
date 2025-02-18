import yaml
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image

classes_to_init = []


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
    def __init__(self, yaml_file, transform=None):
        """
        Args:
            yaml_file (str): Path to the YAML file containing dataset information.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        with open(yaml_file, 'r') as file:
            self.data = yaml.safe_load(file)

        self.transform = transform

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

        image_path = item['image']

        classes = item['classes']

        return image_path, classes

# Example usage
if __name__ == "__main__":
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    yaml_file_path = "dataset/test.yaml"

    dataset = CoinDataset(yaml_file=yaml_file_path, transform=transform)

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    for image_paths, classes in dataloader:
        print("Batch of image paths:", image_paths)
        print("Batch of classes:", classes)
