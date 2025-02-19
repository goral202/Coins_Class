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
    def __init__(self, yaml_file, root_path):
        """
        Args:
            root_path (str): Root path to the dataset dir
            yaml_file (str): Path to the YAML file containing dataset information.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_path = root_path
        self.class_dict = None

        with open(yaml_file, 'r') as file:
            self.data = yaml.safe_load(file)

    def take_classes(self):
        unique_classes = set()

        for entry in self.data:
            side = None
            denomination = None

            side = entry["classes"][0]["side"]
            denomination = entry["classes"][2]["denomination"]

            if side and denomination:
                class_label = f"{side}_{denomination}"
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

        classes = item['classes']
        denomination = self.class_dict[f'{classes[0]["side"]}_{classes[2]["denomination"]}']
        return image_path, classes, denomination

# Example usage
if __name__ == "__main__":
    from torchvision import transforms
    root_dataset_path = "C:\\Users\\jakub\\Desktop\\PULP\\STUDIA\\Praca mgr\\DATASET_COINS"
    yaml_file_path = "dataset/test.yaml"

    dataset = CoinDataset(yaml_file=yaml_file_path, root_path=root_dataset_path)

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    for image_paths, classes in dataloader:
        print("Batch of image paths:", image_paths)
        print("Batch of classes:", classes)
