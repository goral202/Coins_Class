from model import CoinClassifier
from dataloader import CoinDataset
from torch.utils.data import DataLoader
from utils import test_side_classificator
import yaml



with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

root_dataset_path = config["root_dataset_path"]
yaml_file_path = config["yaml_file_path"]

dataset = CoinDataset(yaml_file=yaml_file_path, root_path=root_dataset_path)
classes2label = dataset.take_classes()
dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

model = CoinClassifier()

for image_paths, classes, _ in dataloader:
    side_labels = [1 if label == 'revers' else 0 for label in classes[0]['side']]
    model.prepare_reliefF(image_paths, side_labels, version='side', batch_size=128)
    break

dataloader = DataLoader(dataset, batch_size=1024, shuffle=True)
for image_paths, classes, _ in dataloader:
    model.train_side_classifier(image_paths, classes)
    break

dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
test_side_classificator(model, dataloader)

