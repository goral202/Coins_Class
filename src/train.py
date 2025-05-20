from model_2 import CoinClassifier
from dataloader import CoinDataset
from torch.utils.data import DataLoader
from utils import test_side_classificator
import yaml
import tqdm
from PIL import Image   
from metrics import count_metrics

# TODO: test resnet, regnet, efficientnet, densenet?
feature_model = 'efficientnet'       # 'resnet50', 'regnet', 'efficientnet', 'densenet'
side_test = False
train_from_zero = True

features_numbers = [100, 256, 512, 1024, 2048]
stages = ['stage_1', 'stage_2', 'stage_3']

# features_numbers = [512, 1024, 2048]
# stages = ['stage_3']

for stage in stages:
    for features_number in features_numbers:
        test_name = f'model_2____start_50/{feature_model}/{stage}/{features_number}'


        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)

        root_dataset_path = config["root_dataset_path"]
        train_yaml_file_path = config["train_yaml_file_path"]
        train_config = config["train_config"]

        model = CoinClassifier(features_number, feature_model)

        if train_from_zero:
            dataset_init= CoinDataset(yaml_file=train_yaml_file_path, root_path=root_dataset_path, config_file=train_config, stage= "init")

            classes2label = dataset_init.take_classes()
            dataloader = DataLoader(dataset_init, batch_size=256, shuffle=True)

            print("Preaparing reliefF ...")
            for image_batch, classes, labels, _ in dataloader:
                side_labels = [1 if label == 'revers' else 0 for label in classes[0]['side']]
                model.prepare_reliefF(image_batch, side_labels, version='side', batch_size=256)
                model.prepare_reliefF(image_batch, labels.cpu().tolist(), version='class', batch_size=256)
                break

            model.reliefF.save_relief(path='models',version='side')
            model.reliefF.save_relief(path='models',version='class')

            print("Training side classifier ...")
            dataloader = DataLoader(dataset_init, batch_size=1024, shuffle=True)
            for image_batch, classes, _, _ in dataloader:
                model.train_side_classifier(image_batch, classes)
                break

            model.save_side_classifier('models/')
        else:
            print("Loading  ...")
            model.load_side_classifier('models/')
            model.reliefF.load_relief(path='models',version='side')
            model.reliefF.load_relief(path='models',version='class')

        if side_test:
            print("Testing side classificator ...")
            dataset= CoinDataset(yaml_file=test_yaml_file_path, root_path=root_dataset_path, config_file=train_config, stage= "stage_3")
            dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
            test_side_classificator(model, dataloader, output_dir=f'outputs/{test_name}')


        dataset_init_avers = CoinDataset(yaml_file=train_yaml_file_path, root_path=root_dataset_path, config_file=train_config, stage= "init", side="avers")
        dataset_init_revers = CoinDataset(yaml_file=train_yaml_file_path, root_path=root_dataset_path, config_file=train_config, stage= "init", side="revers")

        dataloader_avers = DataLoader(dataset_init_avers, batch_size=50, shuffle=True)
        dataloader_revers = DataLoader(dataset_init_revers, batch_size=50, shuffle=True)
        print("Training avers class classifier ...")
        for image_batch, classes, labels, _ in dataloader_avers:
            model.train_coin_classifiers(image_batch, labels, is_avers=True)
            break
        print("Training revers class classifier ...")
        for image_batch, classes, labels, _ in dataloader_revers:
            model.train_coin_classifiers(image_batch, labels, is_avers=False)
            break


        test_yaml_file_path = config["test_yaml_file_path"]

        dataset_init = CoinDataset(yaml_file=test_yaml_file_path, root_path=root_dataset_path, config_file=train_config, stage= stage)

        dataloader_init = DataLoader(dataset_init, batch_size=1, shuffle=True)
        print("Predicting ...")
        positive = 0
        negative = 0
        dict_classes = {}
        dict_classes["obverse"] = {}
        dict_classes["reverse"] = {}
        for i, (image_batch, classes, labels, image_path) in enumerate(tqdm.tqdm(dataloader_init, desc="Predicting")):
            img = Image.open(image_path[0]).convert('RGB')
            predicted_side, predicted_class, probability = model.predict(image_batch, img, i + 1)
            try:
                dict_classes[predicted_side][predicted_class].append(labels)
            except:
                dict_classes[predicted_side][predicted_class] = [labels]

            # print('class number ', 'Avers: ', model.obverse_classifier.n_classes, 'Reverse: ', model.reverse_classifier.n_classes)

        count_metrics(dict_classes, output_dir=f'outputs/{test_name}')  

