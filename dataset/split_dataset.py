import yaml
import random
from collections import defaultdict

def split_data_by_class(input_file, train_file, test_file, test_size=0.2):
    with open(input_file, 'r', encoding='utf-8') as file:
        data = yaml.load(file, Loader=yaml.FullLoader)

    class_groups = defaultdict(list)

    for item in data:
        key = (item['classes'][0]['side'], item['classes'][1]['coin'], item['classes'][2]['denomination'])
        class_groups[key].append(item)

    train_data = []
    test_data = []

    for key, group in class_groups.items():
        test_count = int(len(group) * test_size)  
        random.shuffle(group)  
        test_data.extend(group[:test_count])  
        train_data.extend(group[test_count:]) 

    with open(train_file, 'w', encoding='utf-8') as file:
        yaml.dump(train_data, file, default_flow_style=False, allow_unicode=True)

    with open(test_file, 'w', encoding='utf-8') as file:
        yaml.dump(test_data, file, default_flow_style=False, allow_unicode=True)

    print(f'Podzielono dane: {len(train_data)} próbek treningowych, {len(test_data)} próbek testowych')

if __name__ == "__main__":
    input_file = 'dataset/output.yaml' 
    train_file = 'dataset/train.yaml'   
    test_file = 'dataset/test.yaml'    

    split_data_by_class(input_file, train_file, test_file, test_size=0.2) 
