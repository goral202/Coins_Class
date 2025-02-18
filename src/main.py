from model import CoinClassifier
import os

def prepare_coin_dataset(base_dir):
    '''
    Prepare the data structure for the coin dataset.
    
    Parameters:
    base_dir (str): The base directory where coin images are stored. 
                    It should contain subdirectories 'obverse' and 'reverse'.
    
    Returns:
    dict: A dictionary containing paths to images categorized by coin side 
          ('obverse' and 'reverse') and coin type.
    '''
    data = {
        'obverse': {},
        'reverse': {}
    }
    
    for side in ['obverse', 'reverse']:
        side_dir = os.path.join(base_dir, side)
        if not os.path.exists(side_dir):
            continue
            
        for coin_type in os.listdir(side_dir):
            coin_dir = os.path.join(side_dir, coin_type)
            if not os.path.isdir(coin_dir):
                continue
                
            data[side][coin_type] = []
            for img_name in os.listdir(coin_dir):
                if img_name.endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(coin_dir, img_name)
                    data[side][coin_type].append(img_path)
                    
    return data

def main():
    '''
    Main function to train classifiers and predict coin type and side.
    
    This function performs the following steps:
    1. Prepares the dataset using 'prepare_coin_dataset'.
    2. Trains the side classifier to distinguish between obverse and reverse sides.
    3. Trains individual coin classifiers for both obverse and reverse sides.
    4. Predicts the side and type of a test coin image.
    
    Returns:
    None
    '''
    classifier = CoinClassifier()
    
    base_dir = 'coins_dataset'
    data = prepare_coin_dataset(base_dir)
    
    obverse_paths = [img for coin_type in data['obverse'].values() 
                    for img in coin_type]
    reverse_paths = [img for coin_type in data['reverse'].values() 
                    for img in coin_type]
    
    valid_obverse, valid_reverse = classifier.train_side_classifier(
        obverse_paths, reverse_paths)
    print("Side classifier trained")
    
    coin_types = list(data['obverse'].keys())
    type_to_label = {coin_type: i for i, coin_type in enumerate(coin_types)}
    
    obverse_labels = []
    obverse_paths = []
    for coin_type, paths in data['obverse'].items():
        obverse_paths.extend(paths)
        obverse_labels.extend([type_to_label[coin_type]] * len(paths))
    
    valid_obverse = classifier.train_coin_classifiers(
        obverse_paths, obverse_labels, is_obverse=True)
    print("Obverse classifier trained")
    
    reverse_labels = []
    reverse_paths = []
    for coin_type, paths in data['reverse'].items():
        reverse_paths.extend(paths)
        reverse_labels.extend([type_to_label[coin_type]] * len(paths))
    
    valid_reverse = classifier.train_coin_classifiers(
        reverse_paths, reverse_labels, is_obverse=False)
    print("Reverse classifier trained")
    
    test_image = 'coins_dataset/test/test_coin.jpg'
    side, coin_type = classifier.predict(test_image)
    
    if side and coin_type is not None:
        print(f"Image: {test_image}")
        print(f"Predicted side: {side}")
        print(f"Predicted coin type: {coin_types[coin_type]}")

if __name__ == "__main__":
    main()
