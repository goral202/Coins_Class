import os
import yaml

def generate_yaml(dataset_path, output_file):
    data = []
    
    for side in ['avers', 'revers']:
        side_path = os.path.join(dataset_path, side)
        save_path_side = os.path.join(side)

        if not os.path.exists(side_path):
            continue
        
        for i, coin in enumerate(os.listdir(side_path)):
            print(i, ' / ', len(os.listdir(side_path)))
            coin_path = os.path.join(side_path, coin)
            save_path_coin = os.path.join(save_path_side, coin)

            if not os.path.isdir(coin_path):
                continue
            
            for nomination in os.listdir(coin_path):
                nomination_path = os.path.join(coin_path, nomination)
                save_path_nomination = os.path.join(save_path_coin, nomination)

                for image_name in os.listdir(nomination_path):
                    image_path = os.path.join(coin_path, image_name)
                    save_path_image_name = os.path.join(save_path_nomination, image_name)

                    if image_name.endswith('.jpg') or image_name.endswith('.png'):
                        data.append({
                            'image': save_path_image_name,
                            'classes': [
                                {'side': side},
                                {'coin': coin},
                                {'denomination': nomination} 
                            ]
                        })

    with open(output_file, 'w') as file:
        yaml.dump(data, file, default_flow_style=False, allow_unicode=True)


if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    dataset_path = config["root_dataset_path"]
    output_file = 'output.yaml'

    generate_yaml(dataset_path, output_file)
