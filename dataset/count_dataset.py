import yaml
from collections import Counter
import matplotlib.pyplot as plt

with open("dataset/output.yaml", "r", encoding="utf-8") as file:
    data = yaml.safe_load(file)

side_counter = Counter()
coin_counter = Counter()
denomination_counter = Counter()

for item in data:
    for cls in item['classes']:
        if 'side' in cls:
            side_counter[cls['side']] += 1
        if 'coin' in cls:
            coin_counter[cls['coin']] += 1
        if 'denomination' in cls:
            denomination_counter[cls['denomination']] += 1

def plot_counter(counter, title, output_file):
    labels, values = zip(*counter.items())
    plt.figure(figsize=(8, 4))
    plt.bar(labels, values, color='skyblue')
    plt.title(title)
    plt.xlabel('Klasa')
    plt.ylabel('Liczba wystąpień')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.show()


plot_counter(side_counter, "Liczebność klas - side", 'dataset/side.png')
print(side_counter)
plot_counter(coin_counter, "Liczebność klas - coin", 'dataset/coin.png')
print(coin_counter)
plot_counter(denomination_counter, "Liczebność klas - denomination", 'dataset/denomination.png')
print(denomination_counter)