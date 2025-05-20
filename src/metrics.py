import os
from collections import Counter, defaultdict
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def count_metrics(dict_classes, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)

    txt_path = os.path.join(output_dir, "metrics.txt")
    with open(txt_path, "w", encoding="utf-8") as f:

        correct = 0
        total = 0
        cluster_stats = {}

        for side in dict_classes:
            for cluster, labels in dict_classes[side].items():
                label_list = [int(label.item()) for label in labels]
                label_counts = Counter(label_list)
                dominant_label, dominant_count = label_counts.most_common(1)[0]
                correct += dominant_count
                total += len(labels)

        for side in dict_classes:
            for cluster_id, labels in dict_classes[side].items():
                label_list = [int(label.item()) for label in labels]
                label_counts = Counter(label_list)
                dominant_class, dominant_count = label_counts.most_common(1)[0]
                total_count = len(label_list)
                accuracy = dominant_count / total_count

                cluster_stats[(side, cluster_id)] = {
                    "dominant_class": dominant_class,
                    "dominant_count": dominant_count,
                    "total_count": total_count,
                    "accuracy": accuracy
                }

        for (side, cluster_id), stats in cluster_stats.items():
            f.write(f"[{side}][Cluster {cluster_id}] ➜ Klasa: {stats['dominant_class']}, "
                    f"Skuteczność: {stats['accuracy']:.2%} ({stats['dominant_count']}/{stats['total_count']})\n")

        accuracy = correct / total
        f.write(f"\nSkuteczność metody: {accuracy:.2%}\n")

        side_stats = {}
        for (side, cluster_id), stats in cluster_stats.items():
            if side not in side_stats:
                side_stats[side] = {"correct": 0, "total": 0}
            side_stats[side]["correct"] += stats["dominant_count"]
            side_stats[side]["total"] += stats["total_count"]

        for side, counts in side_stats.items():
            accuracy = counts["correct"] / counts["total"]
            f.write(f"Skuteczność dla strony '{side}': {accuracy:.2%} ({counts['correct']}/{counts['total']})\n")

        total_correct = sum(c["correct"] for c in side_stats.values())
        total_examples = sum(c["total"] for c in side_stats.values())
        global_accuracy = total_correct / total_examples
        f.write(f"\n🔍 Globalna skuteczność: {global_accuracy:.2%} ({total_correct}/{total_examples})\n")

    confusion_matrices = {}

    for side in dict_classes:
        cm = defaultdict(lambda: defaultdict(int))

        for cluster_id, labels in dict_classes[side].items():
            dominant_class = cluster_stats[(side, cluster_id)]["dominant_class"]
            for label in labels:
                true_label = int(label.item())
                cm[true_label][dominant_class] += 1

        all_classes = sorted({cls for row in cm.values() for cls in row.keys()} | set(cm.keys()))
        matrix = pd.DataFrame(index=all_classes, columns=all_classes).fillna(0)

        for true_cls in cm:
            for pred_cls in cm[true_cls]:
                matrix.at[true_cls, pred_cls] = cm[true_cls][pred_cls]

        confusion_matrices[side] = matrix

        plt.figure(figsize=(8, 6))
        sns.heatmap(matrix.astype(int), annot=True, fmt='d', cmap="Blues", cbar=False)
        plt.title(f"Confusion Matrix - {side}")
        plt.xlabel("Predykowana klasa")
        plt.ylabel("Rzeczywista klasa")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"confusion_matrix_{side}.png"))
        plt.close()
