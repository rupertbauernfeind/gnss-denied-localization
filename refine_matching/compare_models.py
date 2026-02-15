"""Compare accuracy of different matchers."""
import csv
import math
from pathlib import Path

def load_csv(path):
    """Load CSV into dict: id -> (x_pixel, y_pixel)."""
    data = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            data[int(row["id"])] = (float(row["x_pixel"]), float(row["y_pixel"]))
    return data

def evaluate(pred_file, matcher_name):
    """Evaluate predictions against ground truth."""
    root = Path(__file__).parent.parent
    gt = load_csv(root / "data" / "train_data" / "train_pos.csv")
    pred = load_csv(pred_file)
    
    common_ids = sorted(set(gt) & set(pred))
    
    distances = []
    for i in common_ids:
        dx = pred[i][0] - gt[i][0]
        dy = pred[i][1] - gt[i][1]
        distances.append(math.sqrt(dx * dx + dy * dy))
    
    thresholds = [(25, "5m"), (125, "25m"), (500, "100m")]
    accuracies = []
    for px_thresh, label in thresholds:
        acc = sum(1 for d in distances if d <= px_thresh) / len(distances) * 100
        accuracies.append(acc)
    
    score = sum(accuracies) / len(accuracies)
    
    distances.sort()
    median = distances[len(distances)//2]
    mean = sum(distances)/len(distances)
    
    return {
        'name': matcher_name,
        'score': score,
        'acc_5m': accuracies[0],
        'acc_25m': accuracies[1],
        'acc_100m': accuracies[2],
        'median': median,
        'mean': mean,
        'n_samples': len(common_ids)
    }

# Evaluate all matchers
matchers = [
    ('train_predictions_rdd-star.csv', 'RDD-Star'),
    ('train_predictions_edm.csv', 'EDM'),
    ('train_predictions_affine-steerers.csv', 'Affine-Steerers'),
]

print("="*80)
print("MATCHER COMPARISON")
print("="*80)

results = []
for pred_file, name in matchers:
    result = evaluate(Path(__file__).parent / pred_file, name)
    results.append(result)

# Print results
print(f"\n{'Matcher':<20} {'Score':>8} {'@5m':>8} {'@25m':>8} {'@100m':>8} {'Median':>8} {'Mean':>8}")
print("-"*80)
for r in results:
    print(f"{r['name']:<20} {r['score']:>8.2f} {r['acc_5m']:>7.2f}% {r['acc_25m']:>7.2f}% {r['acc_100m']:>7.2f}% {r['median']:>7.1f}px {r['mean']:>7.1f}px")

# Find best
best = max(results, key=lambda x: x['score'])
print("\n" + "="*80)
print(f"BEST MATCHER: {best['name']} with score {best['score']:.2f}")
print("="*80)
