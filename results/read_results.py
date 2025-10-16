import pickle

# Path to your results
result_path = "../output/pointpillar_custom/default/eval/eval_with_train/epoch_80/val/result.pkl"

with open(result_path, "rb") as f:
    results = pickle.load(f)

print(f"Total frames: {len(results)}")

# Inspect first frame
sample = results[0]
print("Keys:", sample.keys())
print("Boxes:", sample['boxes_lidar'])
print("Scores:", sample['score'])        # <-- singular
print("Labels:", sample['pred_labels'])

# Count how many frames actually have predictions
num_with_preds = sum(len(x['boxes_lidar']) > 0 for x in results)
print(f"Frames with predictions: {num_with_preds}/{len(results)}")
