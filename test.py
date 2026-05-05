#!/usr/bin/env python3
# generate_yolo_comparison.py
# Run in your project root (where runs/ and dataset YAML live)

import os
from pathlib import Path
import traceback

output_dir = Path("comparison_results")
output_dir.mkdir(parents=True, exist_ok=True)

# <-- EDIT THESE PATHS if different -->
v8_path = Path("runs/detect/train2_v8/weights/best.pt")
v9_path = Path("runs/detect/train/weights/best.pt")
# The script will auto-find a dataset YAML under the repo if present:
yaml_candidates = list(Path(".").rglob("*.yaml"))
yaml_path = yaml_candidates[0] if yaml_candidates else None

print("YOLOv8 path:", v8_path)
print("YOLOv9 path:", v9_path)
print("Found YAML:", yaml_path)
print("Output dir:", output_dir)

# Install requirements if needed:
# pip install ultralytics matplotlib pandas pillow

try:
    from ultralytics import YOLO
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    from PIL import Image, ImageOps, ImageDraw, ImageFont
    import pandas as pd
except Exception as e:
    print("Missing packages. Install with: pip install ultralytics matplotlib pandas pillow")
    raise

def try_eval_model(model_path, yaml_path):
    try:
        model = YOLO(str(model_path))
        m = None
        if yaml_path:
            print(f"Running val for {model_path} with {yaml_path} ...")
            r = model.val(data=str(yaml_path), split="test")
            if hasattr(r, "results_dict"):
                rd = r.results_dict
                m = {
                    "precision": rd.get("metrics/precision(B)"),
                    "recall": rd.get("metrics/recall(B)"),
                    "mAP50": rd.get("metrics/mAP50(B)"),
                    "mAP50-95": rd.get("metrics/mAP50-95(B)")
                }
        return m, model
    except Exception as e:
        print(f"Evaluation failed for {model_path}: {e}")
        traceback.print_exc()
        return None, None

metrics = {}
models = {}

if v8_path.exists():
    m8, model8 = try_eval_model(v8_path, yaml_path)
    metrics["YOLOv8"] = m8
    models["YOLOv8"] = model8
else:
    print("YOLOv8 weights not found at", v8_path)

if v9_path.exists():
    m9, model9 = try_eval_model(v9_path, yaml_path)
    metrics["YOLOv9"] = m9
    models["YOLOv9"] = model9
else:
    print("YOLOv9 weights not found at", v9_path)

# Save metrics if present
rows = []
for k, v in metrics.items():
    if v:
        rows.append({"Model": k,
                     "Precision": v.get("precision"),
                     "Recall": v.get("recall"),
                     "mAP50": v.get("mAP50"),
                     "mAP50-95": v.get("mAP50-95")})
if rows:
    df = pd.DataFrame(rows)
    df.to_csv(output_dir/"metrics_table.csv", index=False)
    print("Saved metrics_table.csv")

# Find test images (common places)
test_images = []
for p in Path(".").rglob("images"):
    t = p/"test"
    if t.exists():
        imgs = sorted([i for i in t.iterdir() if i.suffix.lower() in [".jpg",".png"]])
        test_images += imgs
if not test_images:
    # fallback to any images under data folders
    imgs = list(Path(".").rglob("*.jpg")) + list(Path(".").rglob("*.png"))
    test_images = imgs[:10]

sample_images = test_images[:3]
print("Using sample images:", sample_images)

def run_predict_and_get_image(model, model_name, img_path, out_prefix="sample"):
    save_project = output_dir / f"{out_prefix}_{model_name}"
    save_project.mkdir(parents=True, exist_ok=True)
    res = model.predict(source=str(img_path), save=True, project=str(save_project), name="predict", exist_ok=True)
    pred_dir = save_project / "predict" / img_path.name
    if not pred_dir.exists():
        # find any image in predict
        preds = list((save_project / "predict").glob("*"))
        for p in preds:
            if p.suffix.lower() in [".jpg", ".png"]:
                return p
        return None
    return pred_dir

comparison_images = []
for img in sample_images:
    v8_img = None
    v9_img = None
    if "YOLOv8" in models and models["YOLOv8"] is not None:
        v8_img = run_predict_and_get_image(models["YOLOv8"], "YOLOv8", img)
    if "YOLOv9" in models and models["YOLOv9"] is not None:
        v9_img = run_predict_and_get_image(models["YOLOv9"], "YOLOv9", img)
    if v8_img and v9_img:
        # stitch side-by-side
        left = Image.open(v8_img).convert("RGB")
        right = Image.open(v9_img).convert("RGB")
        h = max(left.height, right.height)
        left = ImageOps.contain(left, (left.width, h))
        right = ImageOps.contain(right, (right.width, h))
        total_w = left.width + right.width
        combined = Image.new("RGB", (total_w, h))
        combined.paste(left, (0,0))
        combined.paste(right, (left.width,0))
        draw = ImageDraw.Draw(combined)
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", size=20)
        except:
            font = ImageFont.load_default()
        draw.text((10,10), "YOLOv8", fill=(255,255,255), font=font)
        draw.text((left.width+10,10), "YOLOv9", fill=(255,255,255), font=font)
        out_path = output_dir/f"comparison_{img.stem}.png"
        combined.save(out_path)
        comparison_images.append(out_path)
        print("Saved", out_path)
    else:
        print("Missing predictions for", img, "v8:", bool(v8_img), "v9:", bool(v9_img))

# Plot metrics if available
if rows:
    df = pd.read_csv(output_dir/"metrics_table.csv")
    metrics_to_plot = ["Precision", "Recall", "mAP50", "mAP50-95"]
    x = range(len(metrics_to_plot))
    v8_vals = [df[df['Model']=='YOLOv8'][m].values[0] if 'YOLOv8' in df['Model'].values else 0 for m in metrics_to_plot]
    v9_vals = [df[df['Model']=='YOLOv9'][m].values[0] if 'YOLOv9' in df['Model'].values else 0 for m in metrics_to_plot]
    plt.figure(figsize=(8,5))
    plt.bar([i - 0.2 for i in x], v8_vals, width=0.4, label="YOLOv8")
    plt.bar([i + 0.2 for i in x], v9_vals, width=0.4, label="YOLOv9")
    plt.xticks(x, metrics_to_plot)
    plt.ylabel("Score")
    plt.title("YOLOv8 vs YOLOv9 - Metric Comparison")
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_dir/"metric_comparison.png")
    print("Saved metric_comparison.png")
else:
    # placeholder
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8,5))
    plt.text(0.5, 0.6, "No evaluation metrics found\n(ensure dataset YAML is present or run val manually)", ha='center', va='center', fontsize=14)
    plt.axis('off')
    plt.savefig(output_dir/"metric_comparison_placeholder.png")
    print("Saved placeholder chart")

print("Done. Generated files in", output_dir)
