from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import cv2
import os
import uuid
import json
from datetime import datetime
from collections import defaultdict, Counter

# Setup
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
detect_dir = os.path.join("detect", run_id)
classify_dir = os.path.join("classify", run_id)
os.makedirs(detect_dir, exist_ok=True)
os.makedirs(classify_dir, exist_ok=True)

# Load CLIP
model_name = "openai/clip-vit-large-patch14"
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Expanded prompts
prompt_variants = {
    "human": ["a photo of a human", "a person", "a man", "a woman", "a child", "a human face", "a human figure"],
    "peacock": ["a photo of a peacock", "a peacock", "colorful peacock", "wild peacock", "bird peacock"],
    "lion": ["a photo of a lion", "a lion", "wild lion", "african lion", "male lion"],
    "monkey": ["a photo of a monkey", "a monkey", "wild monkey", "monkey closeup", "baby monkey"],
    "panda": ["a photo of a panda", "a panda", "cute panda", "giant panda", "black and white panda"],
    "rat": ["a photo of a rat", "a rat", "small rat", "gray rat", "wild rat"],
    "rhino": ["a photo of a rhino", "a rhino", "white rhino", "wild rhino", "african rhino"],
    "zebra": ["a photo of a zebra", "a zebra", "striped zebra", "wild zebra", "safari zebra"],
    "tiger": ["a photo of a tiger", "a tiger", "bengal tiger", "wild tiger", "tiger face"],
    "donkey": ["a photo of donkey", "a donkey", "brown donkey", "field donkey", "rural donkey"],
    "goat": ["a photo of goat", "a goat", "farm goat", "white goat", "mountain goat"],
    "cow": ["a photo of cow", "a cow", "brown cow", "black and white cow", "farm cow"],
    "dog": ["a photo of dog", "a dog", "pet dog", "puppy", "brown dog"],
    "cat": ["a photo of cat", "a cat", "pet cat", "white cat", "kitten"],
    "pig": ["a photo of pig", "a pig", "pink pig", "farm pig", "baby pig"],
    "duck": ["a photo of duck", "a duck", "yellow duck", "duck in water", "farm duck"],
    "fish": ["a photo of fish", "a fish", "aquarium fish", "tropical fish", "blue fish"],
    "hen": ["a photo of hen", "a hen", "chicken hen", "farm hen", "egg hen"],
    "elephant": ["a photo of elephant", "an elephant", "african elephant", "baby elephant", "giant elephant"]
}

# All flat prompts (for processor text list)
flat_class_prompts = []
prompt_to_label = {}
for label, phrases in prompt_variants.items():
    for phr in phrases:
        flat_class_prompts.append(phr)
        prompt_to_label[phr] = label

# Classification with multiple prompts
def classify_patch(pil_img):
    inputs = processor(text=flat_class_prompts, images=pil_img, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits_per_image.squeeze(0)

    label_scores = defaultdict(list)
    for i, phrase in enumerate(flat_class_prompts):
        label = prompt_to_label[phrase]
        label_scores[label].append(logits[i].item())

    avg_logits = {label: sum(scores) / len(scores) for label, scores in label_scores.items()}
    best_label = max(avg_logits.items(), key=lambda x: x[1])[0]

    logits_tensor = torch.tensor([avg_logits[label] for label in label_scores], device=device)
    probs = torch.softmax(logits_tensor, dim=0)
    best_idx = list(label_scores.keys()).index(best_label)
    confidence = probs[best_idx].item()

    # Boost confidence for humans if > 0.6
    if best_label == "human" and confidence >= 0.6:
        confidence = min(1.0, confidence + 0.1)

    return best_label, confidence

# Fallback with confidence check
def decide_dominant_class(detections, image_path):
    THRESHOLD = 0.75
    label_scores = defaultdict(float)

    for label, conf, _ in detections:
        if conf >= THRESHOLD:
            label_scores[label.lower()] += conf

    if not label_scores:
        pil_img = Image.open(image_path).convert("RGB")
        fallback_label, fallback_conf = classify_patch(pil_img)
        if fallback_conf >= 0.5:
            return fallback_label.lower()
        return "unknown"

    return max(label_scores.items(), key=lambda x: x[1])[0]

# Detection logic
def detect_objects_in_image(image_path, grid_size=(4, 4)):
    print(f"\nProcessing: {image_path}")
    image = cv2.imread(image_path)
    original_image = image.copy()
    patches = split_into_patches(image, grid_size)

    detections = []
    image_id = str(uuid.uuid4())
    detection_summary = {"image": os.path.basename(image_path), "detections": []}

    for idx, ((x1, y1, x2, y2), crop) in enumerate(patches):
        pil_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        label, confidence = classify_patch(pil_crop)

        if confidence > 0.60:
            category = "Human" if label.lower() == "human" else "Animal"
            print(f"Alert: {category} detected - {label} at ({x1},{y1},{x2},{y2}) [Confidence: {confidence:.2f}]")

            detections.append((label, confidence, (x1, y1, x2, y2)))
            cv2.rectangle(original_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text = f"{label.capitalize()} ({confidence:.2f})"
            cv2.putText(original_image, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36,255,12), 2)

            crop_filename = f"{image_id}_{idx}_{label}.jpg"
            crop_path = os.path.join(detect_dir, crop_filename)
            cv2.imwrite(crop_path, crop)

            detection_summary["detections"].append({
                "label": label,
                "confidence": round(confidence, 2),
                "bbox": [x1, y1, x2, y2],
                "image": crop_filename
            })

    annotated_path = os.path.join(detect_dir, f"{image_id}_vis.jpg")
    cv2.imwrite(annotated_path, original_image)

    summary_path = os.path.join(classify_dir, f"{image_id}.json")
    with open(summary_path, 'w') as f:
        json.dump(detection_summary, f, indent=4)

    return detections

# Evaluation
def evaluate_dataset(folder_path):
    total_images, correct_images = 0, 0
    total_patches, correct_patches = 0, 0
    confusion = defaultdict(lambda: defaultdict(int))
    summary = {}

    class_names = sorted([d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))])

    for true_cls in class_names:
        cls_folder = os.path.join(folder_path, true_cls)
        class_total = 0

        for img_file in os.listdir(cls_folder):
            if not img_file.lower().endswith(('.jpg', '.png')):
                continue

            image_path = os.path.join(cls_folder, img_file)
            detections = detect_objects_in_image(image_path, grid_size=(4, 4))

            total_images += 1
            class_total += 1

            label_scores = defaultdict(float)
            for label, conf, _ in detections:
                total_patches += 1
                if label.lower() == true_cls.lower():
                    correct_patches += 1
                if conf >= 0.60:
                    label_scores[label.lower()] += conf

            if label_scores:
                predicted_label = max(label_scores.items(), key=lambda x: x[1])[0]
            else:
                predicted_label = decide_dominant_class(detections, image_path)

            if predicted_label == true_cls.lower():
                correct_images += 1

            confusion[true_cls.lower()][predicted_label] += 1

        summary[true_cls.lower()] = class_total

    image_acc = correct_images / total_images if total_images else 0
    patch_acc = correct_patches / total_patches if total_patches else 0

    print("\n======== EVALUATION REPORT ========")
    print(f"Image-level Accuracy: {correct_images}/{total_images} = {image_acc:.2%}")
    print(f"Patch-level Accuracy: {correct_patches}/{total_patches} = {patch_acc:.2%}\n")

    print("Confusion Matrix:")
    for true_cls in class_names:
        print(f"\nTrue Class: {true_cls}")
        for pred_cls, count in confusion[true_cls.lower()].items():
            print(f"  Predicted as {pred_cls.capitalize()}: {count}")

    report = {
        "image_level_accuracy": {
            "correct": correct_images,
            "total": total_images,
            "percent": round(image_acc * 100, 2)
        },
        "patch_level_accuracy": {
            "correct": correct_patches,
            "total": total_patches,
            "percent": round(patch_acc * 100, 2)
        },
        "per_class_total": summary,
        "confusion_matrix": json.loads(json.dumps(confusion, default=dict))
    }

    report_path = os.path.join(classify_dir, "evaluation_report.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)
    print(f"\nReport saved to: {report_path}")

# Patching function unchanged
def split_into_patches(image, grid_size=(3, 3)):
    h, w = image.shape[:2]
    patch_h, patch_w = h // grid_size[0], w // grid_size[1]
    patches = []

    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            x1, y1 = j * patch_w, i * patch_h
            x2, y2 = x1 + patch_w, y1 + patch_h
            crop = image[y1:y2, x1:x2]
            patches.append(((x1, y1, x2, y2), crop))
    return patches

# Video mode
def run_on_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % 30 != 0:
            continue
        temp_path = "temp_frame.jpg"
        cv2.imwrite(temp_path, frame)
        detect_objects_in_image(temp_path, grid_size=(4, 4))
    cap.release()

# CLI
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, help="Path to image")
    parser.add_argument("--video", type=str, help="Path to video")
    parser.add_argument("--eval", type=str, help="Path to eval dataset folder")
    args = parser.parse_args()

    if args.image:
        detect_objects_in_image(args.image)
    elif args.video:
        run_on_video(args.video)
    elif args.eval:
        evaluate_dataset(args.eval)
    else:
        print("Provide --image, --video or --eval")
