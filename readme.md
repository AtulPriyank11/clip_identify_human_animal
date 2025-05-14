Description:

This project uses OpenAI's CLIP model to perform zero-shot image classification and object detection. It is designed to distinguish between humans and a wide variety of animals without requiring any additional training. The system uses a patch-based approach where each image is split into smaller regions, and each region is classified using natural language prompts.

Features:

    Zero-shot classification using pre-trained CLIP model

    Works without any model training or fine-tuning

    Classifies and detects both humans and animals

    Patch-wise detection using customizable grid sizes like 3x3 or 4x4

    Generates evaluation metrics, confusion matrix, and confidence scores

    Supports processing of folders, individual images, and video files

    Outputs cropped detections, annotated images, and JSON logs

How to Use:

To evaluate a dataset folder structured by class (for example, human, tiger, zebra):

python main.py --eval path_to_dataset_folder

To classify a single image:

python main.py --image path_to_image.jpg

To run the system on a video file:

python main.py --video path_to_video.mp4

Input Folder Structure:

Each class should have its own folder inside the dataset folder. Example:

test_folder/ human/ tiger/ zebra/

Output:

    Detect folder will contain cropped detected regions and annotated images (bounding boxes and labels)

    Classify folder will contain classification logs and the evaluation report in JSON format

Input and Output Image Suggestions:

Input images can include raw images such as a human photo, tiger photo, zebra photo, etc.

Output images to include:

    Annotated images with bounding boxes showing detected classes and confidence scores

    Cropped patches with class names in the filename

    JSON report showing all detections and evaluation results

Example Evaluation Output:

Image-level Accuracy: 759 correct out of 887 total = 85.57 percent
Patch-level Accuracy: 4114 correct out of 4738 total = 86.83 percent

Example Confusion Matrix Snippet:

True Class: human
Predicted as Human: 39
Predicted as Unknown: 61
Predicted as Duck: 1
...

Requirements:

    Python 3.8 or later

    Transformers

    Torch

    OpenCV (cv2)

    Pillow (PIL)

Install dependencies using:

pip install -r requirements.txt

Notes:
The system relies entirely on the power of the CLIP model and prompt engineering. Each class is associated with multiple descriptive phrases to improve accuracy. Patch-wise classification increases robustness for detection, though some classes such as humans can be misclassified when context is lost.