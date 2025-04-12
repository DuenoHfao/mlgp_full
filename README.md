# Image Preprocessing and YOLOv8 Training

This project explores the impact of various image preprocessing techniques on the performance of YOLOv8 for object detection. The preprocessing methods include Zero-DCE, Retinex, Histogram Equalization, CLAHE, and YOLO's built-in augmentation functions. The goal is to evaluate how these techniques affect the model's loss, accuracy, precision, and recall over several training epochs.

## Table of Contents
- [Introduction](#introduction)
- [Preprocessing Methods](#preprocessing-methods)
- [Dataset Preparation](#dataset-preparation)
- [YOLOv8 Training](#yolov8-training)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [Conclusion](#conclusion)

## Introduction
Image preprocessing can significantly impact the performance of object detection models. This project applies multiple preprocessing techniques to augment the dataset and trains YOLOv8 to measure the effects on model performance.

## Preprocessing Methods
The following preprocessing methods are applied to the dataset:
1. **Zero-DCE**: Enhances low-light images using deep learning.
2. **Retinex**: Improves image visibility by simulating human visual perception.
3. **Histogram Equalization**: Enhances contrast by redistributing pixel intensity.
4. **CLAHE**: Adaptive histogram equalization to improve local contrast.
5. **YOLO's Built-in Augmentation**: Includes flipping, scaling, rotation, and color jitter.

## Dataset Preparation
1. Collect a dataset of images for object detection.
2. Apply each preprocessing method to augment the dataset.
3. Split the dataset into training, validation, and test sets.

## YOLOv8 Training
1. Install [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics).
2. Train YOLOv8 on the preprocessed datasets using the following command:
    ```bash
    yolo task=detect mode=train data=dataset.yaml model=yolov8n.pt epochs=50 imgsz=640
    ```
3. Monitor the training process, including loss, accuracy, precision, and recall.

## Evaluation Metrics
- **Loss**: Measures the error during training.
- **Accuracy**: Measures the percentage of correctly predicted objects.
- **Precision**: Measures the proportion of true positives among predicted positives.
- **Recall**: Measures the proportion of true positives among actual positives.

## Results
The results are compared across different preprocessing methods:
| Preprocessing Method | Loss (Final Epoch) | Accuracy | Precision | Recall |
|-----------------------|--------------------|----------|-----------|--------|
| Zero-DCE             | TBD                | TBD      | TBD       | TBD    |
| Retinex              | TBD                | TBD      | TBD       | TBD    |
| Histogram Equalization| TBD               | TBD      | TBD       | TBD    |
| CLAHE                | TBD                | TBD      | TBD       | TBD    |
| YOLO Augmentation    | TBD                | TBD      | TBD       | TBD    |

## Conclusion
This project demonstrates the impact of various preprocessing techniques on YOLOv8's performance. The results provide insights into which methods are most effective for enhancing object detection accuracy.

## References
- [Zero-DCE Paper](https://arxiv.org/pdf/2001.06826)
- [Retinex Theory](https://en.wikipedia.org/wiki/Color_constancy#Retinex_theory)
- [YOLOv8 Documentation](https://docs.ultralytics.com/models/yolov8/#overview)
