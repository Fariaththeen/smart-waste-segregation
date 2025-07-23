# smart-waste-segregation
This project aims to detect and segregate plastic waste in water bodies using the YOLOv5 object detection model. It enables real-time plastic identification from images and videos to support environmental cleanup efforts. Built using Python, OpenCV, and PyTorch. Ideal for smart water cleaning and pollution control systems.

## 🚀 Features

- 🔍 Real-time plastic waste detection using YOLOv5
- 🧠 Custom-trained model for water-based scenarios
- 📹 Live camera feed and video file support
- 📁 Lightweight and modular codebase
- 📈 Option to expand into analytics and reporting

- ## 📌 Problem Statement

Plastic waste in rivers, lakes, and oceans poses a severe threat to aquatic ecosystems. Manual monitoring is not scalable. This project proposes an AI-powered solution to automatically detect **floating plastic waste** using drone or surveillance imagery.

---

## 🎯 Objectives

- Detect floating plastic waste using **YOLOv5** object detection.
- Classify waste into **biodegradable** and **non-biodegradable**.
- Provide visual outputs (bounding boxes) in real-time.
- Aid authorities and researchers in **monitoring pollution** efficiently.

---

## 🔧 Tech Stack

- **YOLOv5** (PyTorch implementation)
- **Python 3.10+**
- **OpenCV**
- **Google Colab / Local System**
- **LabelImg** (for annotation)
- **Roboflow** (optional for dataset management)

---

## 📁 Dataset

We used publicly available datasets and images of plastic waste floating in water. You can find a sample dataset here:  
🔗 [Plastic Bottle Dataset (GitHub)](https://github.com/m0-n/Plastic-Bottles-Dataset)

- Images: 300–1000+ water surface images
- Classes: `plastic`, `other`

---

## 🧠 Model Training

To train your model:
```bash
python train.py --img 640 --batch 16 --epochs 100 --data waste.yaml --weights yolov5s.pt
