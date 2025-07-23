import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages
from utils.general import check_img_size, non_max_suppression, scale_coords
import torch
import cv2
import numpy as np


# Load model directly
model = torch.load('best.pt', map_location='cpu')['model'].float()
model.eval()

# Load an image
img_path = 'test.jpg'  # Replace with your image
img = cv2.imread(img_path)
img = cv2.resize(img, (640, 640))
img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
img = np.ascontiguousarray(img)
img_tensor = torch.from_numpy(img).float()
img_tensor /= 255.0
img_tensor = img_tensor.unsqueeze(0)

# Inference
pred = model(img_tensor, augment=False)[0]
pred = non_max_suppression(pred, 0.25, 0.45)[0]

# Draw boxes
for *xyxy, conf, cls in pred:
    label = f'{conf:.2f}'
    cv2.rectangle(img.transpose(1, 2, 0), (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (255,0,0), 2)
    cv2.putText(img.transpose(1, 2, 0), label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

# Show image
cv2.imshow('Waste Detection', img.transpose(1, 2, 0))
cv2.waitKey(0)
cv2.destroyAllWindows()
