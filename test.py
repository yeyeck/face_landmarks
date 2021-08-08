import cv2
from utils.datasets import Transform, FaceLandmarksDataset, Resize, RandomEarse, RandomLinear, RandomHorizontalFlip
# from utils.datasets import FacePointDataset
from utils.visual import plot_points
import numpy as np

if __name__ == '__main__':
    dataset = FaceLandmarksDataset('../../data/train_data/images/train', '../../data/train_data/labels/train', transform=Transform([
        Resize(size=(128, 128)),
        RandomLinear(),
        RandomHorizontalFlip(p=1, points=68),
        RandomEarse(p=0.5)
    ]))
    for i in range(100):
        img_pth,img, label = dataset.__getitem__(i)
        img = cv2.resize(img, dsize=(640, 640))
        h, w, _ = img.shape
        points = label[:136]
        points = np.float32(points).reshape(-1, 2) * np.float32([w, h]) 

        pose = label[136:139]
        pitch, yaw, roll = pose * 180 / np.pi
        img = plot_points(img, points)
        
        cv2.putText(img, f'{pitch}, {yaw}, {roll}', (10, 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), thickness=1)
        cv2.imshow('name', img)
        cv2.waitKey(0)