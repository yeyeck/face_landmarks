import cv2
from utils.datasets import Transform, WFLWDataset, Resize, RandomEarse, RandomLinear, RandomHorizontalFlip
from utils.visual import plot_points
import numpy as np

if __name__ == '__main__':
    dataset = WFLWDataset(root='../wflw', train=True, transform=Transform([
        Resize(size=(128, 128)),
        RandomLinear(),
        RandomHorizontalFlip(),
        RandomEarse(p=0.5),
    ]))
    for i in range(100):
        name, img, label = dataset.__getitem__(i)
        h, w, _ = img.shape
        points = label[:196]
        points = np.float32(points).reshape(-1, 2) * np.float32([w, h]) 

        pose = label[196:199]
        pitch, yaw, roll = pose * 180 / np.pi
        img = plot_points(img, points)
        img = cv2.resize(img, dsize=(320, 320))
        cv2.putText(img, f'{pitch}, {yaw}, {roll}', (10, 10), cv2.FONT_HERSHEY_COMPLEX, 0.3, (255, 0, 0), thickness=1)
        cv2.imshow(name, img)
        cv2.waitKey(0)
        print(img.shape, len(label))