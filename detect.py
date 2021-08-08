import torch
import numpy as np
import cv2
import time, glob, os

from utils.datasets import Resize, Normalize, Transform
from utils.visual import plot_points



class FacePointsDetector(object):
    def __init__(self, model_pth, img_size=128):
        self.img_size = img_size
        self.model = torch.load(model_pth, map_location='cpu')
        self.transform = Transform([
            Resize(size=(img_size, img_size)),
            Normalize()
        ])

    def __call__(self, img):
        self.model.eval()
        input = img
        h_, w_ = input.shape[:2]
        input, _ = self.transform(img)
        input = input.unsqueeze(0)
        input = input.to(torch.float32)
        output = self.model(input)
        pose = output[:, -3:]

        points = output[:, :-3].reshape(-1, 2).numpy()
        print(points.shape)
        points = points * np.array([w_, h_])
        points = points.astype(np.int32)
        return points, pose

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--source', type=str, default='./data/images/test', help='data for detecting')
    # parser.add_argument('--weights', type=str, default='best.pth', help='data for detecting')
    # parser.add_argument('--save_to', type=str, default='./runs', help='path to save the results')
    # opt = parser.parse_args()


    # model = torch.load('runs/train/exp/best.pth', map_location='cpu')

    with torch.no_grad():
        detector = FacePointsDetector('runs/train/mse68/best.pth')
        for pth in glob.glob('../test/**.png'):
            input = cv2.imread(pth)
            img = cv2.imread(pth)
            t0 = time.time()
            # input = cv2.resize(input, dsize=(640, 640))
            points, pose = detector(input)
            print(pose * 180/np.pi)
            print(time.time() - t0)
            # img = cv2.resize(img, dsize=(640, 640))
            img = plot_points(img, points)
            
            cv2.imshow('test05.jpg', img)
            cv2.waitKey(0)