import torch
from torch import nn
import math
 
 
class WingLoss(nn.Module):
    def __init__(self, omega=10, epsilon=2, reduction="mean"):
        # 10, 2
        super(WingLoss, self).__init__()
        self.omega = omega
        self.epsilon = epsilon
        self.reduction = reduction
 
    def forward(self, pred, target):
        y = target
        y_hat = pred
        delta_y = (y - y_hat).abs()
        delta_y1 = delta_y[delta_y < self.omega]
        delta_y2 = delta_y[delta_y >= self.omega]
        loss1 = self.omega * torch.log(1 + delta_y1 / self.epsilon)
        C = self.omega - self.omega * math.log(1 + self.omega / self.epsilon)
        loss2 = delta_y2 - C
        loss = torch.cat((loss1, loss2), 0)
 
        if self.reduction == 'mean':
            loss = loss.mean()
        if self.reduction == 'sum':
            loss = loss.sum()
        return loss
         
class PFLDLoss(nn.Module):
    def __init__(self, points=98, dtype='WING', inner_enhance=False):
        super(PFLDLoss, self).__init__()
         
        self.points = points
        if self.points == 98:
            self.inner_start = 102
            self.eye = (120, 152)
        if self.points == 68:
            self.inner_start = 54
            self.eye = (72, 96)

        if dtype == 'MSE':
            self.distance = nn.MSELoss()
        elif dtype == 'WING':
            self.distance = WingLoss()
        else:
            self.distance = nn.SmoothL1Loss()
        self.inner_enhance = inner_enhance     
    def forward(self ,pred, target):
        pt_end = self.points * 2
        theta = torch.sum(1- torch.cos(target[:, pt_end:pt_end+3]), dim=1)
        loss_pts = theta * self.distance(pred[:, :pt_end], target[:, :pt_end])

        loss_pose = self.distance(pred[:, pt_end:pt_end+3], target[:, pt_end:pt_end+3])
        
        loss = loss_pts.mean() + loss_pose.mean()
        if self.inner_enhance:
            loss_inner = self.distance(pred[:, self.inner_start:pt_end], target[:, self.inner_start:pt_end])
            loss_eye = self.distance(pred[:, self.eye[0]:self.eye[1]], target[:, self.eye[0]:self.eye[1]]) * 0.5
            loss += loss_inner.mean() + loss_eye.mean()
            loss += loss_inner.mean()
        return loss 