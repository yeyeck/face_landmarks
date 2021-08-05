import glob, os, cv2, random, math
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms
# from utils.pose import calculate_pitch_yaw_roll
# from utils.visual import plot_points



class Transform(object):
    def __init__(self, transforms):
        super(Transform, self).__init__()
        self.transforms = []
        self.transforms.extend(transforms)
    
    def __call__(self, img, label=None):
        for trans in self.transforms:
            img, label = trans(img, label)
        return img, label

class Normalize(object):
    def __init__(self):
        self.transform = transforms.ToTensor()
    
    def __call__(self, img, label=None):
        img = img / 255.0
        img = self.transform(img)
        return img.type(torch.FloatTensor), label

class Resize(object):
    def __init__(self, size=(128, 128)):
        self.size = size
    
    def __call__(self, img, label=None):
        img = cv2.resize(img, dsize=self.size)
        return img, label

class RandomEarse(object):
    def __init__(self, p, scale=(0.05, 0.3)) -> None:
        self.p = p
        self.scale = scale

    
    def __call__(self, img, label=None):
        if random.random() < self.p:
            h, w, _ = img.shape
            x1, y1, x2, y2 = self._randombox(h, w)
            img[y1:y2, x1:x2, :] = np.random.randint(0, 255)
        return img, label 


    def _randombox(self, h, w):
        box_h = h * random.uniform(self.scale[0], self.scale[1])
        box_w = w * random.uniform(self.scale[0], self.scale[1])
        x1 = random.randint(0, int(w - box_w))
        y1 = random.randint(0, int(h - box_h))
        x2 = x1 + int(box_w)
        y2 = y1 + int(box_h)

        return x1, y1, x2, y2


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, img, label=None):
        if random.random() < self.p:
            img = cv2.flip(img, 1)
            if label.any():
                for i in range(len(label)):
                    if i % 2 == 0:
                        label[i] = 1 - label[i]
                points = np.array(label).reshape((-1, 2))
                label = self._flip(points)
        return img, label

    def _flip(self, points):
        idx = [32,31,30,29,28,27,26,25,24,23,22,21,20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0,46,45,44,43,42,50,49,48,47,37,36,35,34,33,41,40,39,38,51,
               52,53,54,59,58,57,56,55,72,71,70,69,68,75,74,73,64,63,62,61,60,67,66,65,82,81,80,79,78,77,76,87,86,85,84,83,92,91,90,89,88,95,94,93,97,96]
        points_ = []
        for i in idx:
            points_.extend(points[i])
        return points_

class RandomLinear(object):
    def __init__(self, alpha:tuple =(0.3, 1), beta=(0, 150), p=0.5):
        self.p = p
        self.alpha_min = min(alpha)
        self.alpha_max = max(alpha)
        self.beta_min = min(beta)
        self.beta_max = max(beta)
        self.pa = (self.alpha_max - self.alpha_min) / 10
        self.pb = (self.beta_max - self.beta_min) / 10
            
    def __call__(self, img, label=None):
        if random.random() < self.p:
            alpha = self.alpha_min + self.pa * random.randint(0, 10)
            beta = self.beta_min + self.pb * random.randint(0, 10)
            img = np.uint8(np.clip(img * alpha + beta, 0, 255))
        return img, label



def face_bbox(points, img_w, img_h):
    x1, y1 = min(points[:, 0]), min(points[:, 1])
    x2, y2 = max(points[:, 0]), max(points[:, 1])
    w, h = x2 - x1, y2 - y1
    cx, cy = x1 + w / 2, y1 + h / 2
    w, h = w * 1.2, h * 1.2
    x1, y1 = cx - w /2, cy - h / 2
    x2, y2 = cx + w /2, cy + h / 2

    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img_w, x2), min(img_h, y2)
    return int(x1), int(y1), int(x2), int(y2)


class WFLWDataset(Dataset):
    def __init__(self, root, train=False, transform=None) -> None:
        super(WFLWDataset, self).__init__()
        self.root = root
        label_name =  'train.txt' if train else 'test.txt'
        self.label = os.path.join(root, label_name)
        self.transform = transform
        
        self.data = []
        with open(self.label, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split()
                img_name = line[-1]
                points = [float(s) for s in line[:196]]
                points = np.float32(points).reshape(-1, 2)
                status = [int(s) for s in line[200:206]]
                self.data.append([img_name, points, np.float32(status)])
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data =self.data[idx]
        img_name, landmarks, status = data
        
        # img
        img_pth = os.path.join(self.root, 'images', img_name)
        img = cv2.imread(img_pth)
        h, w, _ = img.shape
        # crop face
        x1, y1, x2, y2 = face_bbox(landmarks, w, h)
        img = img[y1:y2, x1:x2, :]
        face_h, face_w, _ = img.shape
        # landmarks
        landmarks = landmarks - np.float32([x1, y1])
        landmarks = landmarks / np.float32([face_w, face_h])
        landmarks = landmarks.reshape(196)

        # transform
        if self.transform:
            img, landmarks = self.transform(img, landmarks)

        landmarks_ = np.float32(landmarks).reshape(-1, 2) * np.array([face_w, face_h])
        TRACKED_POINTS = [33, 38, 50, 46, 60, 64, 68, 72, 55, 59, 76, 82, 85, 16]
        landmark2D = []
        for id in TRACKED_POINTS:
            landmark2D.append(landmarks_[id])
        pitch, yaw, roll = calculate_pitch_yaw_roll(np.float32(landmark2D), face_w, face_h)
        pose = np.float32([pitch, yaw, roll]) * np.pi / 180
        label = np.concatenate((landmarks, pose, status))
        return img_name, img, label


def calculate_pitch_yaw_roll(landmarks_2D ,cam_w=256, cam_h=256,radians=False):
    """ Return the the pitch  yaw and roll angles associated with the input image.
    @param radians When True it returns the angle in radians, otherwise in degrees.
    """
    c_x = cam_w/2
    c_y = cam_h/2
    f_x = c_x / np.tan(60/2 * np.pi / 180)
    f_y = f_x

    #Estimated camera matrix values.
    camera_matrix = np.float32([[f_x, 0.0, c_x],
                                [0.0, f_y, c_y],
                                [0.0, 0.0, 1.0]])

    camera_distortion = np.float32([0.0, 0.0, 0.0, 0.0, 0.0])

    #The dlib shape predictor returns 68 points, we are interested only in a few of those
    # TRACKED_POINTS = [17, 21, 22, 26, 36, 39, 42, 45, 31, 35, 48, 54, 57, 8]
    #wflw(98 landmark) trached points
    # TRACKED_POINTS = [33, 38, 50, 46, 60, 64, 68, 72, 55, 59, 76, 82, 85, 16]
    #X-Y-Z with X pointing forward and Y on the left and Z up.
    #The X-Y-Z coordinates used are like the standard
    # coordinates of ROS (robotic operative system)
    #OpenCV uses the reference usually used in computer vision:
    #X points to the right, Y down, Z to the front
    LEFT_EYEBROW_LEFT  = [6.825897, 6.760612, 4.402142]
    LEFT_EYEBROW_RIGHT = [1.330353, 7.122144, 6.903745]
    RIGHT_EYEBROW_LEFT = [-1.330353, 7.122144, 6.903745]
    RIGHT_EYEBROW_RIGHT= [-6.825897, 6.760612, 4.402142]
    LEFT_EYE_LEFT  = [5.311432, 5.485328, 3.987654]
    LEFT_EYE_RIGHT = [1.789930, 5.393625, 4.413414]
    RIGHT_EYE_LEFT = [-1.789930, 5.393625, 4.413414]
    RIGHT_EYE_RIGHT= [-5.311432, 5.485328, 3.987654]
    NOSE_LEFT  = [2.005628, 1.409845, 6.165652]
    NOSE_RIGHT = [-2.005628, 1.409845, 6.165652]
    MOUTH_LEFT = [2.774015, -2.080775, 5.048531]
    MOUTH_RIGHT=[-2.774015, -2.080775, 5.048531]
    LOWER_LIP= [0.000000, -3.116408, 6.097667]
    CHIN     = [0.000000, -7.415691, 4.070434]

    landmarks_3D = np.float32( [LEFT_EYEBROW_LEFT,
                                LEFT_EYEBROW_RIGHT,
                                RIGHT_EYEBROW_LEFT,
                                RIGHT_EYEBROW_RIGHT,
                                LEFT_EYE_LEFT,
                                LEFT_EYE_RIGHT,
                                RIGHT_EYE_LEFT,
                                RIGHT_EYE_RIGHT,
                                NOSE_LEFT,
                                NOSE_RIGHT,
                                MOUTH_LEFT,
                                MOUTH_RIGHT,
                                LOWER_LIP,
                                CHIN])

    #Return the 2D position of our landmarks
    assert landmarks_2D is not None ,'landmarks_2D is None'
    

    landmarks_2D = np.asarray(landmarks_2D,dtype=np.float32).reshape(-1,2)
    #Applying the PnP solver to find the 3D pose
    #of the head from the 2D position of the
    #landmarks.
    #retval - bool
    #rvec - Output rotation vector that, together with tvec, brings
    #points from the world coordinate system to the camera coordinate system.
    #tvec - Output translation vector. It is the position of the world origin (SELLION) in camera co-ords
    retval, rvec, tvec = cv2.solvePnP(landmarks_3D,
                                      landmarks_2D,
                                      camera_matrix,
                                      camera_distortion)

    #Get as input the rotational vector
    #Return a rotational matrix
    rmat, _ = cv2.Rodrigues(rvec)
    pose_mat = cv2.hconcat((rmat,tvec))

    #euler_angles contain (pitch, yaw, roll)
    # euler_angles = cv2.DecomposeProjectionMatrix(projMatrix=rmat, cameraMatrix=self.camera_matrix, rotMatrix, transVect, rotMatrX=None, rotMatrY=None, rotMatrZ=None)
    _, _, _, _, _, _,euler_angles = cv2.decomposeProjectionMatrix(pose_mat)
    pitch,yaw,roll =map(lambda temp:temp[0],euler_angles)
    return pitch,yaw,roll

    # head_pose = [ rmat[0,0], rmat[0,1], rmat[0,2], tvec[0],

                   # rmat[1,0], rmat[1,1], rmat[1,2], tvec[1],

                   # rmat[2,0], rmat[2,1], rmat[2,2], tvec[2],

                         # 0.0,      0.0,        0.0,    1.0 ]

    #print(head_pose) #TODO remove this line

    # return self.rotationMatrixToEulerAngles(rmat)
# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).

def rotationMatrixToEulerAngles(R) :
    #assert(isRotationMatrix(R))
    #To prevent the Gimbal Lock it is possible to use
    #a threshold of 1e-6 for discrimination
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])

    singular = sy < 1e-6
    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
    return np.array([x, y, z])


def create_dataloader(src, batch_size=256, workers=0, img_size=128, train=False, datatype='WFLW'):
    if train:
        trans = [
            Resize(size=(img_size, img_size)),
            RandomLinear(),
            RandomHorizontalFlip(),
            RandomEarse(p=0.5),
            Normalize()
        ]
    else:
        trans = [
            Resize(size=(img_size, img_size)),
            Normalize()
        ]
    transform = Transform(trans)
    if datatype == 'WFLW':
        dataset = WFLWDataset(root=src, train=train, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers)