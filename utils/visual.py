import cv2
import matplotlib.pyplot as plt
import os



def plot_points(img, points):
    count = 0
    for point in points:
        cv2.circle(img, (int(point[0]), int(point[1])), 0, (0, 255, 0), 3)
        # cv2.putText(img, str(count), (int(point[0]), int(point[1])), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
        count += 1
    return img

def plot_loss(train_loss, val_loss, save_to):
    x = list(range(len(train_loss)))
    y1 = train_loss
    y2 = val_loss
    plt.figure()
    plt.title('Loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.plot(x, y1, color='b', label='train')
    plt.plot(x, y2, color='r', label='val')
    plt.legend()
    plt.savefig(os.path.join(save_to, 'loss.jpg'))
    plt.close()

def plot_lr(lrs, save_to):
    x = list(range(len(lrs)))
    y = lrs
    plt.title('lr scheduler')
    plt.xlabel('epochs')
    plt.ylabel('lr')
    plt.plot(x, y, color='r', label='learning rate')
    plt.legend()
    plt.savefig(os.path.join(save_to, 'lr.jpg'))
    plt.close()
