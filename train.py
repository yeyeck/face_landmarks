import os, time, argparse

from utils.model import create_model
from utils.datasets import create_dataloader
from utils.loss import PFLDLoss
from utils.visual import plot_loss, plot_lr
from tqdm import tqdm
import torch


def train_loop(model, dataloader, lossfn, optimizer, epoch, epochs):
    model.train()
    pbar = tqdm(dataloader)
    print(('\n' + '%12s' * 3) % ('Epoch', 'gpu_mem', 'loss'))
    for _, X, y in pbar:
        # gpu spped up
        if torch.cuda.is_available():
            X = X.cuda()
            y = y.cuda()
        # froward propagation
        pred = model(X)
        loss = lossfn(pred, y)

        # backward propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss = loss.item()
        
        # gpu mem retrieve
        mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)
        # process bar logs
        s = ('%12s' * 3) % ('%g/%g' % (epoch + 1, epochs), mem, '%.7g'% loss)
        pbar.set_description(s)
    return loss

def val_loop(model, dataloader, loss_fn, epoch, epochs):
    model.eval()
    size = len(dataloader)
    pbar = tqdm(dataloader)
    total_loss = 0.
    with torch.no_grad():
        for _, X, y in pbar:
            if torch.cuda.is_available():
                X = X.cuda()
                y = y.cuda()
            pred = model(X)
            loss = loss_fn(pred, y).item()
            total_loss += loss
            mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)
            s = ('%12s' * 3) % ('%g/%g' % (epoch + 1, epochs), mem, '%.7g'% loss)
            pbar.set_description(s)
    return total_loss / size

def main(opt):
    
    # args
    batch_size, img_size, epochs, workers, data, weights, lr, optimizer_type, name, loss_type, points, inner_enhance \
        = opt.batch_size, opt.img_size, opt.epochs, opt.workers, opt.data, opt.weights, opt.lr, opt.optimizer, \
            opt.name, opt.loss, opt.points, opt.inner_enhance
    
    # check and make the path if not exists
    save_to = os.path.join('runs/train', name)
    if os.path.exists(save_to):
        name = time.strftime('%Y%m%d-%H-%M-%S', time.localtime())
        print(f'{save_to} is exists, choose a new name: {name}')
        save_to = os.path.join('runs/train', name)

    if not os.path.exists(save_to):
        os.makedirs(save_to)
    print(f'All results will be saved in {save_to}')

    # data
    train_loader = create_dataloader(data, batch_size=batch_size, workers=workers, img_size=img_size, train=True)
    val_loader = create_dataloader(data, batch_size=batch_size, workers=workers, img_size=img_size, train=False)

    # model
    if weights and weights != '':
        print(f'loading model from {weights}')
        model = torch.load(weights)
    else:
        model = create_model(output_features=points * 2)
    
    print(model)
    
    # loss, optimizer
    lossfn = PFLDLoss(points=points, dtype=loss_type, inner_enhance=inner_enhance)
    if optimizer_type == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif optimizer_type == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.98)

    # use cuda if it is available
    if torch.cuda.is_available():
        model = model.cuda()
        lossfn = lossfn.cuda()  

    # training
    best_loss = 0
    lrs = []
    train_losses = []
    val_losses = []
    for epoch in range(epochs):
        # training and validating
        train_loss = train_loop(model, train_loader, lossfn, optimizer, epoch, epochs)
        val_loss = val_loop(model, val_loader, lossfn, epoch, epochs)
        
        # record the training
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        lrs.append(scheduler.get_last_lr()[0])
        
        scheduler.step()
        if best_loss == 0 or best_loss > val_loss:
            best_loss = val_loss
            torch.save(model, os.path.join(save_to, 'best.pth'))
        
        if (epoch + 1) % 10 == 0:
            plot_loss(train_losses, val_losses, save_to)
            plot_lr(lrs, save_to)

    torch.save(model, os.path.join(save_to, 'final.pth'))
    print(f'All results are saved in {save_to}')\

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='../wflw', help='data for training')
    parser.add_argument('--loss', type=str, default='WING', help='Distance type to caculate loss: MSE, L1 or WING')
    parser.add_argument('--points', type=int, default=98, help='batch-size')
    parser.add_argument('--lr', type=float, default=0.01, help='data for training')
    parser.add_argument('--weights', type=str, default='', help='data for training')
    parser.add_argument('--batch-size', type=int, default=256, help='batch-size')
    parser.add_argument('--img-size', type=int, default=128, help='input image size')
    parser.add_argument('--workers', type=int, default=8, help='number of workers')
    parser.add_argument('--epochs', type=int, default=100, help='Epochs')
    parser.add_argument('--optimizer', type=str, default='sgd', help='optimizer type, adam or sgd')
    parser.add_argument('--name', type=str, default='exp', help='name of the traning')
    parser.add_argument('--inner-enhance', action='store_true', default=False, help='use inner enhance loss')
    opt = parser.parse_args()

    main(opt)
    