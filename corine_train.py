import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from unet import *
import metrics
import dice_loss_function
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import albumentations as A
import time
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
from deeplabV3plus.deeplabv3 import DeepLabV3Plus
from deeplabV3plus import encoders
from model_unet import *
import dataset_corine

#CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs, plot=True):
    start = time.time()

    train_loss = []
    train_wiou = []
    train_oacc = []
    train_miou = []

    validation_loss = []
    validation_wiou = []
    validation_oacc = []
    validation_miou = []

    max_score = 0.0

    for epoch in range(epochs):
        #TRAIN
        model.train() 
        epoch_loss = 0
        epoch_wmiou = 0
        epoch_miou = 0
        epoch_acc = 0

        with tqdm(total=len(train_data), desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch_idx, batch in enumerate(train_loader, 0):
                optimizer.zero_grad()
                images= batch['img']
                true_masks = batch['label']
            
                images = images.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                pred_masks = model(images)#pred_masks shape: [batch * Nclasses * H * W]
                
                loss = criterion(pred_masks, true_masks)
                epoch_loss += loss.item()
                
                wmiou, miou, _   = mIoU(pred_masks, true_masks)
                epoch_wmiou += wmiou.cpu().detach().numpy()
                epoch_miou += miou.cpu().detach().numpy()
                acc, _ = accuracy(pred_masks, true_masks)
                epoch_acc += acc.cpu().detach().numpy()
                
                pbar.set_postfix(**{'Loss': epoch_loss / (batch_idx+1),
                                    'wIoU': epoch_wmiou / (batch_idx+1),
                                    'Accuracy': epoch_acc / (batch_idx+1),
                                    'mIoU': epoch_miou / (batch_idx+1)})
                pbar.update(images.shape[0])

                loss.backward()
                optimizer.step()

        if scheduler != None:
            #scheduler.step()
            scheduler.step(epoch_wmiou) #Reduce on plateau parameter maximitzation
        
        train_loss.append(epoch_loss / (batch_idx+1))
        train_wiou.append(epoch_wmiou / (batch_idx+1))
        train_oacc.append(epoch_acc / (batch_idx+1))
        train_miou.append(epoch_miou / (batch_idx+1))

        #VALIDATION
        model.eval()
        val_loss = 0
        val_wmiou = 0
        val_miou = 0
        val_acc = 0

        with tqdm(total=len(val_data), desc='Validation set', unit='img') as pbar:
            for batch_idx, batch in enumerate(val_loader, 0):
                images= batch['img']
                true_masks = batch['label']
                images = images.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.no_grad():
                    pred_masks = model(images)

                val_loss += criterion(pred_masks, true_masks).item()
                
                wmiou, miou, _   = mIoU(pred_masks, true_masks)
                val_wmiou += wmiou.cpu().detach().numpy()
                acc, _ = accuracy(pred_masks, true_masks)
                val_acc += acc.cpu().detach().numpy()
                val_miou += miou.cpu().detach().numpy()

                pbar.set_postfix(**{'Loss': val_loss / (batch_idx+1),
                                   'wIoU': val_wmiou / (batch_idx+1),
                                   'Accuracy': val_acc / (batch_idx+1),
                                   'mIoU': val_miou / (batch_idx+1)})
                
                pbar.update(images.shape[0])

        validation_loss.append(val_loss / (batch_idx+1))
        validation_wiou.append(val_wmiou / (batch_idx+1))
        validation_oacc.append(val_acc / (batch_idx+1))
        validation_miou.append(val_miou / (batch_idx+1))
        
        if max_score < val_wmiou: #Save the best model (IoU)
            max_score = val_wmiou
            torch.save(model, './Corine/final_tunning/model_21.pth')
            print('Model saved!')
            state = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'train_loss': train_loss,
                'train_w_iou': train_wiou,
                'train_m_iou':train_miou,
                'train_accuracy': train_oacc,
                'valid_loss': validation_loss,
                'valid_w_iou': validation_wiou,
                'valid_m_iou':validation_miou,
                'valid_accuracy':validation_oacc,
            }
            torch.save(state, './Corine/checkpoints/model_21.pth')



    print('Finished Training!')
    stop = time.time()
    hh_mm_ss= time.strftime('%H:%M:%S', time.gmtime(stop-start))
    print(f"Training time: {hh_mm_ss}")

    #PLOTS
    if plot:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots( nrows=2, ncols=2,figsize=(15,10))  # create figure & 1 axis

        ax1.plot(range(0, epochs), train_loss, label="Training Loss")
        ax1.plot(range(0, epochs), validation_loss, label="Validation Loss")
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training vs Validation Loss')
        ax1.legend()
        ax2.plot(range(0, epochs), train_oacc, label="Training Accuracy")
        ax2.plot(range(0, epochs), validation_oacc, label="Validation Accuracy")
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Overall Accuracy')
        ax2.set_title('Training vs Validation Accuracy')
        ax2.legend()
        ax3.plot(range(0, epochs), train_wiou, label="Training wmIoU")
        ax3.plot(range(0, epochs), validation_wiou, label="Validation wmIoU")
        ax3.set_xlabel('Epochs')
        ax3.set_ylabel('wmIoU')
        ax3.set_title('Training vs Validation weighted mIoU')
        ax3.legend()
        ax4.plot(range(0, epochs), train_miou, label="Training mIoU")
        ax4.plot(range(0, epochs), validation_miou, label="Validation mIoU")
        ax4.set_xlabel('Epochs')
        ax4.set_ylabel('mIoU')
        ax4.set_title('Training vs Validation mIoU')
        ax4.legend()

        fig.savefig('./Corine/final_tunning/model_21.png')

if __name__ == "__main__":
    #Load dataset
    DATA_DIR = '/home/usuaris/imatge/manel.davins/Corine/data'
    x_train_dir = os.path.join(DATA_DIR, 'train_images')
    y_train_dir = os.path.join(DATA_DIR, 'train_labels')
    x_valid_dir = os.path.join(DATA_DIR, 'val_images')
    y_valid_dir = os.path.join(DATA_DIR, 'val_labels')

    def transformation():  
        transform = [
            #A.RandomCrop(width=512, height=512,  p=1),
            A.Flip(),
        ]
        return A.Compose(transform)
    
    train_data = dataset_corine.DatasetCorine(x_train_dir, y_train_dir, augmentation = None)
    val_data = dataset_corine.DatasetCorine(x_valid_dir, y_valid_dir, augmentation = None)
    
    train_loader = DataLoader(train_data, batch_size=8, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=4)

    #Define my model:
    #DeeplabV3+
    model = DeepLabV3Plus(encoder_name = "resnet101",
            encoder_depth = 5,
            encoder_weights = "imagenet",
            encoder_output_stride = 16,
            decoder_channels = 256,
            decoder_atrous_rates = (6, 12, 18),
            in_channels = 4,
            classes = 14)

    model.to(device);
    
    #Metrics:
    mIoU = metrics.Jaccard()
    accuracy = metrics.OAAcc()

    #Define optimizer
    lr = 0.0005
    optimizer = optim.Adam(model.parameters(), lr, weight_decay=0.0005)

    #Define Loss
    class_weights = torch.FloatTensor([29.793214  , 15.26759907,  6.01552005, 12.33477792,  4.49127644,
        5.24112275,  7.31527147, 12.66678842,  7.68805826, 29.15104065,
       42.50016961, 20.17288139, 47.35312919, 31.9629094 ]).cuda()
    
    ce = nn.CrossEntropyLoss()
    weighted_ce = nn.CrossEntropyLoss(weight=class_weights)
    dice_loss = dice_loss_function.DiceLoss()
    


    #Define scheduler
    #cyclic_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-4, max_lr = 1e-3)
    #step_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=140, gamma=0.1)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.3, patience=5, verbose=True)
    

    train(model, train_loader, val_loader, dice_loss, optimizer, scheduler, epochs=150, plot=True)

    
    