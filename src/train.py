import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import albumentations as A
import time
from torch.utils.data import DataLoader

from src.models.deeplabV3plus.deeplabv3 import DeepLabV3Plus
from  src.models.unet import *
from src.utils import metrics
from src.utils import losses
from src.data.dataset import DatasetCorine
from utils.utils import load_config


def train_epoch(model, train_loader, criterion, optimizer):
    epoch_loss = 0
    epoch_wmiou = 0
    epoch_miou = 0
    epoch_acc = 0

    with tqdm(total=len(train_loader), desc="Train", unit="batch") as pbar:
        for batch_idx, batch in enumerate(train_loader):
            images, true_masks = batch["img"], batch["label"]
            images = images.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=torch.long)

            optimizer.zero_grad()
            pred_masks = model(images)
            loss = criterion(pred_masks, true_masks)
            epoch_loss += loss.item()

            wmiou, miou, _ = metrics.Jaccard()(pred_masks, true_masks)
            epoch_wmiou += wmiou.cpu().detach().numpy()
            epoch_miou += miou.cpu().detach().numpy()
            acc, _ = metrics.OAAcc()(pred_masks, true_masks)
            epoch_acc += acc.cpu().detach().numpy()

            loss.backward()
            optimizer.step()

            pbar.set_postfix(**{'loss': epoch_loss / (batch_idx + 1),
                                'wIoU': epoch_wmiou / (batch_idx + 1),
                                'acc': epoch_acc / (batch_idx + 1),
                                'mIoU': epoch_miou / (batch_idx + 1)
                                })
            pbar.update(1)  
        
    return (
        epoch_loss / (batch_idx + 1),
        epoch_wmiou / (batch_idx + 1),
        epoch_acc / (batch_idx + 1),
        epoch_miou / (batch_idx + 1),
    )

def validate_epoch(model, val_loader, criterion):
    model.eval()
    val_loss = 0
    val_wmiou = 0
    val_miou = 0
    val_acc = 0

    with torch.no_grad():
        with tqdm(total=len(val_loader), desc="Validation", unit="batch") as pbar:
            for batch_idx, batch in enumerate(val_loader):
                images, true_masks = batch["img"], batch["label"]
                images = images.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                pred_masks = model(images)
                val_loss += criterion(pred_masks, true_masks).item()

                wmiou, miou, _ = metrics.Jaccard()(pred_masks, true_masks)
                val_wmiou += wmiou.cpu().detach().numpy()
                val_miou += miou.cpu().detach().numpy()
                acc, _ = metrics.OAAcc()(pred_masks, true_masks)
                val_acc += acc.cpu().detach().numpy()

                pbar.set_postfix(**{'loss':val_loss / (batch_idx + 1),
                                'wIoU': val_wmiou / (batch_idx + 1),
                                'acc': val_acc / (batch_idx + 1),
                                'mIoU': val_miou / (batch_idx + 1)
                                })
            pbar.update(1)  

    return (
        val_loss / (batch_idx + 1),
        val_wmiou / (batch_idx + 1),
        val_acc / (batch_idx + 1),
        val_miou / (batch_idx + 1),
    )


def train(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs, plot=True):
    start = time.time()

    train_loss_history = []
    train_wiou_history = []
    train_oacc_history = []
    train_miou_history = []

    validation_loss_history = []
    validation_wiou_history = []
    validation_oacc_history = []
    validation_miou_history = []

    max_score = 0.0

    for epoch in range(epochs):
        
        # Train
        train_loss, train_wmiou, train_oacc, train_miou = train_epoch(
            model, train_loader, optimizer, criterion
        )
        train_loss_history.append(train_loss)
        train_wiou_history.append(train_wmiou)
        train_oacc_history.append(train_oacc)
        train_miou_history.append(train_miou)

        # Validation
        val_loss, val_wmiou, val_oacc, val_miou = validate_epoch(
            model, val_loader, criterion
        )
        validation_loss_history.append(val_loss)
        validation_wiou_history.append(val_wmiou)
        validation_oacc_history.append(val_oacc)
        validation_miou_history.append(val_miou)

        # Scheduler based on validation wmiou
        if scheduler is not None:
            scheduler.step(val_wmiou)
        
        # Checkpoint saving based on validation wmiou
        if max_score < val_wmiou: 
            max_score = val_wmiou

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss_history":train_loss_history,
                    "train_wiou_history":train_wiou_history,
                    "train_oacc_history":train_oacc_history,
                    "train_miou_history":train_miou_history,
                    "validation_loss_history":validation_loss_history,
                    "validation_wiou_history":validation_wiou_history,
                    "validation_oacc_history":validation_oacc_history,
                    "validation_miou_history":validation_miou_history
                },
                config["model_checkpoint"],
            )
            print(f"Epoch {epoch + 1}: Best model saved! (wIoU: {val_wmiou:.4f})")

    print('Finished Training!')
    stop = time.time()
    elapsed_time= time.strftime('%H:%M:%S', time.gmtime(stop-start))
    print(f"Training time: {elapsed_time}")

    # Plot training results
    if plot:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots( nrows=2, ncols=2,figsize=(15,10))  # create figure & 1 axis

        ax1.plot(range(0, epochs), train_loss, label="Training Loss")
        ax1.plot(range(0, epochs), validation_loss_history, label="Validation Loss")
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training vs Validation Loss')
        ax1.legend()
        ax2.plot(range(0, epochs), train_oacc, label="Training Accuracy")
        ax2.plot(range(0, epochs), validation_oacc_history, label="Validation Accuracy")
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Overall Accuracy')
        ax2.set_title('Training vs Validation Accuracy')
        ax2.legend()
        ax3.plot(range(0, epochs), train_wiou_history, label="Training wmIoU")
        ax3.plot(range(0, epochs), validation_wiou_history, label="Validation wmIoU")
        ax3.set_xlabel('Epochs')
        ax3.set_ylabel('wmIoU')
        ax3.set_title('Training vs Validation weighted mIoU')
        ax3.legend()
        ax4.plot(range(0, epochs), train_miou, label="Training mIoU")
        ax4.plot(range(0, epochs), validation_miou_history, label="Validation mIoU")
        ax4.set_xlabel('Epochs')
        ax4.set_ylabel('mIoU')
        ax4.set_title('Training vs Validation mIoU')
        ax4.legend()

        fig.savefig(config["training_plot"])

if __name__ == "__main__":

    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_data = DatasetCorine(
        config["train_images_dir"],
        config["train_labels_dir"],
        augmentation = None
    )

    val_data = DatasetCorine(
        config["val_images_dir"],
        config["val_labels_dir"], 
        augmentation = None
    )
    
    train_loader = DataLoader(
        train_data, 
        batch_size = config["batch_size"],
        shuffle = True,
        num_workers = 4
    )
    
    val_loader = DataLoader(
        val_data, 
        batch_size = config["batch_size"], 
        shuffle = False, 
        num_workers = 4
    )

    # Model instantiation: either UNet or DeepLabV3+
    #model = UNet(...)
    model = DeepLabV3Plus(
        encoder_name=config["deeplabv3plus"]["encoder_name"],
        encoder_depth=5,
        encoder_weights=config["deeplabv3plus"]["encoder_weights"],
        encoder_output_stride=16,
        decoder_channels=256,
        decoder_atrous_rates=(6, 12, 18),
        in_channels=config["deeplabv3plus"]["in_channels"],
        classes=config["deeplabv3plus"]["num_classes"],
        )
    model.to(device);

    # Optimizer
    optimizer = optim.Adam(model.parameters(), config["lr"], weight_decay=config["weight_decay"])
    
    # Loss 
    class_weights = torch.FloatTensor(config["class_weights"])

    if config["loss_function"] == "cross_entropy":
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    elif config["loss_function"] == "dice":
        criterion = losses.DiceLoss()
    elif config["loss_function"] == "focal":
        criterion = losses.FocalLoss(alpha = config["class_weights"])
    elif config["loss_function"] == "jaccard":
        criterion = losses.JaccardLoss()
    else:
        raise ValueError(f"Invalid loss function: {config['loss_function']}")

    # Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max', 
        factor=0.3, 
        patience=5, 
        verbose=True)
    
    train(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=config["epochs"], plot=True)