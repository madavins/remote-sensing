import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sn

from src.models.deeplabV3plus.deeplabv3 import DeepLabV3Plus 
from src.models.unet import UNet
from src.utils import metrics
from src.data.dataset import DatasetCorine
from src.utils.utils import load_config

def evaluate():
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    val_data = DatasetCorine(
        config["val_images_dir"],
        config["val_labels_dir"], 
        augmentation = None
    )
   
    # DeepLabV3+ or U-Net
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
   
    checkpoint = torch.load(config["model_checkpoint"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()  

    true_masks = []
    predicted_masks = []

    with torch.no_grad():  
        for batch in val_data:
            image, true_mask = batch["img"], batch["label"]
            # Add batch dimension (B, C, H, W)
            image = image.unsqueeze(0).to(device)
            true_mask = true_mask.unsqueeze(0)

            output = model(image)  
            output = torch.softmax(output, dim=1) 
            output = output.detach().cpu() 

            output_onelayer = np.argmax(output.squeeze(0).numpy(), axis=0)  # Remove batch dimension and get argmax
            true_mask = true_mask.squeeze(0).numpy()

            true_masks.append(true_mask)
            predicted_masks.append(output_onelayer)

    y_true = np.array(true_masks).flatten() 
    y_pred = np.array(predicted_masks).flatten()

    # Metrics:
    wmiou, miou, class_iou = metrics.Jaccard()(torch.from_numpy(y_pred).unsqueeze(0).unsqueeze(0), torch.from_numpy(y_true).unsqueeze(0).unsqueeze(0))
    acc, _ = metrics.OAAcc()(torch.from_numpy(y_pred).unsqueeze(0).unsqueeze(0), torch.from_numpy(y_true).unsqueeze(0).unsqueeze(0))

    print(f"Weighted mIoU: {wmiou:.4f}")
    print(f"Mean IoU: {miou:.4f}")
    print(f"Overall Accuracy: {acc:.4f}")

    print("Classification Report:")
    classes = config["classes"]
    print(classification_report(y_true, y_pred, target_names=classes, zero_division=0))

    # Confusion matrix
    conf_mat = confusion_matrix(y_true, y_pred, normalize='true')
    df_cm = pd.DataFrame(conf_mat, index=classes, columns=classes)
    plt.figure(figsize=(10, 7))
    sn.heatmap(df_cm, annot=True, fmt=".2f", cmap="viridis", cbar=False)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(config["evaluation_plot"]) 

if __name__ == "__main__":
    evaluate()