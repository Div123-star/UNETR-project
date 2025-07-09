import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from torch.utils.data import DataLoader, random_split
from monai.networks.nets import UNETR
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import Activations, AsDiscrete

from Transforms.amos_transform import amos_train_transforms, amos_val_transforms
from DataLoaders.amos_dataloader import NPZVolumeDataset

# === Configuration ===
NUM_CLASSES = 15
IMG_SIZE = (256, 256, 64)
BATCH_SIZE = 2
EPOCHS = 100
LEARNING_RATE = 1e-4
AMOS_DATA_FOLDER = "/Users/dibya/PycharmProjects/Data/amos_npz"
SAVE_PATH = "/Users/dibya/PycharmProjects/UNETR/outputs/unetr_amos_scratch.pth"
PLOT_DIR = "/Users/dibya/PycharmProjects/UNETR/outputs/plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# === Device ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# === Load Dataset ===
print("Loading dataset...")
dataset = NPZVolumeDataset(AMOS_DATA_FOLDER, transforms=amos_train_transforms)
train_ds, val_ds = random_split(dataset, [int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

# === Shape Check ===
print("\n🔍 Sample shape inspection from training loader:")
sample_batch = next(iter(train_loader))
image, label = sample_batch["image"], sample_batch["label"]
print(f"Raw input shape: {image.shape}")
print(f"Label shape:     {label.shape}")
print(f"Training samples: {len(train_ds)}, Validation samples: {len(val_ds)}")

# === Model ===
model = UNETR(
    in_channels=1,
    out_channels=NUM_CLASSES,
    img_size=IMG_SIZE,
    feature_size=16,
    hidden_size=768,
    mlp_dim=3072,
    num_heads=12,
    norm_name='instance',
    res_block=True
).to(device)

print("\nModel Summary:")
print(model)

weights = torch.ones(NUM_CLASSES).to(device)
weights[0] = 0.1
loss_fn = DiceCELoss(to_onehot_y=False, softmax=True, weight=weights)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)

# === Post-processing & Metrics ===
post_pred = Activations(softmax=True)
post_label = AsDiscrete()
dice_metric = DiceMetric(include_background=True, reduction="none", get_not_nans=True)

# === Logs ===
train_loss_hist, val_loss_hist = [], []
val_dice_hist = []
all_class_dice = [[] for _ in range(NUM_CLASSES)]

# === Visualization Helper ===
def visualize_prediction(image, label, prediction, epoch=None, output_dir=None):
    from matplotlib.colors import ListedColormap

    color_list = [
        (0, 0, 0), (0, 255, 255), (255, 0, 0), (255, 255, 0), (0, 0, 255),
        (0, 255, 0), (255, 0, 255), (255, 165, 0), (128, 0, 128), (255, 192, 203),
        (165, 42, 42), (0, 255, 128), (0, 128, 128), (128, 128, 128), (173, 216, 230)
    ]
    cmap = ListedColormap(np.array(color_list[:15]) / 255.0)

    image_np = image[0, 0].detach().cpu().numpy()
    label_np = label[0].detach().cpu().numpy()
    pred_np = prediction[0].detach().cpu().numpy()

    label_mask = np.argmax(label_np, axis=0)
    pred_mask = np.argmax(pred_np, axis=0)
    z = label_mask.shape[-1] // 2

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(image_np[:, :, z], cmap="gray")
    axs[1].imshow(label_mask[:, :, z], cmap=cmap, vmin=0, vmax=14)
    axs[2].imshow(pred_mask[:, :, z], cmap=cmap, vmin=0, vmax=14)
    axs[0].set_title("Image"); axs[1].set_title("Label"); axs[2].set_title("Prediction")
    for ax in axs: ax.axis('off')
    plt.tight_layout()
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f"epoch_{epoch:03d}.png"))
    plt.close()

# === Training Loop ===
print("\nTraining Configuration:")
print(f"Model: UNETR, Input size: {IMG_SIZE}, Output classes: {NUM_CLASSES}")
print(f"Data path: {AMOS_DATA_FOLDER}")
print(f"Batch size: {BATCH_SIZE}, Epochs: {EPOCHS}")
print(f"Loss: DiceCELoss (weighted), Optimizer: Adam, LR: {LEARNING_RATE}")
print(f"Saving model to: {SAVE_PATH}")

for epoch in range(1, EPOCHS + 1):
    print(f"\nEpoch {epoch}/{EPOCHS}")
    model.train()
    epoch_loss = 0

    for step, batch in enumerate(train_loader, start=1):
        print(f"Running Batch {step}/{len(train_loader)}")

        image, label = batch["image"].to(device), batch["label"].to(device)
        output = model(image)

        loss = loss_fn(output, label)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        epoch_loss += loss.item()

    train_loss = epoch_loss / len(train_loader)
    train_loss_hist.append(train_loss)
    print(f"Train Loss: {train_loss:.4f}")

    # === Validation ===
    model.eval()
    val_loss = 0
    dice_metric.reset()

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            image, label = batch["image"].to(device), batch["label"].to(device)
            output = sliding_window_inference(image, roi_size=IMG_SIZE, sw_batch_size=1, predictor=model)
            loss = loss_fn(output, label)
            val_loss += loss.item()

            output_post = post_pred(output)
            label_post = post_label(label)
            dice_metric(y_pred=output_post, y=label_post)

            if i == 0:
                visualize_prediction(image, label_post, output_post, epoch=epoch, output_dir=PLOT_DIR)

    val_loss /= len(val_loader)
    val_loss_hist.append(val_loss)

    val_dice_tensor, _ = dice_metric.aggregate()
    val_dice_tensor = torch.nan_to_num(val_dice_tensor)
    val_dice = val_dice_tensor.mean().item()
    val_dice_hist.append(val_dice)

    for i in range(len(val_dice_tensor)):
        class_dice = val_dice_tensor[i].mean().item()
        all_class_dice[i].append(class_dice)

    for i in range(len(val_dice_tensor), NUM_CLASSES):
        all_class_dice[i].append(0.0)

    print(f"Val Loss: {val_loss:.4f}, Mean Dice: {val_dice:.4f}")
    scheduler.step(val_loss)

    # === Save model every epoch ===
    torch.save(model, SAVE_PATH)
    print(f"Model saved to: {SAVE_PATH}")

# === Save Logs ===
log_df = pd.DataFrame({
    "epoch": list(range(1, EPOCHS + 1)),
    "train_loss": train_loss_hist,
    "val_loss": val_loss_hist,
    "val_dice": val_dice_hist
})
csv_path = os.path.join(PLOT_DIR, "training_logs.csv")
log_df.to_csv(csv_path, index=False)
print(f"Training log saved to: {csv_path}")

# === Plot overall loss & Dice ===
plt.figure(figsize=(10, 6))
sns.lineplot(x="epoch", y="train_loss", data=log_df, label="Train Loss")
sns.lineplot(x="epoch", y="val_loss", data=log_df, label="Val Loss")
sns.lineplot(x="epoch", y="val_dice", data=log_df, label="Val Dice")
plt.xlabel("Epoch")
plt.ylabel("Value")
plt.title("UNETR AMOS Training History")
plt.grid(True)
plt.legend()
plot_path = os.path.join(PLOT_DIR, "training_curve.png")
plt.savefig(plot_path)
plt.close()
print(f"Training curve plot saved to: {plot_path}")

# === Save and plot per-class Dice ===
class_names = [f"Class_{i}" for i in range(NUM_CLASSES)]
dice_df = pd.DataFrame({name: scores for name, scores in zip(class_names, all_class_dice)})
dice_df.insert(0, "epoch", list(range(1, EPOCHS + 1)))
dice_csv_path = os.path.join(PLOT_DIR, "dice_per_class.csv")
dice_df.to_csv(dice_csv_path, index=False)
print(f"Per-class Dice scores saved to: {dice_csv_path}")

plt.figure(figsize=(12, 8))
for name in class_names:
    sns.lineplot(x="epoch", y=name, data=dice_df, label=name)
plt.xlabel("Epoch")
plt.ylabel("Dice Score")
plt.title("Per-Class Dice Score Trends")
plt.grid(True)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
dice_plot_path = os.path.join(PLOT_DIR, "dice_per_class_plot.png")
plt.savefig(dice_plot_path)
plt.close()
print(f"Per-class Dice plot saved to: {dice_plot_path}")
