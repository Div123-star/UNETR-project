import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from monai.networks.nets import UNETR
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from monai.transforms import Activations, AsDiscrete
from torch.utils.data import DataLoader, random_split

from Transforms.chaos_transform import chaos_train_transforms, chaos_val_transforms
from DataLoaders.chaos_dataloader import NPZVolumeDataset

# === Config ===
DATA_PATH = "/Users/dibya/PycharmProjects/Data/chaos_npz"
SAVE_PATH = "./outputs/unetr_chaos.pth"
PLOT_DIR = "./outputs/plots"
NUM_CLASSES = 5  # CHAOS has 4 organs + background
IMG_SIZE = (256, 256, 64)
BATCH_SIZE = 2
EPOCHS = 100
LR = 1e-4

os.makedirs(PLOT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# === Dataset & Loader ===
dataset = NPZVolumeDataset(DATA_PATH, transforms=chaos_train_transforms)
train_ds, val_ds = random_split(dataset, [int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))])
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

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

# === Loss & Optimizer ===
weights = torch.ones(NUM_CLASSES).to(device)
weights[0] = 0.1  # reduce background importance
loss_fn = DiceCELoss(to_onehot_y=False, softmax=True, weight=weights)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)

# === Post-processing and Metrics ===
post_pred = Activations(softmax=True)
post_label = AsDiscrete()  # No one-hot assumed
dice_metric = DiceMetric(include_background=True, reduction="none", get_not_nans=True)

# === Logs ===
train_loss_hist, val_loss_hist, val_dice_hist = [], [], []
best_dice = 0.0

# === Training Loop ===
for epoch in range(1, EPOCHS + 1):
    print(f"\nEpoch {epoch}/{EPOCHS}")
    model.train()
    epoch_loss = 0

    for step, batch in enumerate(train_loader, start=1):
        image, label = batch["image"].to(device), batch["label"].to(device)
        output = model(image)
        loss = loss_fn(output, label)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        epoch_loss += loss.item()
        print(f"  Running Batch {step}/{len(train_loader)} - Loss: {loss.item():.4f}")

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
                print("Validation shape check:")
                print(f"  Input: {image.shape}, Label: {label.shape}, Output: {output.shape}")

    val_loss /= len(val_loader)
    val_loss_hist.append(val_loss)
    val_dice_tensor, _ = dice_metric.aggregate()
    val_dice = val_dice_tensor.mean().item()
    val_dice_hist.append(val_dice)
    print(f"Val Loss: {val_loss:.4f}, Mean Dice: {val_dice:.4f}")

    scheduler.step(val_loss)

    # Save best model
    if val_dice > best_dice:
        best_dice = val_dice
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"✅ Saved new best model at {SAVE_PATH}")

# === Save logs ===
df = pd.DataFrame({
    "epoch": list(range(1, EPOCHS + 1)),
    "train_loss": train_loss_hist,
    "val_loss": val_loss_hist,
    "val_dice": val_dice_hist
})
df.to_csv(os.path.join(PLOT_DIR, "chaos_training_logs.csv"), index=False)

# === Plot Loss and Dice ===
plt.figure(figsize=(10, 6))
plt.plot(df["epoch"], df["train_loss"], label="Train Loss")
plt.plot(df["epoch"], df["val_loss"], label="Val Loss")
plt.plot(df["epoch"], df["val_dice"], label="Val Dice")
plt.xlabel("Epoch")
plt.ylabel("Value")
plt.title("CHAOS UNETR Training")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "chaos_training_plot.png"))
plt.close()

print("🎉 Training complete. Logs and plots saved.")
