import os

import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

try:
    from .dataset import GlomerulusDataset
    from .loss import BCEDiceLoss
    from .model import UNet
except ImportError:
    from dataset import GlomerulusDataset
    from loss import BCEDiceLoss
    from model import UNet


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    images_dir = "./data/images"
    masks_dir = "./data/masks"
    image_ext = ".png"
    mask_ext = ".tiff"

    # Smoke test settings for data/polarity verification
    batch_size = 1
    learning_rate = 1e-4
    num_epochs = 20
    test_size = 0.2
    save_path = "./weights/unet_glom_dice.pth"

    dataset = GlomerulusDataset(
        images_dir=images_dir,
        masks_dir=masks_dir,
        image_ext=image_ext,
        mask_ext=mask_ext,
        transform=None,
        foreground_value=0,
        debug=False,
        debug_max_prints=0,
    )

    indices = list(range(len(dataset)))
    train_idx, val_idx = train_test_split(indices, test_size=test_size, random_state=42, shuffle=True)
    train_data = Subset(dataset, train_idx)
    val_data = Subset(dataset, val_idx)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=0)

    model = UNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = BCEDiceLoss()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    best_val_loss = float("inf")
    first_batch_debug_printed = False

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Training"):
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)

            if not first_batch_debug_printed:
                probs = torch.sigmoid(outputs)
                pred_fg_ratio = (probs > 0.5).float().mean().item()
                print(
                    "[Debug] first batch: "
                    f"logits_mean={outputs.mean().item():.6f}, "
                    f"logits_min={outputs.min().item():.6f}, "
                    f"logits_max={outputs.max().item():.6f}, "
                    f"masks_mean={masks.mean().item():.6f}, "
                    f"probs_mean={probs.mean().item():.6f}, "
                    f"pred_fg_ratio@0.5={pred_fg_ratio:.6f}"
                )
                first_batch_debug_printed = True

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

        avg_train_loss = train_loss / len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Validation"):
                images = images.to(device)
                masks = masks.to(device)

                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item() * images.size(0)

        avg_val_loss = val_loss / len(val_loader.dataset)

        print(
            f"Epoch [{epoch + 1}/{num_epochs}], "
            f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            print(f"Model saved at epoch {epoch + 1} with val loss {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
