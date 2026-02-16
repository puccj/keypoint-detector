import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import cv2
import numpy as np
from PIL import Image
import os

# ==========================================
# 1. The Model Architecture (ResNet-18)
# ==========================================
class ResNet18(nn.Module):
    def __init__(self, num_keypoints=9, pretrained=True):
        super(ResNet18, self).__init__()
        
        # Load pre-trained ResNet18
        # weights='DEFAULT' loads the best available weights (ImageNet)
        weights = 'DEFAULT' if pretrained else None
        self.backbone = models.resnet18(weights=weights)
        
        # Determine the number of input features to the final layer
        # For ResNet18, this is typically 512
        in_features = self.backbone.fc.in_features
        
        # Replace the fully connected layer
        # Output is num_keypoints * 2 (x and y for each point)
        self.backbone.fc = nn.Linear(in_features, num_keypoints * 2)
        
        # Sigmoid ensures outputs are between 0 and 1 (relative image coordinates)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.backbone(x)
        x = self.sigmoid(x)
        return x

# ==========================================
# 2. Custom Dataset Loader
# ==========================================

class KeypointDataset(Dataset):
    """
    Images and labels should be placed in separate directories with matching filenames.
    """
    def __init__(self, images_dir, labels_dir, transform=None):
        self.image_dir = images_dir
        self.label_dir = labels_dir
        self.transform = transform
        self.images = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        label_file = os.path.splitext(self.images[idx])[0] + '.txt'
        label_path = os.path.join(self.label_dir, label_file)
        
        image = Image.open(img_path).convert('RGB')
        label = []  # List to hold keypoints
        with open(label_path, 'r') as f:
            for line in f:
                x, y = map(float, line.strip().split())
                label.append([x, y])
        
        # Normalize keypoints to [0, 1]
        w, h = image.size
        keypoints = np.array(label, dtype=np.float32)
        keypoints[:, 0] /= w
        keypoints[:, 1] /= h
        
        # Flatten to shape (N*2,)
        keypoints = keypoints.flatten()

        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(keypoints, dtype=torch.float32)

# ==========================================
# 3. Training & Inference Utils
# ==========================================

def save_checkpoint(model, optimizer, file_path = "checkpoints/last.pth"):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, file_path)

def load_checkpoint(model, optimizer, file_path = "checkpoints/last.pth"):
    if not os.path.isfile(file_path):
        print(f"Checkpoint file {file_path} does not exist.")
        return False
    
    checkpoint = torch.load(file_path)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return True

def train_model(model, train_loader, val_loader, num_epochs=10, lr=1e-4, save_interval=0, patience=0):
    """
    Trains the model and optionally saves predictions at specified intervals.

    Parameters
    ----------
    model : nn.Module
        The neural network model to train.
    train_loader : DataLoader
        DataLoader for the training dataset.
    val_loader : DataLoader
        DataLoader for the validation dataset.
    num_epochs : int, optional
        Number of epochs to train (default is 10).
    lr : float, optional
        Learning rate for the optimizer (default is 1e-4).
    save_interval : int, optional
        Interval (in epochs) at which to save predictions (default is 0, meaning no saving).
    patience : int, optional
        Number of epochs to wait before early stopping (default is 0, meaning no early stopping).
    
    Returns
    -------
    train_losses : list of float
        List of training losses for each epoch.
    val_losses : list of float
        List of validation losses for each epoch.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0

    # MSE Loss is standard for regression
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # --- Training Loop ---
    print(f"Starting training on {device}...")
    model.train()
    for epoch in range(num_epochs):

        # Train step
        running_loss = 0.0
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)

            predictions = model(images)
            loss = criterion(predictions, targets)
            running_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if save_interval > 0 and (epoch + 1) % save_interval == 0:
                # save predictions for visualization
                # Here you would typically save the input images, predicted keypoints, and true keypoints
                pass
        
        train_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.6f}")
        train_losses.append(train_loss)

        # Validation step
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device)
                predictions = model(images)
                loss = criterion(predictions, targets)
                val_running_loss += loss.item()

                if save_interval > 0 and (epoch + 1) % save_interval == 0:
                    # save predictions for visualization
                    # Here you would typically save the input images, predicted keypoints, and true keypoints
                    pass
        
        val_loss = val_running_loss / len(val_loader)
        print(f"Validation Loss: {val_loss:.6f}")
        val_losses.append(val_loss)

        # Save checkpoint at intervals
        if save_interval > 0 and (epoch + 1) % save_interval == 0:
            save_checkpoint(model, optimizer, file_path=f"checkpoints/epoch_{epoch+1}.pth")

        # Early stopping check
        if patience > 0:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered after {epoch + 1} epochs.")
                    break
        
    # Save the last checkpoiunt
    save_checkpoint(model, optimizer, file_path="checkpoints/last.pth")

    return train_losses, val_losses

def predict_image(model, image_path, original_dims=(1920, 1080)):
    """
    Runs inference on a single image and returns pixel coordinates.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    # Preprocess
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Load Image (using OpenCV for easy drawing later)
    img = cv2.imread(image_path)
    # For demo: create fake image
    # img = np.zeros((original_dims[1], original_dims[0], 3), dtype=np.uint8) 
    
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    img_tensor = transform(pil_img).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        output = model(img_tensor)
    
    # Post-process (0-1 to Pixels)
    preds = output.cpu().numpy().flatten()
    keypoints = []
    
    w, h = original_dims
    
    # Reshape flattened [x1, y1, x2, y2...] into pairs
    for i in range(0, len(preds), 2):
        px = int(preds[i] * w)
        py = int(preds[i+1] * h)
        keypoints.append((px, py))
        
    return keypoints

# ==========================================
# 4. Main Execution
# ==========================================
if __name__ == "__main__":
    # 1. Train
    trained_model = train_model()
    
    # 2. Predict
    points = predict_image(trained_model, "test_court.jpg", original_dims=(1280, 720))
    print("\nDetected Keypoints (Pixel Coords):")
    print(points)

    # For visualization, you would typically draw these points on the image using OpenCV
    img = cv2.imread("test_court.jpg")
    for (x, y) in points:
        cv2.circle(img, (x, y), 10, (0, 255, 0), -1)
    # cv2.imshow("Detected Keypoints", img)
    cv2.imwrite("detected_keypoints.jpg", img)  # Save the image with detected keypoints
    cv2.waitKey(0)
    cv2.destroyAllWindows()