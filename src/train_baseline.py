import os
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score



# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

TRAIN_DIR = os.path.join(ROOT, "processed_data", "train")
VAL_DIR   = os.path.join(ROOT, "processed_data", "val")

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Datasets
train_ds = datasets.ImageFolder(TRAIN_DIR, transform=transform)
val_ds   = datasets.ImageFolder(VAL_DIR, transform=transform)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=16, shuffle=False)

# Model (Baseline)
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Train (1–2 epochs baseline)
for epoch in range(2):
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            preds += out.argmax(1).cpu().tolist()
            labels += y.cpu().tolist()

    acc = accuracy_score(labels, preds)
    print(f"Epoch {epoch+1} | Val Accuracy: {acc:.4f}")

# Save model
os.makedirs(os.path.join(ROOT, "models"), exist_ok=True)
torch.save(model.state_dict(), os.path.join(ROOT, "models", "baseline_resnet18.pth"))
print("Baseline model trained & saved")
