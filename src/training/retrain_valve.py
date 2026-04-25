"""
Retrain ONLY valve with original V2 settings (bottleneck=128).
Saves to weights/valve/.
"""
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import os, copy, time, joblib

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

PATCH_WIDTH = 64; PATCH_STRIDE = 32; BATCH_SIZE = 128
LEARNING_RATE = 1e-4; WEIGHT_DECAY = 1e-5; NUM_EPOCHS = 150; PATIENCE = 25

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

class CNNAutoencoderV2(nn.Module):
    def __init__(self, bottleneck_dim=128):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(1,32,3,stride=2,padding=1),nn.BatchNorm2d(32),nn.ReLU(True))
        self.enc2 = nn.Sequential(nn.Conv2d(32,64,3,stride=2,padding=1),nn.BatchNorm2d(64),nn.ReLU(True))
        self.enc3 = nn.Sequential(nn.Conv2d(64,128,3,stride=2,padding=1),nn.BatchNorm2d(128),nn.ReLU(True))
        self.flatten_size = 128*16*8
        self.fc_enc = nn.Linear(self.flatten_size, bottleneck_dim)
        self.fc_dec = nn.Linear(bottleneck_dim, self.flatten_size)
        self.dec3 = nn.Sequential(nn.ConvTranspose2d(128,64,3,stride=2,padding=1,output_padding=1),nn.BatchNorm2d(64),nn.ReLU(True))
        self.dec2 = nn.Sequential(nn.ConvTranspose2d(64,32,3,stride=2,padding=1,output_padding=1),nn.BatchNorm2d(32),nn.ReLU(True))
        self.dec1 = nn.ConvTranspose2d(32,1,3,stride=2,padding=1,output_padding=1)
    def forward(self, x, return_embedding=False):
        e1=self.enc1(x); e2=self.enc2(e1); e3=self.enc3(e2)
        flat=e3.view(e3.size(0),-1); emb=self.fc_enc(flat)
        if return_embedding: return emb
        d=self.fc_dec(emb).view(-1,128,16,8)
        return self.dec1(self.dec2(self.dec3(d)))

def extract_patches(spec, width=PATCH_WIDTH, stride=PATCH_STRIDE):
    _, W = spec.shape
    patches = []
    for s in range(0, W-width+1, stride): patches.append(spec[:, s:s+width])
    if patches and (W-width)%stride!=0: patches.append(spec[:, W-width:W])
    if not patches and W>=width: patches.append(spec[:, :width])
    return np.array(patches)

# Load valve data
X_train = np.load(os.path.join("data","processed_v2","valve","X_train.npy"))
X_val = np.load(os.path.join("data","processed_v2","valve","X_val.npy"))
print(f"Valve data: train={X_train.shape}, val={X_val.shape}")

train_patches = np.concatenate([extract_patches(s) for s in X_train])
val_patches = np.concatenate([extract_patches(s) for s in X_val])
print(f"Patches: train={train_patches.shape}, val={val_patches.shape}")

X_tr = torch.FloatTensor(train_patches).unsqueeze(1)
X_va = torch.FloatTensor(val_patches).unsqueeze(1)
train_loader = DataLoader(TensorDataset(X_tr), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(TensorDataset(X_va), batch_size=BATCH_SIZE, shuffle=False)

model = CNNAutoencoderV2(bottleneck_dim=128).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=7, min_lr=1e-6)

best_val = float('inf'); best_state = None; no_improve = 0
t0 = time.time()

for epoch in range(NUM_EPOCHS):
    model.train(); losses = []
    for (batch,) in train_loader:
        batch = batch.to(device)
        loss = criterion(model(batch), batch)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        losses.append(loss.item())
    model.eval(); val_losses = []
    with torch.no_grad():
        for (batch,) in val_loader:
            batch = batch.to(device)
            val_losses.append(criterion(model(batch), batch).item())
    avg_va = np.mean(val_losses)
    scheduler.step(avg_va)
    if avg_va < best_val:
        best_val = avg_va; best_state = copy.deepcopy(model.state_dict()); no_improve = 0; mk = 'Best'
    else:
        no_improve += 1; mk = f'{no_improve}/{PATIENCE}'
    if (epoch+1)%10==0 or no_improve==0 or no_improve>=PATIENCE:
        print(f"  Ep {epoch+1:3d}/{NUM_EPOCHS} | Va: {avg_va:.6f} | LR: {optimizer.param_groups[0]['lr']:.1e} | {mk}")
    if no_improve >= PATIENCE:
        print(f"  Early stop at epoch {epoch+1}"); break

model.load_state_dict(best_state)
elapsed = time.time() - t0
print(f"\nTraining done in {elapsed:.0f}s | Best val: {best_val:.6f}")

# Compute training stats for z-score scoring
model.eval()
loader = DataLoader(TensorDataset(X_tr), batch_size=BATCH_SIZE, shuffle=False)
all_errs = []
with torch.no_grad():
    for (batch,) in loader:
        batch = batch.to(device)
        recon = model(batch)
        errs = torch.mean((batch-recon)**2, dim=(1,2,3)).cpu().numpy()
        all_errs.append(errs)
all_errs = np.concatenate(all_errs)
train_mu, train_sigma = float(np.mean(all_errs)), float(np.std(all_errs))
print(f"Train MSE stats: mu={train_mu:.6f}, sigma={train_sigma:.6f}")

# Save to weights/
for d in ["weights/valve"]:
    os.makedirs(d, exist_ok=True)
    torch.save(best_state, os.path.join(d, "best_model.pth"))
    torch.save({
        'best_val': best_val, 'n_frames': X_train.shape[2],
        'train_mse_mean': train_mu, 'train_mse_std': train_sigma
    }, os.path.join(d, "metadata.pth"))
    print(f"Saved to {d}/")

print("\nDone! Run eval_best.py to see updated results.")
