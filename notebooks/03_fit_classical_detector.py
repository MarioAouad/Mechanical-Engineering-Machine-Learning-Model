import os
import torch
import numpy as np
import joblib
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader, TensorDataset
from run_cnn_training import CNNAutoencoder, extract_patches

# ==========================================================================
# CONFIGURATION
# ==========================================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models_cnn")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 256

def main():
    machine_types = sorted([d for d in os.listdir(PROCESSED_DIR)
                            if os.path.isdir(os.path.join(PROCESSED_DIR, d))])
    
    print(f"Device: {device}")
    print("=" * 60)
    print("PHASE 3: FITTING CLASSICAL DETECTOR ON EMBEDDINGS")
    print("=" * 60)

    for machine in machine_types:
        print(f"\nProcessing: {machine}")
        model_dir = os.path.join(MODELS_DIR, machine)
        if not os.path.exists(model_dir):
            print(f"  [SKIPPED] No CNN model found for {machine}")
            continue

        # 1. Load CNN Model
        model = CNNAutoencoder().to(device)
        model.load_state_dict(torch.load(os.path.join(model_dir, "best_model.pth"), map_location=device, weights_only=False))
        model.eval()

        # 2. Load Normal Training Data
        X_train = np.load(os.path.join(PROCESSED_DIR, machine, "X_train.npy"))
        train_patches = np.concatenate([extract_patches(s) for s in X_train])
        
        # 3. Create DataLoader for extraction
        x_tensor = torch.FloatTensor(train_patches).unsqueeze(1)
        dataset = TensorDataset(x_tensor)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

        # 4. Extract Embeddings
        embeddings = []
        with torch.no_grad():
            for (batch,) in loader:
                batch = batch.to(device)
                emb = model(batch, return_embedding=True)
                embeddings.append(emb.cpu().numpy())
        
        embeddings = np.concatenate(embeddings, axis=0)
        print(f"  Extracted embeddings shape: {embeddings.shape}")

        # 5. Fit KNN
        knn = NearestNeighbors(n_neighbors=5, metric='cosine')
        knn.fit(embeddings)
        
        print(f"  KNN fitted on {embeddings.shape[0]} normal samples.")

        # 6. Save Classical Model
        knn_path = os.path.join(model_dir, "knn.save")
        joblib.dump(knn, knn_path)
        print(f"  Saved KNN to {knn_path}")

if __name__ == "__main__":
    main()
