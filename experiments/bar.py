import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pathlib import Path
import argparse
import logging
from utils.plot import plot_concept_accuracy_bar, plot_feature_importance
from explanations.concept import CAR, CAV
from explanations.feature import CARFeatureImportance
from models.umbrae.model import BrainX, BrainXS
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

CATEGORIES = [
    'person', 'chair', 'car', 'dining table', 'cup', 'bottle', 'bowl', 'handbag', 'truck', 'bench',
    'book', 'backpack', 'sink', 'clock', 'dog', 'sports ball', 'cat', 'potted plant', 'cell phone',
    'surfboard', 'knife', 'tie', 'skis', 'bus', 'traffic light', 'tv', 'bed', 'train', 'umbrella',
    'toilet', 'tennis racket', 'spoon', 'couch', 'bird', 'skateboard', 'airplane', 'motorcycle',
    'boat', 'vase', 'bicycle', 'fork', 'pizza', 'oven', 'giraffe', 'laptop'
]

# Utils for flattening the image features
def vis_token_process(image_features, vis_token_scale):
    N, H_W, C = image_features.shape
    H = W = int(H_W ** 0.5)
    reshaped_tensor = image_features.view(N, H, W, C)
    reshaped_tensor = reshaped_tensor.permute(0, 3, 1, 2)
    reshaped_tensor = reshaped_tensor.float()
    pool_size = stride = int( np.sqrt(H_W / vis_token_scale) )
    pooled_tensor = F.avg_pool2d(reshaped_tensor, kernel_size=pool_size, stride=stride)
    image_features = pooled_tensor.permute(0, 2, 3, 1)
    image_features = image_features.reshape(N, -1, C)
    return image_features

def concept_accuracy(random_seed: int, plot: bool, save_dir: Path = Path.cwd() / "results/bar/concept_accuracy") -> None:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    # Set random seeds for reproducibility
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # Prepare the results directory
    if not save_dir.exists():
        os.makedirs(save_dir)

    accuracies = {}

    for category in CATEGORIES:
        # Load positive and negative embeddings for each category
        positive_path = Path(f"concept/feature/{category}_positive_features.pt")
        negative_path = Path(f"concept/feature/{category}_negative_features.pt")
        positive_embeddings = torch.load(positive_path)
        negative_embeddings = torch.load(negative_path)

        X_pos = vis_token_process(positive_embeddings, 1).numpy()
        X_neg = vis_token_process(negative_embeddings, 1).numpy()

        X_pos = X_pos.reshape(positive_embeddings.size(0), -1)
        X_neg = X_neg.reshape(negative_embeddings.size(0), -1)

        # Create labels: 1 for positive, 0 for negative
        y_pos = np.ones(X_pos.shape[0])
        y_neg = np.zeros(X_neg.shape[0])

        # Combine the data
        X = np.concatenate((X_pos, X_neg), axis=0)
        y = np.concatenate((y_pos, y_neg), axis=0)

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_seed, stratify=y
        )

        # Initialize and train the CAR classifier
        car = CAR(device)
        car.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = car.predict(X_test)

        # Calculate accuracy
        acc = accuracy_score(y_test, y_pred)
        accuracies[category] = acc
        logging.info(f"{category} Test Accuracy: {acc:.4f}")

    # Save all accuracies to a single CSV file
    metrics_df = pd.DataFrame(list(accuracies.items()), columns=['Category', 'Test_Accuracy'])
    metrics_df.to_csv(save_dir / "all_categories_metrics.csv", index=False)
    logging.info(f"Saved all metrics to {save_dir / 'all_categories_metrics.csv'}")

    # Generate and save plots if requested
    if plot:
        plot_concept_accuracy_bar(accuracies, save_dir)
        logging.info(f"Plots saved to {save_dir}")



def feature_importance(random_seed: int, batch_size: int, plot: bool, save_dir: Path = Path.cwd() / "results/bar/feature_importance") -> None:
    """
    Compute feature importance for the Bar dataset using a trained model.

    Args:
        random_seed (int): Seed for reproducibility.
        batch_size (int): Batch size for processing.
        plot (bool): Whether to generate and save plots.
        save_dir (Path): Directory to save the results.
    """
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    # Set random seeds for reproducibility
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # Load voxel data
    positive_voxel_path = Path("concept/positive_voxels.npy")
    negative_voxel_path = Path("concept/negative_voxels.npy")
    positive_voxels = np.load(positive_voxel_path)
    negative_voxels = np.load(negative_voxel_path)

    # Convert to torch tensors and take mean across repetitions
    positive_voxels = torch.from_numpy(positive_voxels).float().mean(axis=1)
    negative_voxels = torch.from_numpy(negative_voxels).float().mean(axis=1)

    # Initialize model
    voxels_per_subj = {1: 15724, 2: 14278, 3: 15226, 4: 13153, 5: 13039, 6: 17907, 7: 12682, 8: 14386}
    num_voxels = voxels_per_subj.get(5)  # Assuming subject 5, adjust as needed

    kwargs = {'hidden_dim': 1024, 'out_dim': 1024, 'num_latents': 256, 
              'use_norm': False, 'use_token': False}
    voxel2emb = BrainX(**kwargs)
    voxel2emb.to(device)

    # Load checkpoint
    checkpoint = torch.load("train_logs_umbrae/brainx-v-1-4/last.pth", map_location='cpu', weights_only=False)
    voxel2emb.load_state_dict(checkpoint['model_state_dict'], strict=False)
    voxel2emb.eval()

    # Generate embeddings
    def generate_embeddings(voxels):
        embeddings = []
        with torch.no_grad():
            for i in range(0, len(voxels), batch_size):
                batch = voxels[i:i + batch_size].to(device)
                with torch.cuda.amp.autocast():
                    emb = voxel2emb(batch)
                embeddings.append(emb.cpu())
        return torch.cat(embeddings, dim=0)

    # Generate embeddings for positive and negative voxels
    X_pos = generate_embeddings(positive_voxels)
    X_neg = generate_embeddings(negative_voxels)

    X_pos = vis_token_process(X_pos, 1).numpy()  # Shape: [1000, 1024]
    X_neg = vis_token_process(X_neg, 1).numpy()  # Shape: [1000, 1024]

    X_pos = X_pos.reshape(X_pos.shape[0], -1) # Shape: [1000, 1024]
    X_neg = X_neg.reshape(X_neg.shape[0], -1)  # Shape: [1000, 1024]

    # Create labels: 1 for positive, 0 for negative
    y_pos = np.ones(X_pos.shape[0])
    y_neg = np.zeros(X_neg.shape[0])

    # Combine the data
    X = np.concatenate((X_pos, X_neg), axis=0)
    y = np.concatenate((y_pos, y_neg), axis=0)

    # Initialize and train the CAR classifier
    car = CAR(device)
    car.fit(X, y)
    # car.tune_kernel_width(X, y)

    # Create a new dataset for feature importance
    X_voxels = torch.cat((positive_voxels, negative_voxels), dim=0)
    y_embeddings = torch.cat((torch.tensor(X_pos), torch.tensor(X_neg)), dim=0)
    test_dataset = TensorDataset(X_voxels, y_embeddings)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    # Compute feature importance
    attribution_method = CARFeatureImportance("Integrated Gradient", car, voxel2emb, device)
    attributions = attribution_method.attribute(test_loader, baselines=torch.zeros((1, 15724)).to(device))

    # Save the feature importance results
    if not save_dir.exists():
        os.makedirs(save_dir)
    np.savez(save_dir / "attributions.npz", attributions=attributions)
    logging.info(f"Saved feature importance to {save_dir / 'attributions.npz'}")

    # Generate and save plots if requested
    if plot:
        plot_feature_importance(attributions, save_dir)
        logging.info(f"Plots saved to {save_dir}")

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train and evaluate concept classifier for Bar dataset.")
    parser.add_argument("--name", type=str, default="concept_accuracy",
                       help="Name of the experiment to run (concept_accuracy or feature_importance)")
    parser.add_argument("--random_seed", type=int, default=42, 
                       help="Random seed for reproducibility.")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size for processing")
    parser.add_argument("--plot", action="store_true", 
                       help="Whether to generate plots.")
    args = parser.parse_args()

    # Execute the appropriate function based on the experiment name
    if args.name == "concept_accuracy":
        concept_accuracy(args.random_seed, args.plot)
    elif args.name == "feature_importance":
        feature_importance(args.random_seed, args.batch_size, args.plot)
    else:
        raise ValueError(f"Unknown experiment name: {args.name}")
