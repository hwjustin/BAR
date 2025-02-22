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
from utils.plot import plot_concept_accuracy
from explanations.concept import CAR, CAV
from explanations.feature import CARFeatureImportance
from models.brain import BrainNetwork
import matplotlib.pyplot as plt

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
    """
    Train a classifier to distinguish between concept positive and negative samples.

    Args:
        random_seed (int): Seed for reproducibility.
        plot (bool): Whether to generate and save plots.
        save_dir (Path): Directory to save the results.
    """
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    # Set random seeds for reproducibility
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # Load positive and negative embeddings
    positive_path = Path("concept/positive_feature_1.4.pt")
    negative_path = Path("concept/negative_feature_1.4.pt")
    positive_embeddings = torch.load(positive_path)  # Shape: [1000, 256, 1024]
    negative_embeddings = torch.load(negative_path)  # Shape: [1000, 256, 1024]

    X_pos = vis_token_process(positive_embeddings, 1).numpy()  # Shape: [1000, 1024]
    X_neg = vis_token_process(negative_embeddings, 1).numpy()  # Shape: [1000, 1024]

    X_pos = X_pos.reshape(positive_embeddings.size(0), -1) # Shape: [1000, 1024]
    X_neg = X_neg.reshape(negative_embeddings.size(0), -1)  # Shape: [1000, 1024]
    # X_pos = positive_embeddings.mean(dim=1).numpy()  # Shape: [1000, 1024]
    # X_neg = negative_embeddings.mean(dim=1).numpy()  # Shape: [1000, 1024]

    print(X_pos.shape, X_neg.shape)
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
    logging.info(f"Test Accuracy: {acc:.4f}")

    # Prepare the results directory
    if not save_dir.exists():
        os.makedirs(save_dir)

    # Save the accuracy metric
    metrics_df = pd.DataFrame({'Test_Accuracy': [acc]})
    metrics_df.to_csv(save_dir / "metrics.csv", index=False)
    logging.info(f"Saved metrics to {save_dir / 'metrics.csv'}")

    # Generate and save plots if requested
    if plot:
        plot_concept_accuracy(save_dir, "bar_dataset", "bar")
        logging.info(f"Plots saved to {save_dir}")

def plot_feature_importance(attributions: np.ndarray, save_dir: Path, title: str = "Voxel Feature Importance") -> None:
    """
    Plot feature importance scores for all voxels.
    
    Args:
        attributions (np.ndarray): Array of attribution scores for each voxel
        save_dir (Path): Directory to save the plot
        title (str): Title for the plot
    """
    plt.figure(figsize=(15, 5))
    
    # Calculate mean absolute attribution across samples
    mean_attributions = np.abs(attributions).mean(axis=0)
    
    # Create the plot
    plt.plot(range(len(mean_attributions)), mean_attributions)
    plt.title(title)
    plt.xlabel("Voxel Index")
    plt.ylabel("Average Absolute Attribution Score")
    
    # Add a horizontal line at y=0 for reference
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.3)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(save_dir / "voxel_importance.png")
    plt.close()
    
    # Also save top 100 most important voxels
    top_indices = np.argsort(mean_attributions)[-100:]
    top_values = mean_attributions[top_indices]
    
    plt.figure(figsize=(12, 6))
    plt.bar(range(100), top_values[::-1])
    plt.title("Top 100 Most Important Voxels")
    plt.xlabel("Voxel Rank")
    plt.ylabel("Attribution Score")
    plt.tight_layout()
    plt.savefig(save_dir / "top_100_voxels.png")
    plt.close()
    
    # Save the top voxel indices and their scores
    top_voxels_df = pd.DataFrame({
        'Voxel_Index': top_indices[::-1],
        'Attribution_Score': top_values[::-1]
    })
    top_voxels_df.to_csv(save_dir / "top_voxels.csv", index=False)

def feature_importance(
    random_seed: int,
    batch_size: int,
    plot: bool,
    save_dir: Path = Path.cwd() / "results/bar/feature_importance",
) -> None:
    """
    Compute feature importance using CAR classifier for the bar dataset.
    
    Args:
        random_seed (int): Random seed for reproducibility
        batch_size (int): Batch size for processing
        plot (bool): Whether to generate and save plots
        save_dir (Path): Directory to save results
    """
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    if not save_dir.exists():
        os.makedirs(save_dir)

    # Load the brain network model
    num_voxels = 15724  # From fMRI configuration
    out_dim = 768
    model = BrainNetwork(in_dim=num_voxels, out_dim=out_dim)
    
    # Load checkpoint
    outdir = 'train_logs/prior_1x768_final_subj01_bimixco_softclip_byol'
    ckpt_path = os.path.join(outdir, 'last.pth')
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.to(device)
    model.eval()

    # Load positive and negative embeddings
    positive_path = Path("concept/positive_embedding.pt")
    negative_path = Path("concept/negative_embedding.pt")
    positive_embeddings = torch.load(positive_path)
    negative_embeddings = torch.load(negative_path)

    # Reshape embeddings and create labels
    X_pos = positive_embeddings.view(positive_embeddings.size(0), -1).numpy()
    X_neg = negative_embeddings.view(negative_embeddings.size(0), -1).numpy()
    y_pos = np.ones(X_pos.shape[0])
    y_neg = np.zeros(X_neg.shape[0])

    # Combine the data
    X = np.concatenate((X_pos, X_neg), axis=0)
    y = np.concatenate((y_pos, y_neg), axis=0)

    # Train CAR classifier
    car = CAR(device)
    car.fit(X, y)
    car.tune_kernel_width(X, y)

    # Load voxel data
    positive_voxels = np.load('concept/positive_voxels.npy')
    negative_voxels = np.load('concept/negative_voxels.npy')
    
    # Combine voxels and convert to torch tensors
    voxels = np.concatenate((positive_voxels, negative_voxels), axis=0)
    voxels = torch.from_numpy(voxels).to(device)
    
    # Create labels for voxels (same order as the embeddings)
    voxel_labels = np.concatenate((np.ones(positive_voxels.shape[0]), 
                                 np.zeros(negative_voxels.shape[0])))
    voxel_labels = torch.from_numpy(voxel_labels).to(device)  # Convert to torch tensor
    
    # Create data loader
    dataset = torch.utils.data.TensorDataset(voxels, voxel_labels)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Compute feature importance
    attribution_method = CARFeatureImportance(
        "Integrated Gradient",
        car,
        model,
        device
    )
    
    # Use mean voxel activation as baseline
    baselines = torch.mean(voxels, dim=0, keepdim=True).to(device)
    
    # Calculate attributions
    attributions = attribution_method.attribute(
        data_loader, 
        baselines=baselines,
        internal_batch_size=batch_size
    )

    # Save results
    np.save(save_dir / "attributions.npy", attributions)
    logging.info(f"Saved attributions to {save_dir / 'attributions.npy'}")

    if plot:
        plot_feature_importance(attributions, save_dir)
        logging.info(f"Feature importance plots saved to {save_dir}")

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
