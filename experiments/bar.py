import torch
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
    positive_path = Path("concept/positive_embedding.pt")
    negative_path = Path("concept/negative_embedding.pt")
    positive_embeddings = torch.load(positive_path)
    negative_embeddings = torch.load(negative_path)

    # Reshape embeddings to [num_samples, embedding_dim]
    X_pos = positive_embeddings.view(positive_embeddings.size(0), -1).numpy()  # Shape: [100, 768]
    X_neg = negative_embeddings.view(negative_embeddings.size(0), -1).numpy()  # Shape: [100, 768]

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

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train and evaluate concept classifier for Bar dataset.")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--plot", action="store_true", help="Whether to generate plots.")
    args = parser.parse_args()

    # Execute the concept_accuracy function with provided arguments
    concept_accuracy(args.random_seed, args.plot)
