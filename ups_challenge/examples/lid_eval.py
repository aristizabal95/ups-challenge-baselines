import os
import pickle
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModel, AutoFeatureExtractor
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from ups_challenge.dataloaders.build_index import (
    build_lid_index,
    build_lid_index_splits,
)
from ups_challenge.dataloaders.lid import build_lid_dataset, collate_embeddings
from dotenv import load_dotenv

load_dotenv()


def train_and_evaluate(
    checkpoint_path=None,
    model_name="facebook/wav2vec2-base-960h",
    test_index_path="./data/lid_index_test.pkl",
    train_index_path="./data/lid_index_train.pkl",
    batch_size=8,
    max_train_samples=None,
    max_test_samples=None,
    num_epochs=10,
    learning_rate=0.01,
    device=None,
    hf_token=None,
    embedding_dim=None,
):
    """
    Train linear classifier on frozen embeddings and evaluate on test set.
    Uses streaming data through the frozen model to avoid storing all embeddings in memory.

    Args:
        checkpoint_path: Path to model checkpoint (optional)
        model_name: HuggingFace model name (used if checkpoint_path is None)
        test_index_path: Path to test index pickle file
        train_index_path: Path to train index pickle file
        batch_size: Batch size for training and evaluation
        max_train_samples: Maximum training samples (None for all)
        max_test_samples: Maximum test samples (None for all)
        num_epochs: Number of training epochs
        learning_rate: Learning rate for SGD optimizer
        device: torch device (auto-detected if None)
        hf_token: HuggingFace token
        embedding_dim: Embedding dimension (auto-detected from model config if None)
    """
    # Setup device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")

    # Load model
    print(f"Loading model: {model_name}")
    model = AutoModel.from_pretrained(model_name)

    # Load checkpoint if provided
    if checkpoint_path is not None:
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)

    # Move model to device and verify
    model.to(device)
    model.eval()

    # Verify model is on the correct device
    next_param_device = next(model.parameters()).device
    print(f"Model is on device: {next_param_device}")
    if device.type == "cuda" and next_param_device.type != "cuda":
        print("WARNING: Model is not on GPU despite CUDA being available!")

    # Freeze model
    for param in model.parameters():
        param.requires_grad = False

    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)

    # Check if train/test indices exist, build them if not
    train_index_path_obj = Path(train_index_path)
    test_index_path_obj = Path(test_index_path)

    if not train_index_path_obj.exists() or not test_index_path_obj.exists():
        print("Train or test index files not found. Building indices...")

        # Infer the full index path from train/test paths (same directory, lid_index.pkl)
        index_dir = train_index_path_obj.parent
        full_index_path = index_dir / "lid_index.pkl"

        # Build full index if it doesn't exist
        if not full_index_path.exists():
            print(f"Full index not found at {full_index_path}. Building full index...")
            build_lid_index(str(full_index_path), hf_token=hf_token)
        else:
            print(f"Found full index at {full_index_path}")

        # Build train/test splits
        print("Building train/test splits...")
        build_lid_index_splits(str(full_index_path))

        # Verify the files were created
        if not train_index_path_obj.exists() or not test_index_path_obj.exists():
            raise FileNotFoundError(
                f"Failed to create index files. Expected:\n"
                f"  - {train_index_path}\n"
                f"  - {test_index_path}"
            )

    # Load indices
    print(f"Loading train index from {train_index_path}")
    with open(train_index_path, "rb") as f:
        train_index = pickle.load(f)
    print(f"Train index has {len(train_index)} samples")

    print(f"Loading test index from {test_index_path}")
    with open(test_index_path, "rb") as f:
        test_index = pickle.load(f)
    print(f"Test index has {len(test_index)} samples")

    # Get HuggingFace token
    if hf_token is None:
        hf_token = os.getenv("HF_TOKEN")
    if hf_token is None:
        raise ValueError("HF_TOKEN is not set")

    # Get all unique languages from both indices to build label mapping
    all_langs = sorted(set(list(train_index.values()) + list(test_index.values())))
    lang_to_idx = {lang: idx for idx, lang in enumerate(all_langs)}
    idx_to_lang = {idx: lang for lang, idx in lang_to_idx.items()}
    num_classes = len(all_langs)

    print(f"Found {num_classes} languages: {all_langs}")

    # Get embedding dimension from model config or use provided value
    if embedding_dim is None:
        embedding_dim = model.config.hidden_size
        print(f"Embedding dimension (from model config): {embedding_dim}")
    else:
        print(f"Embedding dimension (provided): {embedding_dim}")

    # Create datasets that extract embeddings via mapped functions
    if max_train_samples is not None:
        print(f"Limiting training samples to {max_train_samples}")
    train_dataset = build_lid_dataset(
        train_index_path,
        model,
        feature_extractor,
        device,
        lang_to_idx,
        hf_token,
        max_samples=max_train_samples,
    )

    if max_test_samples is not None:
        print(f"Limiting test samples to {max_test_samples}")
    test_dataset = build_lid_dataset(
        test_index_path,
        model,
        feature_extractor,
        device,
        lang_to_idx,
        hf_token,
        max_samples=max_test_samples,
    )

    # Create data loaders with collate function
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=0,  # num_workers=0 for IterableDataset
        collate_fn=collate_embeddings,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=0,
        collate_fn=collate_embeddings,
    )

    # Create linear classification head
    classifier = nn.Linear(embedding_dim, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(classifier.parameters(), lr=learning_rate, momentum=0.9)

    # Train the classifier
    print("Training linear classifier (streaming data)...")
    classifier.train()

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        for embeddings, labels in progress_bar:
            embeddings = embeddings.to(device)
            labels = labels.to(device)

            # Forward pass
            logits = classifier(embeddings)
            loss = criterion(logits, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1
            progress_bar.set_postfix(
                {"loss": loss.item(), "avg_loss": epoch_loss / num_batches}
            )

        print(
            f"Epoch {epoch + 1} completed. Average loss: {epoch_loss / num_batches:.4f}"
        )

    # Evaluate on test set
    print("Evaluating on test set (streaming data)...")
    classifier.eval()
    all_test_preds = []
    all_test_labels = []

    with torch.no_grad():
        for embeddings, label_ids in tqdm(test_loader, desc="Evaluating"):
            embeddings = embeddings.to(device)
            label_ids = label_ids.to(device)

            logits = classifier(embeddings)
            pred_ids = torch.argmax(logits, dim=1)

            all_test_preds.extend(pred_ids.cpu().numpy())
            all_test_labels.extend(label_ids.cpu().numpy())

    # Convert predictions and labels back to language strings
    test_pred_labels = [idx_to_lang[idx] for idx in all_test_preds]
    test_labels = [idx_to_lang[idx] for idx in all_test_labels]

    accuracy = accuracy_score(all_test_labels, all_test_preds)
    print(f"\nTest Accuracy: {accuracy:.4f}")

    # Classification report
    print("\nClassification Report:")
    print(
        classification_report(
            test_labels, test_pred_labels, labels=all_langs, zero_division=0
        )
    )

    # Confusion matrix (optional, for large number of languages)
    if len(all_langs) <= 20:
        print("\nConfusion Matrix:")
        cm = confusion_matrix(test_labels, test_pred_labels, labels=all_langs)
        print(cm)

    return {
        "accuracy": accuracy,
        "classifier": classifier,
        "lang_to_idx": lang_to_idx,
        "idx_to_lang": idx_to_lang,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate wav2vec2 model on language identification task"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to model checkpoint (optional, uses base model if not provided)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="facebook/wav2vec2-base-960h",
        help="HuggingFace model name",
    )
    parser.add_argument(
        "--test_index_path",
        type=str,
        default="./data/lid_index_test.pkl",
        help="Path to test index pickle file",
    )
    parser.add_argument(
        "--train_index_path",
        type=str,
        default="./data/lid_index_train.pkl",
        help="Path to train index pickle file",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for embedding extraction",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="Maximum number of training samples (None for all)",
    )
    parser.add_argument(
        "--max_test_samples",
        type=int,
        default=None,
        help="Maximum number of test samples (None for all)",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=10,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.01,
        help="Learning rate for SGD optimizer",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="HuggingFace token (or set HF_TOKEN env var)",
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=None,
        help="Embedding dimension (auto-detected from model config if not provided)",
    )

    args = parser.parse_args()

    hf_token = args.hf_token or os.getenv("HF_TOKEN")

    train_and_evaluate(
        checkpoint_path=args.checkpoint_path,
        model_name=args.model_name,
        test_index_path=args.test_index_path,
        train_index_path=args.train_index_path,
        batch_size=args.batch_size,
        max_train_samples=args.max_train_samples,
        max_test_samples=args.max_test_samples,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        hf_token=hf_token,
        embedding_dim=args.embedding_dim,
    )
