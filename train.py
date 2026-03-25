#!/usr/bin/env python3
"""
Standalone training script.
Run:  python train.py --dataset ./dataset
"""

import argparse
from model_utils import HandSignClassifier


def main():
    parser = argparse.ArgumentParser(description="Train Hand Sign Classifier")
    parser.add_argument(
        "--dataset",
        type=str,
        default="dataset",
        help="Path to dataset folder (default: ./dataset)",
    )
    args = parser.parse_args()

    print("=" * 50)
    print("  Hand Sign Recognition — Training")
    print("=" * 50)
    print(f"Dataset path : {args.dataset}")
    print()

    clf = HandSignClassifier()
    success = clf.train(args.dataset)

    if success:
        print()
        print("✅ Training complete!")
        print(f"   Classes : {clf.get_classes()}")
        print("   Model saved to hand_sign_model.pkl")
    else:
        print()
        print("❌ Training failed. Check the errors above.")


if __name__ == "__main__":
    main()
