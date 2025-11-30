#!/usr/bin/env python3
import torch
import os
from pathlib import Path

def analyze_checkpoint(file_path):
    """Analyze a PyTorch checkpoint file and return detailed information."""
    print(f"\n{'='*60}")
    print(f"Analyzing: {os.path.basename(file_path)}")
    print(f"{'='*60}")

    try:
        # Load the checkpoint
        checkpoint = torch.load(file_path, map_location='cpu')

        print(f"File size: {os.path.getsize(file_path):,} bytes")
        print(f"Type: {type(checkpoint)}")

        if isinstance(checkpoint, dict):
            print(f"Keys: {list(checkpoint.keys())}")

            for key, value in checkpoint.items():
                print(f"\n--- Key: '{key}' ---")
                print(f"Type: {type(value)}")

                if isinstance(value, torch.Tensor):
                    print(f"Shape: {value.shape}")
                    print(f"Dtype: {value.dtype}")
                    print(f"Device: {value.device}")
                    print(f"Requires grad: {value.requires_grad}")
                    print(f"Memory usage: {value.element_size() * value.numel():,} bytes")

                    # Show some statistics
                    if value.numel() > 0:
                        print(f"Min: {value.min().item():.6f}")
                        print(f"Max: {value.max().item():.6f}")
                        print(f"Mean: {value.mean().item():.6f}")
                        print(f"Std: {value.std().item():.6f}")

                elif isinstance(value, (list, tuple)):
                    print(f"Length: {len(value)}")
                    if len(value) > 0:
                        print(f"First element type: {type(value[0])}")
                        if isinstance(value[0], torch.Tensor):
                            print(f"First tensor shape: {value[0].shape}")

                elif isinstance(value, dict):
                    print(f"Sub-dict keys: {list(value.keys())}")

                else:
                    print(f"Value: {value}")

        elif isinstance(checkpoint, torch.Tensor):
            print(f"Direct tensor:")
            print(f"Shape: {checkpoint.shape}")
            print(f"Dtype: {checkpoint.dtype}")
            print(f"Memory usage: {checkpoint.element_size() * checkpoint.numel():,} bytes")

        else:
            print(f"Content: {checkpoint}")

    except Exception as e:
        print(f"Error loading {file_path}: {e}")

def main():
    checkpoint_dir = Path("/root/code/reduced-unet3d/block_checkpoints")

    # Get all .pth files
    pth_files = sorted(checkpoint_dir.glob("*.pth"))

    print("PyTorch Checkpoint Analysis")
    print("=" * 60)
    print(f"Found {len(pth_files)} checkpoint files")

    # Analyze each file
    for pth_file in pth_files:
        analyze_checkpoint(pth_file)

    # Summary
    print(f"\n\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    total_size = 0
    for pth_file in pth_files:
        size = os.path.getsize(pth_file)
        total_size += size
        print(f"{pth_file.name:25} {size:>10,} bytes")

    print(f"{'Total size:':<25} {total_size:>10,} bytes")

if __name__ == "__main__":
    main()
