"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
#!/bin/bash
#SBATCH --job-name=linear_probe
#SBATCH --output=/path/to/output/file
#SBATCH --error=/path/to/error/file
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus-per-node=1
#SBATCH --time=2:00:00
#SBATCH --mem=100gb

source env/bin/activate

srun python3 src/linear_probe.py \
    --train_images_path="/path/to/training/images/folder" \
    --val_images_path="/path/to/validation/images/folder" \
    --checkpoint_path="/path/to/model/checkpoint" \
    --model_type="ViT-B-32" \
    --emb="pj" \
    --normalize
