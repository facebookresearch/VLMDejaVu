"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
#!/bin/bash
#SBATCH --job-name=knn_attack
#SBATCH --output=/path/to/output/file
#SBATCH --error=/path/to/error/file
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=2
#SBATCH --mem=256gb

source env/bin/activate

srun python3 src/dejavu_attack.py \
    --tgt_model_path="/path/to/target/model/checkpoint" \
    --ref_model_path="/path/to/reference/model/checkpoint" \
    --model_type="ViT-B-32" \
    --target_images="/path/to/target/set/images/folder" \
    --public_images="/path/to/public/set/images/folder" \
    --target_captions="/path/to/target/set/captions/file" \
    --public_captions="/path/to/public/set/captions/file" \
    --target_annotations="/path/to/target/set/annotations/file.p" \
    --public_annotations="/path/to/public/set/annotations/file.p" \
    --k=10 \
    --score_fn="max_pred_gap" \
