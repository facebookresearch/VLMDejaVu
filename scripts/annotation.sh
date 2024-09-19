"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
#!/bin/bash
#SBATCH --job-name=annotation
#SBATCH --output=/path/to/output/file
#SBATCH --error=/path/to/error/file
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=10
#SBATCH --gpus-per-node=1
#SBATCH --tasks-per-node=1

source env/bin/activate

srun python3 src/annotation.py \
    --images_path="/path/to/images/folder" \
    --save_path="/path/to/save/annotations/to.p" \
    --batch_size=4 \
