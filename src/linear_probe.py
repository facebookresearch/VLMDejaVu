"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import os
import numpy as np
import torch
import argparse
from tqdm.auto import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from torchvision.models.feature_extraction import create_feature_extractor
from model import VLM
from dataset import get_dataloader_wrapper


class LinearProbe():

    def __init__(self, args):
        self.args = args
        train_tuple = self.get_embeddings(self.train_images_path)
        val_tuple = self.get_embeddings(self.val_images_path)
        self.train_features, self.train_labels = train_tuple
        self.val_features, self.val_labels = val_tuple

    def get_embedding_func(self, model, model_type, emb):
        if emb == 'pj':
            return lambda x: model.encode_image(x, normalize=True)
        if model_type == 'RN50':
            l4 = create_feature_extractor(model.visual, {'layer4': 'layer4'})
            return lambda x: torch.nn.AvgPool2d(7)(l4(x)['layer4']).squeeze()
        # bb encoding for ViT-B-32
        model.visual.proj = None
        return lambda x: model.encode_image(x, normalize=True)

    def get_embeddings(self, images_path):
        device = torch.device('cuda')
        vlm = VLM(
            self.args.model_type,
            self.args.checkpoint_path,
            device,
            self.args.pretrained
        )
        emb_fn = self.get_embedding_func(
            vlm.model,
            self.args.model_type,
            self.args.emb
        )
        dataloader = get_dataloader_wrapper(
            images_path=images_path,
            batch_size=self.args.batch_size,
            workers=self.args.workers,
            processor=vlm.preprocess_val
        )
        feats, labels = [], []
        with torch.no_grad():
            for batch in tqdm(dataloader):
                feats.extend(emb_fn(batch["image"].to(device)).cpu().numpy())
                labels.extend(batch["label"].numpy())
        feats = np.array(feats)
        labels = np.array(labels)
        print(f"Embeddings Shape: {feats.shape}")
        return feats, labels

    def evaluate(self):
        print(self.train_features.shape, self.test_features.shape)
        if self.args.normalize:
            mu = np.mean(self.train_features, axis=0)
            std = np.std(self.train_features, axis=0) + 1e-8
            self.train_features = (self.train_features - mu) / std
            self.test_features = (self.test_features - mu) / std
        probe_model = LogisticRegression(
            solver='lbfgs',
            C=self.args.C,
            max_iter=self.args.num_epochs,
            random_state=0,
            verbose=0
        )
        probe_model.fit(self.train_features, self.train_labels)
        train_pred = probe_model.predict(self.train_features)
        test_pred = probe_model.predict(self.test_features)
        tr_acc = accuracy_score(self.train_labels, train_pred)
        ts_acc = accuracy_score(self.test_labels, test_pred)
        print(f"Training Accuracy: {tr_acc}")
        print(f"Test Accuracy: {ts_acc}")
        tr_bacc = balanced_accuracy_score(self.train_labels, train_pred)
        ts_bacc = balanced_accuracy_score(self.test_labels, test_pred)
        print(f"Class-Balanced Training Accuracy: {tr_bacc}")
        print(f"Class-Balanced Test Accuracy: {ts_bacc}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train_images_path',
        type=str,
        help="Path to the training image folder"
    )
    parser.add_argument(
        '--val_images_path',
        type=str,
        help="Path to the validation image folder"
    )
    parser.add_argument(
        '--checkpoint_path',
        type=str,
        help="Path to model checkpoint",
        default=""
    )
    parser.add_argument(
        '--pretrained',
        type=str,
        choices=['openai', 'yfcc15m', 'cc12m'],
        default=None
    )
    parser.add_argument('--model_type', type=str, default='ViT-B-32')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--emb', type=str, choices=['pj', 'bb'], default='pj')
    parser.add_argument('--normalize', action="store_true", default=False)
    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--C', type=float, default=1)
    args = parser.parse_args()
    print(vars(args))

    linear_probe = LinearProbe(args)
    linear_probe.evaluate()
