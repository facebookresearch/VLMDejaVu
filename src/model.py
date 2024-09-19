"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import torch.nn as nn
from open_clip import create_model_and_transforms, get_tokenizer
from training.file_utils import pt_load
from sentence_transformers import SentenceTransformer


class VLM(nn.Module):

    def __init__(self, model_type, model_path, device, pretrained=None):
        super().__init__()
        pretrained_choice = ['openai', 'yfcc15m', 'cc12m', None]
        assert pretrained in pretrained_choice, f"{pretrained} not supported"
        self.tokenizer = get_tokenizer(model_type)
        mtuple = create_model_and_transforms(
            model_type,
            pretrained=pretrained,
            precision='amp',
            device=device,
            jit=False,
            force_quick_gelu=False,
            force_custom_text=False,
            force_patch_dropout=None,
            force_image_size=None,
            pretrained_image=False,
            image_mean=None,
            image_std=None,
            aug_cfg=None,
            output_dict=True,
        )
        self.model, self.preprocess_train, self.preprocess_val = mtuple
        if not pretrained:
            checkpoint = pt_load(model_path, map_location='cpu')
            sd = checkpoint['state_dict']
            if next(iter(sd.items()))[0].startswith('module'):
                sd = {k[len('module.'):]: v for k, v in sd.items()}
            self.model.load_state_dict(sd)
        self.model.eval()


class LLM(nn.Module):

    def __init__(self, model_type):
        super().__init__()
        model_choice = { # base: 768 dim, large: 1024 dim
            'e5-base-v2': 'intfloat/e5-base-v2',
            'e5-large-v2': 'intfloat/e5-large-v2',
            'gte-base': 'thenlper/gte-base',
            'gte-large': 'thenlper/gte-large'
        }
        assert model_type in model_choice, f"{model_type} not supported"
        self.model = SentenceTransformer(model_choice[model_type])
        self.tokenizer = None
