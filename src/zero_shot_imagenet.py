"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import argparse
import torch
import torch.nn.functional as F
from functools import partial
from more_itertools import batched
from typing import Callable, List, Optional, Sequence, Union
from tqdm import tqdm
from utils import IMAGENET_CLASSNAMES, OPENAI_IMAGENET_TEMPLATES
from model import VLM
from dataset import get_dataloader_wrapper


class ImagenetZeroShot():

    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda')
        self.vlm = VLM(
            args.model_type,
            args.checkpoint_path,
            self.device,
            args.pretrained
        )
        self.dataloader = get_dataloader_wrapper(
            images_path=args.images_path,
            batch_size=args.batch_size,
            workers=args.workers,
            processor=self.vlm.preprocess_val
        )
        self.classifier = self.build_zero_shot_classifier(
                self.vlm.model,
                tokenizer=self.vlm.tokenizer,
                classnames=IMAGENET_CLASSNAMES,
                templates=OPENAI_IMAGENET_TEMPLATES,
                num_classes_per_batch=10,
                device=self.device,
                use_tqdm=True,
            )

    def accuracy(self, output, target, topk=(1,)):
        pred = output.topk(max(topk), 1, True, True)[1].t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(float(correct_k.cpu().numpy()))
        return res

    def build_zero_shot_classifier(
            self,
            model,
            tokenizer,
            classnames: Sequence[str],
            templates: Sequence[Union[Callable, str]],
            num_classes_per_batch: Optional[int] = 10,
            device: Union[str, torch.device] = 'cpu',
            use_tqdm: bool = False,
    ):
        """ Build zero-shot classifier weights by iterating over class names
        in batches
        Args:
            model: CLIP model instance
            tokenizer: CLIP tokenizer instance
            classnames: A sequence of class (label) names
            templates: A sequence of callables or format() friendly strings to
                produce templates per class name
            num_classes_per_batch: The number of classes to batch together in
                each forward, all if None
            device: Device to use.
            use_tqdm: Enable TQDM progress bar.
        """
        assert isinstance(templates, Sequence) and len(templates) > 0
        assert isinstance(classnames, Sequence) and len(classnames) > 0
        use_format = isinstance(templates[0], str)
        num_templates = len(templates)
        num_classes = len(classnames)
        if use_tqdm:
            import tqdm
            num_iter = 1
            if num_classes_per_batch is not None:
                num_iter = ((num_classes - 1) // num_classes_per_batch + 1)
            iter_wrap = partial(
                tqdm.tqdm,
                total=num_iter,
                unit_scale=num_classes_per_batch
            )
        else:
            iter_wrap = iter

        def _process_batch(bc):
            nbc = len(bc)
            texts = []
            for tpt in templates:
                if use_format:
                    texts.extend([tpt.format(c) for c in bc])
                else:
                    texts.extend([tpt(c) for c in bc])
            texts = tokenizer(texts).to(device)
            class_emb = F.normalize(model.encode_text(texts), dim=-1)
            class_emb = class_emb.reshape(nbc, num_templates, -1).mean(dim=1)
            class_emb = class_emb / class_emb.norm(dim=1, keepdim=True)
            class_emb = class_emb.T
            return class_emb

        with torch.no_grad():
            if num_classes_per_batch:
                batched_embeds = [
                    _process_batch(batch) for batch in iter_wrap(
                        batched(classnames, num_classes_per_batch)
                    )
                ]
                zeroshot_weights = torch.cat(batched_embeds, dim=1)
            else:
                zeroshot_weights = _process_batch(classnames)
        return zeroshot_weights

    def evaluate(self):
        acc, n = 0, 0
        top1, top5 = 0, 0
        with torch.no_grad():
            for batch in tqdm(self.dataloader):
                images, labels = batch['image'], batch['label']
                images = images.to(device=self.device)
                labels = labels.to(device=self.device)
                image_features = F.normalize(
                    self.vlm.model.encode_image(images),
                    dim=-1
                )
                logits = 100. * image_features @ self.classifier
                acc1, acc5 = self.accuracy(logits, labels, topk=(1, 5))
                top1 += acc1
                top5 += acc5
                n += images.size(0)
        top1 = (top1 / n)
        top5 = (top5 / n)
        print(n, top1, top5)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--images_path',
        type=str,
        help="Path to the ImageNet folder"
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
    args = parser.parse_args()
    print(vars(args))

    zero_shot = ImagenetZeroShot(args)
    zero_shot.evaluate()
