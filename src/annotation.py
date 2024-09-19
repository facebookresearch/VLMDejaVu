"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import numpy as np
import os, json, cv2, random, sys
import detectron2
import webdataset as wds
import pickle
import nltk
import argparse
import torch

from tqdm.auto import tqdm
from nltk.corpus import wordnet
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms.functional import pil_to_tensor
from detectron2.utils.logger import setup_logger
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer

# Note: need to set the correct path to the Detic folder
sys.path.insert(0, 'Detic/')
sys.path.insert(0, 'Detic/third_party/CenterNet2/')

from centernet.config import add_centernet_config
from detic.config import add_detic_config
from detic.modeling.utils import reset_cls_test
from detic.modeling.text.text_encoder import build_text_encoder
from dataset import get_dataloader_wrapper
from visualizer import Visualizer
from utils import _COLORS


class Annotator():

    def __init__(self, thresh=0.5, use_cpu=False):
        self.cfg = get_cfg()
        self.cfg.MODEL.DEVICE = "cpu" if use_cpu else "cuda"
        self.thresh = thresh
        self.class_names = get_imagenet_class_names()
        self.vocab_path = "data/imagenet21k_clip_a+cname.p"
        self.setup_detector()

    def setup_vocab(self):
        num_classes = len(self.metadata.thing_classes)
        prompt='a '
        text_encoder = build_text_encoder(pretrain=True)
        text_encoder.eval()
        text_encoder = text_encoder.cuda()
        classifier = []
        batch_size = 1024
        i = 0
        pbar = tqdm(total=math.ceil(num_classes / batch_size) * batch_size)
        while i < num_classes:
            batch_names = self.class_names[i: min(i + batch_size, num_classes)]
            texts = [prompt + x for x in batch_names]
            with torch.no_grad():
                emb = text_encoder(texts).detach().permute(1, 0).contiguous()
                classifier.append(emb.cpu())
            i += batch_size
            pbar.update(batch_size)
        pbar.close()
        classifier = torch.cat(classifier, dim=1)
        pickle.dump(classifier, open(self.vocab_path, 'wb'))

    def setup_detector(self):
        add_centernet_config(self.cfg)
        add_detic_config(self.cfg)
        mname = "Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size"
        murl = f"https://dl.fbaipublicfiles.com/detic/{mname}.pth"
        self.cfg.merge_from_file(f"data/{mname}.yaml")
        self.cfg.MODEL.WEIGHTS = murl
        self.cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = 'rand'
        self.cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.thresh
        self.model = build_model(self.cfg)
        DetectionCheckpointer(self.model).load(self.cfg.MODEL.WEIGHTS)
        self.model.train(False)
        self.metadata = MetadataCatalog.get("in21k")
        self.metadata.thing_classes = self.class_names
        if not os.path.exists(self.vocab_path):
            self.setup_vocab()
        classifier = pickle.load(open(self.vocab_path, 'rb'))
        num_classes = len(self.metadata.thing_classes)
        reset_cls_test(self.model, classifier, num_classes)

    def annotate(self, dataloader, save_path):
        labels, scores = [], []
        pbar = tqdm(total=dataloader.num_batches)
        with torch.no_grad():
            for batch in dataloader:
                images = []
                for img in batch['image']:
                    images.append({"image": pil_to_tensor(img)})
                batch_outputs = self.model(images)
                for op in batch_outputs:
                    labels.append(op["instances"].pred_classes.cpu().numpy())
                    scores.append(op["instances"].scores.cpu().numpy())
                pbar.update()
        pbar.close()
        pickle.dump([labels, scores], open(save_path, 'wb'))

    def visual_annotate(self, im, gt=None):
        im = [{"image": pil_to_tensor(im_)} for im_ in im]
        with torch.no_grad():
            batch_outputs = self.model(im)
        out_im = []
        for idx, outputs in enumerate(batch_outputs):
            objs = outputs["instances"].pred_classes.cpu().numpy()
            objs_filt = []
            for obj in objs:
                if obj not in objs_filt:
                    objs_filt.append(obj)
            objs = objs_filt
            c = [_COLORS[10]] * len(self.metadata.get("thing_classes"))
            if gt is not None:
                for i, obj in enumerate(objs):
                    if obj not in gt:
                        c[i] = _COLORS[0]
            c = np.array(c).astype(np.float32).reshape(-1, 3)
            v = Visualizer(
                im[idx]["image"].permute(1, 2, 0),
                self.metadata,
                scale=1.2,
                colors=c
            )
            v_op = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            out_im.append(v_op.get_image())
        return out_im


def get_imagenet_class_names():
    nltk.download('wordnet')
    wnids = [x.strip() for x in open('data/imagenet21k_wordnet_ids.txt', 'r')]
    class_names = []
    for wnid in wnids:
        synset = wordnet.synset_from_pos_and_offset('n', int(wnid[1:]))
        synonyms = [x.name() for x in synset.lemmas()]
        class_names.append(synonyms[0])
    return class_names


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--images_path',
        type=str,
        help="Path to the image folder"
    )
    parser.add_argument(
        '--save_path',
        type=str,
        help="Path to save annotations: provide file name with .p extension"
    )
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--thresh', type=float, default=0.5)
    parser.add_argument('--use_cpu', default=False, action='store_true')
    args = parser.parse_args()
    print(vars(args))
    setup_logger()

    dataloader = get_dataloader_wrapper(
            images_path=args.images_path,
            batch_size=args.batch_size,
            workers=args.workers,
            processor=transforms.Compose([transforms.Resize(256)])
        )
    annotator = Annotator(thresh=args.thresh, use_cpu=args.use_cpu)
    annotator.annotate(dataloader, args.save_path)
