"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import pickle
import argparse
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision import transforms

class VLMDataset(Dataset):
    """
    VLMDataset has images, texts (captions) and labels. Off-the-shelf
    object annotator (see annotation.py) can be used to get the image labels.
    """
    def __init__(
        self,
        images,
        texts,
        labels=None,
        processor=None,
        tokenizer=None
    ):
        self.images = images
        self.texts = texts
        self.labels = [None]* len(images) if labels == None else labels
        self.processor = lambda x: x if processor == None else processor
        self.tokenizer = lambda x: x if tokenizer == None else tokenizer

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        encoding = {'image': self.processor(self.images[idx]),
                    'text': self.tokenizer(self.texts[idx]),
                    'label': self.labels[idx]}
        return {k: v.squeeze() for k, v in encoding.items()}


class ImageDataset(Dataset):
    """
    ImageDataset has images and labels, such as ImageNet.
    """
    def __init__(self, images, labels=None, processor=None):
        self.images = images
        self.labels = [None]* len(images) if labels == None else labels
        self.processor = lambda x: x if processor == None else processor

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        encoding = {'image': self.processor(self.images[idx]),
                    'label': self.labels[idx]}
        return {k: v.squeeze() for k, v in encoding.items()}


def get_dataloader(
    images,
    texts=None,
    labels=None,
    batch_size=4,
    workers=8,
    processor=None,
    tokenizer=None
):
    """
    Custom dataloader that takes a list of images, (optional) list of texts,
    and (optional) list of labels. Returns a dataloader over VLMDataset if
    captions are provided otherwise returns a dataloader over ImageDataset.
    """
    if texts is not None:
        dataset = VLMDataset(
            images=images,
            texts=texts,
            labels=labels,
            processor=processor,
            tokenizer=tokenizer
        )
    else:
        dataset = ImageDataset(
            images=images,
            labels=labels,
            processor=processor
        )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
    )
    dataloader.num_samples = len(images)
    dataloader.num_batches = len(dataloader)
    return dataloader


def get_dataloader_wrapper(
    images_path,
    caption_path=None,
    annotation_path=None,
    batch_size=4,
    workers=8,
    processor=None,
    tokenizer=None
):
    """
    Custom dataloader wrapper that takes seperate image directory path,
    (optional) path to captions file, and (optional) path to annotation file.
    Returns a dataloader.
    """
    data_set = datasets.ImageFolder(images_path)

    # Datasets like ImageNet returns a tuple of (image, label)
    # Note: need to be careful whether this format is valid
    images = [row[0] for row in data_set]
    labels = [row[1] for row in data_set]

    if caption_path is not None:
        texts = []
          with open(caption_path, 'r') as f:
            for line in f:
                texts.append(line.strip())
    else:
          texts = None

    # Overwrite labels with precomputed annotations if path is provided
    if annotation_path is not None:
        labels, scores = pickle.load(open(annotation_path, 'rb'))

    return get_dataloader(
        images=images,
        texts=texts,
        labels=labels,
        batch_size=batch_size,
          workers=workers,
        processor=processor,
        tokenizer=tokenizer
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--images_path',
        type=str,
        help="Path to the image folder"
    )
    parser.add_argument(
        '--caption_path',
        type=str,
        help="Path to the captions file",
        default=None
    )
    parser.add_argument(
        '--annotation_path',
        type=str,
        help="Path to the annotations file, stored in pickle format",
        default=None
    )
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    print(vars(args))

    dataloader = get_dataloader_wrapper(
        images_path=args.images_path,
        caption_path=args.caption_path,
        annotation_path=args.annotation_path,
        batch_size=args.batch_size,
        workers=args.workers,
        processor=transforms.Compose([transforms.Resize(256)]),
        tokenizer=None
    )
    print(dataloader.num_samples, dataloader.num_batches)
