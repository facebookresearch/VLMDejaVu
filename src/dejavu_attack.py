"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import torch
import numpy as np
import faiss
import argparse
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from model import VLM, LLM
from dataset import get_dataloader_wrapper
from annotation import Annotator


class VisualizeMemorization():
    '''
    Visualize the most memorized training set examples along with their nearest
    public set images.
    '''
    def __init__(self, args):
        self.annotator = Annotator(thresh=args.thresh, use_cpu=args.use_cpu)
        self.batch_size = args.batch_size
        self.target_dataloader = get_dataloader_wrapper(
            images_path=args.target_images,
            caption_path=args.target_captions,
            annotation_path=args.target_annotations,
            batch_size=ars.batch_size,
            workers=args.workers,
        )
        self.public_dataloader = get_dataloader_wrapper(
            images_path=args.public_images,
            caption_path=args.public_captions,
            annotation_path=args.public_annotations,
            batch_size=ars.batch_size,
            workers=args.workers,
        )

    def show_image(self, img, gt=None):
        fig, ax = plt.subplots(figsize=(3, 3))
        if gt is not None:
            img = self.annotator.visual_annotate([img], gt)[0]
        ax.imshow(img, aspect='auto')
        ax.axis('off')
        fig.tight_layout()
        plt.show()

    def show_image_grid(self, imgs, cols, gt=None):
        fig, ax = plt.subplots(1, cols, figsize=(cols*3, 3))
        if gt is not None:
            imgs = self.annotator.visual_annotate(imgs, gt)
        for i in range(cols):
            ax[i].imshow(imgs[i], aspect='auto')
            ax[i].axis('off')
        fig.tight_layout()
        plt.show()

    def get_samples(self, tgt_samples, pub_samples):
        data_set = {}
        for i, batch in enumerate(self.target_dataloader):
            for j in range(len(batch['image'])):
                if i * self.batch_size + j in tgt_samples:
                    data_set[f"tgt_{i * self.batch_size + j}"] = {
                        'image': batch['image'][j],
                        'text': batch['text'][j],
                        'label': batch['label'][j]
                    }
        for i, batch in enumerate(self.public_dataloader):
            for j in range(len(batch['image'])):
                if i * self.batch_size + j in pub_samples:
                    data_set[f"pub_{i * self.batch_size + j}"] = {
                        'image': batch['image'][j],
                        'text': batch['text'][j],
                        'label': batch['label'][j]
                    }
        return data_set

    def visualize(self, target_idx, tgt_neighbors, ref_neighbors):
        pub_samples = []
        for samples in tgt_neighbors:
            pub_samples.extend(samples)
        for samples in ref_neighbors:
            pub_samples.extend(samples)
        data_set = self.get_samples(target_idx, set(pub_samples))
        for i in range(len(target_idx)):
            print(f"\nTarget Image {i+1}:")
            self.show_image(
                data_set[f"tgt_{target_idx[i]}"]['image'],
                data_set[f"tgt_{target_idx[i]}"]['label']
            )
            print(f"Caption: {data_set[f'tgt_{target_idx[i]}']['text']}")
            print("Nearest Public Images for Target Model:")
            self.show_image_grid(
                [data_set[f"pub_{idx}"]['image'] for idx in tgt_neighbors[i]],
                5,
                data_set[f"tgt_{target_idx[i]}"]['label']
            )
            print("Nearest Public Images for Reference Model:")
            self.show_image_grid(
                [data_set[f"pub_{idx}"]['image'] for idx in ref_neighbors[i]],
                5,
                data_set[f"tgt_{target_idx[i]}"]['label']
            )


class DejaVuAdversary():
    '''
    kNN adversary that finds the nearest neighbors in public set to the target
    data set records.
    '''
    def __init__(
        self,
        model,
        public_dataloader,
        target_dataloader,
        k=100,
        is_llm=False,
        text_index=False
    ):
        self.model = model
        self.public_dataloader = public_dataloader
        self.target_dataloader = target_dataloader
        self.k = k
        self.is_llm = is_llm
        self.text_index = text_index
        self.public_index = None
        self.target_neighbors = []
        self.target_scores = []
        self.public_labels = []
        self.target_labels = []
        self.p = faiss.GpuMultipleClonerOptions()
        self.p.shard = True

        self.build_index()
        self.search_nns()

    def build_index(self):
        if self.is_llm:
            assert text_index, "Error: LLM only works on text modality."
        embeddings = []
        pbar = tqdm(total=self.public_dataloader.num_batches)
        with torch.no_grad():
            for batch in self.public_dataloader:
                if self.text_index:
                    if self.is_llm:
                        embeds = self.model.encode(
                            batch['text'],
                            normalize_embeddings=True
                        )
                    else:
                        embeds = self.model.encode_text(
                            batch['text'].to(device=device),
                            normalize=True
                        ).cpu().numpy()
                else:
                    embeds = self.model.encode_image(
                        batch['image'].to(device=device),
                        normalize=True
                    ).cpu().numpy()
                embeddings.extend(embeds)
                self.public_labels.extend(batch['label'])
                pbar.update()
        pbar.close()
        embeddings = np.array(embeddings, dtype='float32')
        self.public_index = faiss.IndexFlatL2(embeddings.shape[1])
        self.public_index = faiss.index_cpu_to_all_gpus(
            self.public_index,
            self.p
        )
        self.public_index.add(embeddings)

    def search_nns(self):
        pbar = tqdm(total=self.target_dataloader.num_batches)
        with torch.no_grad():
            for batch in self.target_dataloader:
                if self.is_llm:
                    embeds = self.model.encode(
                        batch['text'],
                        normalize_embeddings=True
                    )
                else:
                    embeds = self.model.encode_text(
                        batch['text'].to(device=device),
                        normalize=True
                    ).cpu().numpy()
                D, I = self.public_index.search(embeds, self.k)
                self.target_neighbors.append(I)
                self.target_scores.append(D)
                self.target_labels.extend(batch['label'])
                pbar.update()
        pbar.close()
        self.target_neighbors = np.concatenate(self.target_neighbors, axis=0)
        self.target_scores = np.concatenate(self.target_scores, axis=0)


class DejaVuAttack():
    '''
    DejaVu attack uses the kNN adversary to quantify the memorization gap
    between target and reference models.
    '''
    def __init__(self, args):
        self.score_fn = args.score_fn
        self.k = args.k
        self.device = torch.device('cuda')
        self.visualizeMem = VisualizeMemorization(args)

        tgt_model = VLM(
            args.model_type,
            args.tgt_model_path,
            self.device,
            args.pretrained
        )

        target_dataloader = get_dataloader_wrapper(
            images_path=args.target_images,
            caption_path=args.target_captions,
            annotation_path=args.target_annotations,
            batch_size=ars.batch_size,
            workers=args.workers,
            processor=tgt_model.preprocess_val,
            tokenizer=tgt_model.tokenizer
        )
        public_dataloader = get_dataloader_wrapper(
            images_path=args.public_images,
            caption_path=args.public_captions,
            annotation_path=args.public_annotations,
            batch_size=ars.batch_size,
            workers=args.workers,
            processor=tgt_model.preprocess_val,
            tokenizer=tgt_model.tokenizer
        )
        tgt_adversary = DejaVuAdversary(
            model=tgt_model.model,
            public_dataloader=public_dataloader,
            target_dataloader=target_dataloader,
            k=args.k,
            is_llm=False,
            text_index=args.text_index
        )

        if args.one_model_test:
            ref_model = LLM(args.llm_type)
        else:
            ref_model = VLM(
                args.model_type,
                args.ref_model_path,
                self.device,
                args.pretrained
            )

        target_dataloader = get_dataloader_wrapper(
            images_path=args.target_images,
            caption_path=args.target_captions,
            annotation_path=args.target_annotations,
            batch_size=ars.batch_size,
            workers=args.workers,
            processor=tgt_model.preprocess_val,
            tokenizer=ref_model.tokenizer
        )
        public_dataloader = get_dataloader_wrapper(
            images_path=args.public_images,
            caption_path=args.public_captions,
            annotation_path=args.public_annotations,
            batch_size=ars.batch_size,
            workers=args.workers,
            processor=tgt_model.preprocess_val,
            tokenizer=ref_model.tokenizer
        )
        ref_adversary = DejaVuAdversary(
            model=ref_model.model,
            public_dataloader=public_dataloader,
            target_dataloader=target_dataloader,
            k=args.k,
            is_llm=args.one_model_test,
            text_index=args.text_index
        )

        self.tgt_labels = tgt_adversary.target_labels
        self.pub_labels = tgt_adversary.public_labels
        self.tgt_neighbors = tgt_adversary.target_neighbors
        self.tgt_scores = tgt_adversary.target_scores
        self.ref_neighbors = ref_adversary.target_neighbors
        self.ref_scores = ref_adversary.target_scores

    def sorting_function(self):
        sz = lambda ar: len(ar)
        # sorting based on least nn embedding distance
        if self.score_fn == 'min_dist':
            return np.argsort(np.min(self.tgt_scores, axis=1))
        # sorting based on the number of correct unique objects detected by
        # target model
        elif self.score_fn == 'crct_pred':
            crct = []
            for i in range(len(self.tgt_scores)):
                orig = set(self.tgt_labels[i])
                tset = set(self.tgt_pred_labels[i])
                crct.append(sz(orig.intersection(tset)))
            return np.argsort(-np.array(crct))
         # sorting based on the gap between number of correct unique objects
         # detected
        else:
            diff = []
            for i in range(len(self.tgt_scores)):
                orig = set(self.tgt_labels[i])
                tset = set(self.tgt_pred_labels[i])
                rset = set(self.ref_pred_labels[i])
                v_ = sz(orig.intersection(tset)) - sz(orig.intersection(rset))
                diff.append(v_)
            return np.argsort(-np.array(diff))

    def safe_div(self, a, b):
        return 0 if b == 0 else a / b

    def get_precision_scores(self, ground_truth, nn_labels):
        return list(
            map(
                lambda i: self.safe_div(
                    len(set(ground_truth[i]).intersection(set(nn_labels[i]))),
                    len(set(nn_labels[i]))
                ),
                range(len(ground_truth))
            )
        )

    def get_recall_scores(self, ground_truth, nn_labels):
        return list(
            map(
                lambda i: self.safe_div(
                    len(set(ground_truth[i]).intersection(set(nn_labels[i]))),
                    len(set(ground_truth[i]))
                ),
                range(len(ground_truth))
            )
        )

    def evaluate_memorization(self):
        self.tgt_pred_labels = []
        for neighbors in self.tgt_neighbors:
            neighbor_labels = []
            for neighbor in neighbors:
                neighbor_labels.extend(np.unique(self.pub_labels[neighbor]))
            self.tgt_pred_labels.append(neighbor_labels)

        self.ref_pred_labels = []
        for neighbors in self.ref_neighbors:
            neighbor_labels = []
            for neighbor in neighbors:
                neighbor_labels.extend(np.unique(self.pub_labels[neighbor]))
            self.ref_pred_labels.append(neighbor_labels)

        sorted_idx = list(
            filter(
                lambda idx: len(self.tgt_labels[idx]) > 0,
                self.sorting_function()
            )
        )

        tgt_prec = self.get_precision_scores(
            [self.tgt_labels[idx] for idx in sorted_idx],
            [self.tgt_pred_labels[idx] for idx in sorted_idx]
        )
        ref_prec = self.get_precision_scores(
            [self.tgt_labels[idx] for idx in sorted_idx],
            [self.ref_pred_labels[idx] for idx in sorted_idx]
        )
        tgt_rec = self.get_recall_scores(
            [self.tgt_labels[idx] for idx in sorted_idx],
            [self.tgt_pred_labels[idx] for idx in sorted_idx]
        )
        ref_rec = self.get_recall_scores(
            [self.tgt_labels[idx] for idx in sorted_idx],
            [self.ref_pred_labels[idx] for idx in sorted_idx]
        )

        print("\n******** Population Memorization Metrics ********")
        res = [tar > ref for tar, ref in zip(tgt_prec, ref_prec)]
        aa = sum(res) / len(res)
        res = [tar < ref for tar, ref in zip(tgt_prec, ref_prec)]
        bb = sum(res) / len(res)
        print("Population Precision Gap: %.4f" % (aa - bb))
        res = [tar > ref for tar, ref in zip(tgt_rec, ref_rec)]
        aa = sum(res) / len(res)
        res = [tar < ref for tar, ref in zip(tgt_rec, ref_rec)]
        bb = sum(res) / len(res)
        print("Population Recall Gap: %.4f" % (aa - bb))

        n1, b1, patches = plt.hist(tgt_rec, 100, label='Target')
        n2, b2, patches = plt.hist(ref_rec, 100, label='Reference')
        plt.close()
        n1 = n1 / len(res)
        n2 = n2 / len(res)
        for x_ in np.arange(len(n1)-2, -1, -1):
            n1[x_] += n1[x_+1]
        b1 = [0] + list(b1) + [1]
        n1 = [1] + list(n1) + [0]
        tgt_auc = sum([n1[i] * (b1[i+1] - b1[i]) for i in range(len(n1))])
        for x_ in np.arange(len(n2)-2, -1, -1):
            n2[x_] += n2[x_+1]
        b2 = [0] + list(b2) + [1]
        n2 = [1] + list(n2) + [0]
        ref_auc = sum([n2[i] * (b2[i+1] - b2[i]) for i in range(len(n2))])
        print("AUC Gap: %.4f" % (tgt_auc - ref_auc))

        print("\n******** Fine-Grained Memorization Metrics ********")
        rng = [0.001, 0.01, 0.1]
        records = [max(1, int(len(tgt_prec) * fraction)) for fraction in rng]
        for topk in records:
            print("\nEvaluating on Top-%d records..." % topk)
            tgt_p = np.mean(tgt_prec[:topk])
            ref_p = np.mean(ref_prec[:topk])
            tgt_r = np.mean(tgt_rec[:topk])
            ref_r = np.mean(ref_rec[:topk])
            tgt_f = 2 * (tgt_p * tgt_r) / (tgt_p + tgt_r + 1e-5)
            ref_f = 2 * (ref_p * ref_r) / (ref_p + ref_r + 1e-5)
            print("Precision Gap: %.4f" % (tgt_p - ref_p))
            print("Recall Gap: %.4f" % (tgt_r - ref_r))
            print("F-score Gap: %.4f" % (tgt_f - ref_f))

        print("\n******** Visualizing Top-5 Memorized Examples ********")
        self.visualizeMem.visualize(
            sorted_idx[:5],
            self.tgt_neighbors[sorted_idx[:5]][:min(5, self.k)],
            self.ref_neighbors[sorted_idx[:5]][:min(5, self.k)]
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--tgt_model_path',
        type=str,
        help="Path to targt model checkpoint"
    )
    parser.add_argument(
        '--ref_model_path',
        type=str,
        help="Path to reference model checkpoint"
    )
    parser.add_argument(
        '--model_type',
        type=str,
        default='ViT-B-32',
        help="CLIP model type, e.g. ViT-B-32 or RN50"
    )
    parser.add_argument(
        '--pretrained',
        type=str,
        choices=['openai', 'yfcc15m', 'cc12m'],
        default=None
    )
    parser.add_argument('--batch_size', type=int, default=500)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--thresh', type=float, default=0.5)
    parser.add_argument('--use_cpu', default=False, action='store_true')
    parser.add_argument(
        '--target_images',
        type=str,
        help="Path to the target image folder"
    )
    parser.add_argument(
        '--public_images',
        type=str,
        help="Path to the public image folder"
    )
    parser.add_argument(
        '--target_captions',
        type=str,
        help="Path to the target captions file"
    )
    parser.add_argument(
        '--public_captions',
        type=str,
        help="Path to the public captions file"
    )
    parser.add_argument(
        '--target_annotations',
        type=str,
        help="Path to the target annotations file, stored in pickle format"
    )
    parser.add_argument(
        '--public_annotations',
        type=str,
        help="Path to the public annotations file, stored in pickle format"
    )
    parser.add_argument(
        '--k',
        type=int,
        default=100,
        help="Number of nearest neighbors to find for nearest neighbor attack"
    )
    parser.add_argument(
        '--llm_type',
        type=str,
        default='gte-base',
        help="Type of LLM to use as reference model for one model test"
    )
    parser.add_argument(
        '--one_model_test',
        default=False,
        action='store_true',
        help="Whether to run one-model or two-model test, default is two-model"
    )
    parser.add_argument(
        '--text_index',
        default=False,
        action='store_true',
        help="Whether to build index over text or image, default is image"
    )
    parser.add_argument(
        '--score_fn',
        type=str,
        default='max_pred_gap',
        choices=['min_dist', 'crct_pred', 'max_pred_gap'],
        help="Criterion to sort the target records, default is max_pred_gap"
    )
    args = parser.parse_args()
    print(vars(args))

    attack = DejaVuAttack(args)
    attack.evaluate_memorization()
