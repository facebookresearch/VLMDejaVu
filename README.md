# Déjà Vu Memorization in VLMs

The Déjà Vu Memorization framework allows for evaluating the memorization in
CLIP models. The code is based on the paper titled `Déjà Vu Memorization in
Vision--Language Models' ([see here](https://arxiv.org/abs/2402.02103)). This
framework implements both two-model (a more accurate memorization
quantification attack where you need to provide two model checkpoints, a
target CLIP model and a reference CLIP model) and one-model (an approximate
memorization quantification attack that does not need a separate reference
model checkpoint and can thus be applied to any CLIP-style model out of the
box) memorization attacks. All you need to provide are the target CLIP model
checkpoint (and additionally the reference CLIP model checkpoint for the
two-model test), the target model's training set consisting of images and
captions (called the target set), and a separate hold-out data set (called the
public set) in our framework. The framework outputs a suite of metrics to
quantify déjà vu memorization at both population level and sample level. The
framework also outputs the most memorized samples from the target training set.


## Installation Instructions

1. Create a virtual environment:

```
$ python3 -m venv env
$ source env/bin/activate
```

2. Install the dependencies in the order given in the requirements.txt:

`$ cat requirements.txt  | xargs -n 1 -L 1 python3 -m pip install`

3. Install the Detic package:

```
$ git clone https://github.com/facebookresearch/Detic.git --recurse-submodules
$ cd Detic
$ python3 -m pip install -r requirements.txt
```


## Object Annotation

VLMDejaVu requires object annotations in images to quantify memorization.
To obtain object annotations for your dataset images, provide the image folder
path to the `src/annotation.py` script.

```
$ python3 src/annotation.py \
    --images_path="/path/to/images/folder" \
    --save_path="/file/path/to/save/annotations.p"
```

Alternative, you can edit and run the following slurm script:

`$ sbatch scripts/annotation.sh`


## Utility Evaluation

The framework also provides utility benchmarking of target CLIP models via
linear probe accuracy and ImageNet zero-shot accuracy. To evaluate the linear
probe accuracy, run the following command:

```
$ python3 src/linear_probe.py \
    --train_images_path="/path/to/training/images/folder" \
    --val_images_path="/path/to/validation/images/folder" \
    --checkpoint_path="/path/to/model/checkpoint" \
    --model_type="ViT-B-32" \
    --emb="pj" \
    --normalize
```

To evaluate the ImageNet zero-shot accuracy, run the following command:

```
$ python3 src/zero_shot_imagenet.py \
    --images_path="/path/to/imagenet/folder" \
    --checkpoint_path="/path/to/model/checkpoint" \
    --model_type="ViT-B-32"
```

Alternative, you can edit and run the following slurm scripts:

`$ sbatch scripts/linear_proble.sh`

`$ sbatch scripts/zero_shot_imagenet.sh`


## Memorization Attack

Déjà Vu memorization quantifies the object memorization gap between the target
model and a reference model. VLMDejaVu framework provides two types of
memorization tests:

* Two-Model Test: Requires two CLIP models, a target model (trained on target
training set) and a reference model (trained on a separate training set),
access to the target training set and a public set of image--caption pairs.

* One-Model Test: Requires only the target CLIP model, access to the target
training set and a public set. For reference model, the framework uses a
pretrained LLM model.

To run the two-model test on CLIP target and reference models, run the
following command:

```
$ python3 src/dejavu_attack.py \
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
```

Alternatively, you can edit and run the following slurm script:

`$ sbatch scripts/two_model_test.sh`

To run the one-model test on the target CLIP model (using pretrained LLM as
reference model), run the following command:

```
$ python3 src/dejavu_attack.py \
    --tgt_model_path="/path/to/target/model/checkpoint" \
    --model_type="ViT-B-32" \
    --llm_type="gte-base" \
    --target_images="/path/to/target/set/images/folder" \
    --public_images="/path/to/public/set/images/folder" \
    --target_captions="/path/to/target/set/captions/file" \
    --public_captions="/path/to/public/set/captions/file" \
    --target_annotations="/path/to/target/set/annotations/file.p" \
    --public_annotations="/path/to/public/set/annotations/file.p" \
    --k=10 \
    --score_fn="max_pred_gap" \
    --one_model_test \
    --text_index \
```

Alternatively, you can also edit and run the following slurm script:

`$ sbatch scripts/one_model_test.sh`

You can also evaluate the memorization in pretrained CLIP models with the
one-model test:

```
$ python3 src/dejavu_attack.py \
    --tgt_model_path="" \
    --pretrained="openai" \
    --model_type="RN50" \
    --llm_type="gte-base" \
    --target_images="/path/to/target/set/images/folder" \
    --public_images="/path/to/public/set/images/folder" \
    --target_captions="/path/to/target/set/captions/file" \
    --public_captions="/path/to/public/set/captions/file" \
    --target_annotations="/path/to/target/set/annotations/file.p" \
    --public_annotations="/path/to/public/set/annotations/file.p" \
    --k=10 \
    --score_fn="max_pred_gap" \
    --one_model_test \
    --text_index \
```

Alternatively, you can also edit and run the following slurm script:

`$ sbatch scripts/one_model_test_oob.sh`


## License

VLMDejaVu is licensed under the CC-BY-NC 4.0 license.
