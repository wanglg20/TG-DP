# Semantic Noise Reduction via Teacher-Guided Dual-Path Audio-Visual Representation Learning

Official implementation of the CVPR 2026 paper  
**Semantic Noise Reduction via Teacher-Guided Dual-Path Audio-Visual Representation Learning**

[[Paper]](https://arxiv.org/abs/2604.08147)

---

## Table of Contents

- [Introduction](#introduction)
- [TODO List](#todo-list)
- [Model Checkpoints and Dataset Preparation](#Model-Checkpoints-and-Dataset-Preparation)
- [Retrieval Guideline](#retrieval-guideline)
- [Acknowledgement](#acknowledgement)

---

## Introduction

Cross-modal contrastive learning often suffers from semantic noise introduced by modality-private or weakly correlated regions.  
This project proposes a teacher-guided dual-path framework for audio-visual representation learning, which decouples reconstruction and alignment objectives and further introduces attribution-guided subview learning to improve cross-modal semantic correspondence.

<!-- <p align="center">
  <img src="docs/pipeline.pdf" width="100%">
</p> -->

For more details, please refer to our paper:

> **Semantic Noise Reduction via Teacher-Guided Dual-Path Audio-Visual Representation Learning**  
> CVPR 2026  
> [https://arxiv.org/abs/2604.08147](https://arxiv.org/abs/2604.08147)

## TODO List

- [✔] Retrieval evaluation pipeline
- [✔] Pretrained checkpoints
- [ ] Data preprocessing pipeline
- [ ] Pretraining scripts
- [ ] Classification finetuning



---
## Model Checkpoints and Dataset Preparation

### Pretrained Checkpoints

The pretrained retrieval checkpoint can be downloaded from Google Drive:

- [TG-DP Retrieval Checkpoint](https://drive.google.com/file/d/188I2Lkm8oQfyZsCwgm2BhyKrCd9Fkwbc/view?usp=drive_link)

---

### Datasets

#### VGGSound

We use the VGGSound benchmark for retrieval evaluation.

The processed dataset can be downloaded from HuggingFace:

- [VGGSound Dataset](https://huggingface.co/datasets/Loie/VGGSound/tree/main)

---

#### AudioSet-2M

Due to copyright restrictions, we are unable to redistribute the original AudioSet videos.

Please download and preprocess AudioSet following the official instructions provided by the original dataset/paper resources.


### Data preprocessing pipeline

Our data preprocessing pipeline generally follows the official CAV-MAE implementation:

- [CAV-MAE Data Preparation](https://github.com/yuangongnd/cav-mae#data-preparation)

For efficient large-scale training, we additionally adopt WebDataset-based data packaging to improve I/O throughput during distributed training.

The corresponding preprocessing and packaging scripts are still being organized and will be released in a future update.


## Retrieval Guideline

### Environment Setup

Create the conda environment with:

```bash
conda env create -f environment.yml
conda activate TG-DP
```

### Prepare Retrieval Metadata Files

Generate the required json metadata files with:
```bash
bash src/data_info/datafiles_generate/generate_datafiles.sh
```
For convenience, we also provide the metadata files used in our experiments under `src/data_info`
### Configure Paths
Before evaluation, please modify the checkpoint path and dataset path in `eval_script/retrieval.sh`
### Run Retrieval Evaluation
After configuration, run:
```bash
bash eval_script/retrieval.sh
```



## Acknowledgement

Our implementation is heavily based on the following excellent open-source projects:

- [CAV-MAE](https://github.com/yuangongnd/cav-mae)
- [CAV-MAE Sync](https://github.com/edsonroteia/cav-mae-sync)

We sincerely thank the authors for making their code publicly available.