# Datasets Directory

This folder is designated for storing optional training and evaluation benchmark datasets.

Please see [`../DATASETS.md`](../DATASETS.md) for the official Dataset Manifest, licenses, URLs, train/val/test split protocols, and evaluation metrics.

## Recommended Directory Structure for Local Datasets

```
datasets/
├── rwf2000/
│   ├── train/
│   │   ├── Fight/
│   │   └── NonFight/
│   └── val/
│       ├── Fight/
│       └── NonFight/
├── urfd/
│   ├── falls/
│   └── adl/
└── ucf_crime_snatch/
    ├── snatch/
    └── normal/
```

*Note: Raw video datasets are kept out of Git version control. Pretrained baseline models are loaded automatically.*
