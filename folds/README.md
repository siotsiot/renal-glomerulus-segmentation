# Fold file contract

The clean production experiment reads pre-existing five-fold assignments. The
original CSV files are not included because their values are identifiers from a
non-public clinical dataset.

## Expected structure

```text
folds/
├── fold_1/
│   ├── paired_ids.csv
│   ├── train_ids.csv
│   └── val_ids.csv
├── fold_2/
│   └── ...
├── fold_3/
│   └── ...
├── fold_4/
│   └── ...
└── fold_5/
    └── ...
```

Every CSV must contain exactly one column:

```csv
id
```

Values are file stems without extensions. For example, an authorized ID
`example_001` refers to `images/example_001.png` and
`masks/example_001.tiff`.

## Roles

- `paired_ids.csv`: all paired image-mask IDs available to that fold.
- `train_ids.csv`: IDs used for training.
- `val_ids.csv`: IDs used for validation and the fixed primary evaluation.

Within each fold, training and validation IDs must be disjoint, contain no
duplicates, and have a union equal to `paired_ids.csv`. All four architectures
must receive the same files for a given fold.

## Original aggregate sizes

| Fold | Training | Validation | Paired total |
|---:|---:|---:|---:|
| 1 | 288 | 73 | 361 |
| 2 | 289 | 72 | 361 |
| 3 | 289 | 72 | 361 |
| 4 | 289 | 72 | 361 |
| 5 | 289 | 72 | 361 |

These counts do not disclose the underlying identifiers. Patient/group
independence of the original assignments was not verifiable.

## Preparing files for another dataset

For an independently authorized dataset:

1. pair each image and mask by a unique shared stem;
2. create one deterministic five-fold partition outside this repository;
3. write the three one-column CSV files for every fold;
4. verify disjoint train/validation sets and full paired-ID coverage;
5. reuse each fold's CSV files unchanged across all four architectures.

Do not commit generated fold CSVs. The repository `.gitignore` excludes
`folds/**/*.csv` because identifiers may be sensitive.

