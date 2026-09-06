# Data acquisition and layout

## DSB2018

Source: [2018 Data Science Bowl on Kaggle](https://www.kaggle.com/competitions/data-science-bowl-2018).

The experiments use the labelled `stage1_train.zip` archive. A Kaggle account
and acceptance of the competition terms may be required.

Using the Kaggle CLI:

```bash
python -m pip install kaggle
mkdir -p data/dsb2018
kaggle competitions download -c data-science-bowl-2018 \
  -f stage1_train.zip -p data/dsb2018
unzip data/dsb2018/stage1_train.zip -d data/dsb2018/stage1_train
```

Expected structure:

```text
data/dsb2018/stage1_train/
└── <image-id>/
    ├── images/<image-id>.png
    └── masks/<instance-id>.png
```

The full labelled archive contains 670 image-ID directories. The scripts sort
the IDs before applying seeded shuffling.

The `stage1_train.zip` archive available during the project had SHA-256 digest
`dcb6edc2690f137406638b2309581a71522c4dff19157d118453b448dcddcb68`.

## RxRx1

Sources:

- [RxRx1 dataset page](https://www.rxrx.ai/rxrx1)
- [Official RxRx1 README](https://github.com/recursionpharma/rxrx-datasets/tree/trunk/rxrx1)
- [RxRx1 dataset licence](https://www.rxrx.ai/recursion-dataset-license)

The image archive is approximately 45-47 GB. The released data are licensed
CC-BY-NC-SA 4.0; follow the provider's current licence terms.

```bash
mkdir -p data/rxrx1
curl -L https://storage.googleapis.com/rxrx/rxrx1/rxrx1-metadata.zip \
  -o data/rxrx1/rxrx1-metadata.zip
curl -L https://storage.googleapis.com/rxrx/rxrx1/rxrx1-images.zip \
  -o data/rxrx1/rxrx1-images.zip
unzip data/rxrx1/rxrx1-metadata.zip -d data/rxrx1
unzip data/rxrx1/rxrx1-images.zip -d data/rxrx1
```

Expected structure after extraction:

```text
data/rxrx1/rxrx1/
├── metadata.csv
└── images/
    └── HUVEC-1/Plate1/M23_s2_w1.png
```

The project selects the first 500 sorted `*_w1.png` paths for the
pseudo-labelling experiments. Preserve the provider's directory and filenames;
changing them changes the selected subset.

The `rxrx1-metadata.zip` archive available during the project had SHA-256
digest `8919a2f5a09d428c8b1447febeb356441441edaf621734ec5d282e723202a36f`.
The full image archive was not retained with the repository, so no local digest
is asserted for that file.

## External data locations

If data are stored elsewhere, set environment variables before running any
script:

```bash
export DSB2018_DIR=/data/dsb2018/stage1_train
export RXRX1_ROOT=/data/rxrx1
```

`RXRX1_IMAGES_DIR` and `RXRX1_METADATA` may be set separately when the image and
metadata files are not under one root.

## Validation

```bash
python scripts/check_data.py
```

This checks the expected directory structure, counts DSB image IDs, and confirms
that RxRx1 contains metadata and at least 500 channel-1 images.

Downloaded archive digests can be checked with:

```bash
shasum -a 256 stage1_train.zip rxrx1-metadata.zip
```
