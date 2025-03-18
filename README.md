Text Normalization Project

A comprehensive implementation of text normalization combining the power of T5 models with rule-based preprocessing techniques.
This repository provides a complete workflow for data cleaning, augmentation, model training, and inference.

## Table of Contents
---------------

* [Overview](#overview)
* [Installation](#installation)
* [Data Preparation](#data-preparation)
* [Usage](#usage)
  * [Training](#training)
  * [Testing](#testing)
  * [Inference](#inference)
  * [Output Examples](#output-examples)
* [Assessment's Reports](#assessments-reports)

## Overview
--------

This project implements a text normalization pipeline featuring:

| Component | Description |
|-----------|-------------|
| Model     | T5 transformer from HuggingFace |
| Preprocessing | Rule-based cleaning and augmentation |
| Output    | Normalized text with quality metrics |

### Key Features

- Data cleaning and preprocessing
-  Data augmentation
-  Model training with early stopping
-  Comprehensive evaluation metrics
-  Command-line interface

## Installation

### Clone Repository

```bash
git clone git@github.com:amitsou/T5-TextNormalizer.git
cd T5-TextNormalizer
```

### Set Up Environment

Create and activate the virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Data Preparation

### Dataset Setup

Place your CSV file in the following directory:
```markdown
root/
├── data/
│   └── raw_data/
│       └── normalization_assessment_dataset_10k.csv
└── ...
```

### Data Processing

Process your dataset using the preparation script. This script
split the dataset into train/test/val using the (80/10/10) rule.

In order to retrain the T5, a data augmentation approach has been conducted.
Thus, the --augment argument controls the data augmentation during the data preparation phase.

When you execute the script with --prepare --augment N, it creates N additional
copies of each training example with controlled modifications

For example:

```bash
python main.py --prepare --augment 2
```

## Usage

### Training

Train the T5 model:

In order to customize the model's parameters, please adjsut the model_params.py file.
For the moment the configuration is adjusted for an RTX3060 GPU.
The model will be trained for 5 epochs.

```bash
python main.py --train
```

This command:

### Testing

Evaluate the model:

```bash
python main.py --test --samples 100
```

The script will:
- Load the test data
- Randomly select 100 samples
- Generate predictions for each sample
- Calculate all metrics

Metrics reported:
The metrics calculated provide different perspectives on the model's performance:

- BLEU Score: Measures how well the predicted text matches the ground truth, focusing on word order and accuracy

- Character Accuracy: Shows the percentage of characters that match exactly between prediction and ground truth

- Word Accuracy: Measures the percentage of words that match exactly

- Normalized Edit Distance: Shows how many operations (insertions, deletions, substitutions) are needed to transform the prediction into the ground truth

### Inference
Run inference on standard example inputs:

```bash
python app.py --inference
```

The aforementioned command generates normalized text output for built-in (given) examples.


### Output Examples

#### Inference examples:

| RAW TEXT | PREDICTED NORMALIZED |
|----------|----------------------|
| Pixouu/Abdou Gambetta/Copyright Control | Pixouu/Abdou Gambetta |
| Mike Hoyer/JERRY CHESNUT/SONY/ATV MUSIC PUBLISHING (UK) LIMITED | Correct and normalize: Mike Ho |


#### Model Test/Evaluation Examples:

#####  Evaluation Results Table

| Metric                      | Score  |
|-----------------------------|--------|
| BLEU Score                 | 0.1658 |
| Character-level Accuracy   | 0.4467 |
| Word-level Accuracy        | 0.4431 |
| Normalized Edit Distance   | 0.4197 |

##### Example Predictions Table

| Input | Predicted | Actual |
|--------------------------------|----------------|----------------|
| Paweł Jabłoński | Pawe Jaboski | Paweł Jabłoński |
| Yuki Kishida/Kentaro Sonoda | *No Prediction* | Yuki Kishida/Kentaro Sonoda |
| R.K.M./Nico Gomez/Universal Music publish GmbH/Universal Music Publishing N.V./Universal Music Publishing Gmbh | Nico Gomez | Nico Gomez |


## Hardware Limitations

The development setup consists of the following hardware specifications:

- **Laptop** with **32GB RAM**
- **Intel Core i7** Processor
- **NVIDIA RTX 3060 (6GB VRAM)**

Due to these hardware constraints, particularly the **limited GPU memory (6GB VRAM)**, challenges were faced in running large-scale deep learning models and high-resolution experiments efficiently.

As a result, some model training and evaluations were conducted with optimizations for **lower VRAM consumption**, and certain large-scale experiments were not feasible within this setup.


In order to learn more regarding the task, consider reading the following:

- [Text Normalization Report](/docs/Text_Normalization_Report.pdf)
