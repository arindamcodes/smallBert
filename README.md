# BERT from Scratch with PyTorch

This repository contains code for training BERT (Bidirectional Encoder Representations from Transformers) from scratch using PyTorch and Python 3.10.

BERT is a transformer-based machine learning technique for natural language processing (NLP) pre-training developed by Google.

## Prerequisites

The project uses the following dependencies:

- Python 3.10
- PyTorch
- Transformers (just to train tokenizer)
  

You can install the necessary dependencies using `pip` (conda virtual env is highly recommended):

```bash
pip install -r requirements.txt
```

## Project Structure
This project contains the following key files:

- requirements.txt: This file lists all necessary Python packages, compatible with Python 3.10.
- pairs_bert.pkl: This pickle file contains sentence pairs in the format [[sent1, sent2], ....]. If you want to train with your own data, please format it in this way and pickle dump it with the same name.
- train_tokenizer.py: This script trains the tokenizer based on the sentence pairs in pairs_bert.pkl.
- dataloader.py: This script uses the trained tokenizer to create a BERT dataset compatible with PyTorch.
- model.py: This script defines the BERT model architecture.
- train_bert.py: This script trains the BERT model on the prepared dataset and evaluates the trained model. You can adjust the batch_size, n_head, and epochs according to your needs


## Training
```bash
python train_tokenizer.py
python dataloader.py
python train_bert.py
```

## Contribution
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
- Apache License



