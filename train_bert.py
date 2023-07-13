import os
import torch
import torch.nn as nn
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import dataloader
import model


batch_size = 32
context_len = 512
d_k = 512
n_heads = 8
n_layers = 6
epochs = 2
device = "mps" if torch.backends.mps.is_available() else "cpu"

train_file_path = os.path.join(os.getcwd(), "pairs_bert.pkl")
# Load the pairs data
pairs = pd.read_pickle(train_file_path)

# split the data in train, test and val
train_data, test_data = train_test_split(pairs, test_size=0.2, random_state=42)
train_data, val_data = train_test_split(train_data, test_size=0.25, random_state=42)


# Init pretained tokenizer
tokenizer = BertTokenizer.from_pretrained('./bert-ari/bert-it-vocab.txt', local_files_only=True)

# Creating Bert Datasets
train_dataset = dataloader.BERTDataset(train_data, tokenizer)
val_dataset = dataloader.BERTDataset(val_data, tokenizer)
test_dataset = dataloader.BERTDataset(test_data, tokenizer)

# Creating dataloaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# Init bert model
bert = model.Bert(tokenizer.vocab_size, context_len, d_k, n_heads, n_layers, device).to(device)

# Init optimizer
optimizer = torch.optim.AdamW(bert.parameters(), lr=0.001)

# Init loss function
criterion = nn.CrossEntropyLoss(ignore_index=0)



for epoch in range(epochs):
    bert.train()
    total_loss = 0

    progress_bar = tqdm(train_loader, desc='Epoch {:03d}'.format(epoch), leave=False, disable=False)
    for batch in progress_bar:
        x = batch['bert_input'].to(device)
        y = batch['segment_label'].to(device)
        target = batch['bert_label'].to(device)
        nsp_target = batch['is_next'].to(device)

        logits, nsp = bert(x, y)

        loss_mlm = criterion(logits.view(-1, tokenizer.vocab_size), target.view(-1))
        loss_nsp = criterion(nsp, nsp_target)
        loss = loss_mlm + loss_nsp

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item()/len(batch))})

    torch.mps.empty_cache()

    tqdm.write('Epoch {:03d} training loss: {:.3f}'.format(epoch, total_loss/len(train_loader)))

    bert.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            x = batch['bert_input'].to(device)
            y = batch['segment_label'].to(device)
            target = batch['bert_label'].to(device)
            nsp_target = batch['is_next'].to(device)

            logits, nsp = bert(x, y)

            loss_mlm = criterion(logits.view(-1, logits.shape[-1]), target.view(-1))
            loss_nsp = criterion(nsp, nsp_target)
            val_loss = loss_mlm + loss_nsp

            total_val_loss += val_loss.item()

    print(f"Epoch {epoch}, Validation loss: {total_val_loss/len(val_loader)}")

# Save the model after training
torch.save(bert.state_dict(), "model_after_training.pt")

# Testing the model after training
bert.eval()
total_test_loss = 0
with torch.no_grad():
    for batch in test_loader:
        x = batch['bert_input'].to(device)
        y = batch['segment_label'].to(device)
        target = batch['bert_label'].to(device)
        nsp_target = batch['is_next'].to(device)

        logits, nsp = bert(x, y)

        loss_mlm = criterion(logits.view(-1, logits.shape[-1]), target.view(-1))
        loss_nsp = criterion(nsp, nsp_target)
        test_loss = loss_mlm + loss_nsp

        total_test_loss += test_loss.item()

print(f"Test loss: {total_test_loss/len(test_loader)}")



