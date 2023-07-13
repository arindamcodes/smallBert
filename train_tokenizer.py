import os
import sys
import tqdm
import pandas  as pd
from tokenizers import BertWordPieceTokenizer
from pathlib import Path



# Pairs data file path
filepath = os.path.join(os.getcwd(), "pairs_bert.pkl")

# Load the pairs data
pairs = pd.read_pickle(filepath)


# Create chunks batches of text files
os.mkdir('./data')
text_data = []
file_count = 0

for sample in tqdm.tqdm([x[0] for x in pairs]):
    text_data.append(sample)

    # once we hit the 10K mark, save to file
    if len(text_data) == 10000:
        with open(f'./data/text_{file_count}.txt', 'w', encoding='utf-8') as fp:
            fp.write('\n'.join(text_data))
        text_data = []
        file_count += 1


# Get list of paths of text files
paths = [str(x) for x in Path('./data').glob('**/*.txt')]


# train tokenizer
tokenizer = BertWordPieceTokenizer(
    clean_text=True,
    handle_chinese_chars=False,
    strip_accents=False,
    lowercase=True
)

tokenizer.train( 
    files=paths,
    vocab_size=30_000, 
    min_frequency=5,
    limit_alphabet=1000, 
    wordpieces_prefix='##',
    special_tokens=['[PAD]', '[CLS]', '[SEP]', '[MASK]', '[UNK]']
    )


# Save the tokenizer for future use cases
os.mkdir('./bert-ari')
tokenizer.save_model('./bert-ari', 'bert-it')
