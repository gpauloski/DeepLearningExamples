from tokenizers import BertWordPieceTokenizer

input_file = 'formatted_one_article_per_line/cord19_one_article_per_line.txt'
vocab_file = '../vocab/cord19_500k_vocab.txt'

# Initialize an empty BERT tokenizer
tokenizer = BertWordPieceTokenizer(
  clean_text=False,
  handle_chinese_chars=False,
  strip_accents=False,
  lowercase=True,
)

# prepare text files to train vocab on them
files = [input_file]

# train BERT tokenizer
tokenizer.train(
  files,
  vocab_size=500000,
  min_frequency=2,
  show_progress=True,
  special_tokens=['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'],
  limit_alphabet=1000,
  wordpieces_prefix="##"
)

# save the vocab
vocab = [v for v in tokenizer.get_vocab()]
vocab.sort()
with open(vocab_file, 'w') as f:
    for v in vocab:
        f.write(v + '\n')

