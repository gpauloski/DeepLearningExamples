import sentencepiece as spm
import tqdm

from nltk import tokenize

input_file = 'formatted_one_article_per_line/cord19_one_article_per_line.txt'
raw_vocab_file = 'm.input'
vocab_file = '../vocab/cord19_vocab.txt'

with open(input_file) as fi:
    with open(raw_vocab_file, 'w') as fo:
        for line in tqdm.tqdm(fi.readlines()):
            sentences = tokenize.sent_tokenize(line)
            for sentence in sentences:
                fo.write(sentence + '\n')

spm.SentencePieceTrainer.train('--input=m.input --model_prefix=m '
        '--vocab_size=30000 --max_sentence_length=60000 --model_type=unigram '
        '--shuffle_input_sentence=true --input_sentence_size=10000 '
        '--add_dummy_prefix=false --hard_vocab_limit=false'
    )

sp = spm.SentencePieceProcessor()
sp.load('m.model')

vocabs = [sp.id_to_piece(id) for id in range(sp.get_piece_size())]

with open(vocab_file, 'w') as f:
    for vocab in vocabs:
        f.write(vocab + '\n')
print('Vocab written to ' + vocab_file)
