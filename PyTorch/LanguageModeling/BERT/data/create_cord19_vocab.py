import sentencepiece as spm
import nltk.data

input_file = 'formatted_one_article_per_line/cord19_one_article_per_line.txt'
output_file = 'm.input'

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
with open(input_file) as fi:
    data = fi.read()
    sentences = tokenizer.tokenize(data)
    with open(sentence_file, 'w') as fo:
        fo.write(sentences)

spm.SentencePieceTrainer.train('--input=m.input --model_prefix=m '
        '--vocab_size=30000 --max_sentence_length=50000'
    )

sp = spm.SentencePieceProcessor()
sp.load('m.model')
