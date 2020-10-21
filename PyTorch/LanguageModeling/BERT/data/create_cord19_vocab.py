import sentencepiece as spm
import tqdm

from nltk import tokenize

input_file = 'formatted_one_article_per_line/cord19_one_article_per_line.txt'
output_file = 'm.input'

#with open(input_file) as fi:
#    with open(output_file, 'w') as fo:
#        for line in tqdm.tqdm(fi.readlines()):
#            sentences = tokenize.sent_tokenize(line)
#            for sentence in sentences:
#                fo.write(sentence + '\n')

spm.SentencePieceTrainer.train('--input=m.input --model_prefix=m '
        '--vocab_size=30000 --max_sentence_length=60000 '
    )

sp = spm.SentencePieceProcessor()
sp.load('m.model')
