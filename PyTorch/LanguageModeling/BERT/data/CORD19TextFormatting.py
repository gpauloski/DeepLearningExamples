import argparse
import glob
import json
import tqdm
import re


def get_value(obj, key):
    if key in obj:
        return obj[key]
    for k, v in obj.items():
        if isinstance(v, dict):
            value = get_value(v, key)
            if value is not None:
                return value


def get_text(filepath):
    keys = ['abstract', 'body_text']

    text = ''
    with open(filepath) as json_file:
        json_data = json.load(json_file)
        for sections in (get_value(json_data, k) for k in keys):
            if sections is not None:
                for section in sections:
                    if 'text' in section:
                        text += format_line(section['text']) + ' '

    return text


def format_line(line):
    line = line.strip()
    line = line.replace('\n', '')
    line = re.sub(r'\[([0-9],? ?)+\]', ' ', line)
    line = line.replace('  ', ' ')
    return line


class CORD19TextFormatter:
    def __init__(self, data_path, output_file):
        self.data_path = data_path
        self.output_file = output_file

    def merge(self):
        count = 0
        filepaths = list(glob.iglob(self.data_path + '/**/*.json', recursive=True))
        with open(self.output_file, mode='w', newline='\n') as f:
            for filepath in tqdm.tqdm(filepaths):
                text = get_text(filepath)
                f.write(text + '\n\n')
                count += 1

        print('Parsed {} files in {}'.format(count, self.data_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CORD-19 Text Formatter"
    )

    parser.add_argument('--document_parses', type=str,
        default='cord19/document_parses',
        help='Directory containing the document parses'
    )
    parser.add_argument('--formatted_file', type=str,
        default='cord19/cord19_one_article_per_line.txt',
        help='Output text file where each lines contains the text of one '
             'article found in --directory'
    )
    args = parser.parse_args()

    formatter = CORD19TextFormatter(
            data_path=args.document_parses, 
            output_file=args.formatted_file)
    formatter.merge()


