import os
import subprocess

class CORD19Downloader:
    def __init__(self, save_path):
        self.save_path = os.path.join(save_path, 'cord19')
        
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        self.download_url = 'https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/historical_releases/cord-19_2020-10-01.tar.gz'

    def download(self):
        doc_parses_tarfile = os.path.join(self.save_path, 'document_parses.tar.gz')
        doc_parses_dir = os.path.join(self.save_path, 'document_parses')

        if os.path.isdir(doc_parses_dir):
            print('CORD19 data directory ({}) already exists. Skipping.'.format(
                doc_parses_dir))
            return

        if not os.path.isfile(doc_parses_tarfile):
            download_cmd = 'wget -c {url} -O - | tar -xz -C {dest} --strip-components 1'.format(
                    url=self.download_url, dest=self.save_path)
            subprocess.run(download_cmd, shell=True, check=True)

        extract_document_cmd = 'tar -xzf {tarfile} -C {dest}'.format(
                tarfile=doc_parses_tarfile, dest=self.save_path)
        subprocess.run(extract_document_cmd, shell=True, check=True)

