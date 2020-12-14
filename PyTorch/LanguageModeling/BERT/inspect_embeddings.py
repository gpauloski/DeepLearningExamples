# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Extract embeddings from a PyTorch BERT model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import torch
import os
import numpy as np
import modeling
from modeling import BertModel, BertConfig

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--embeddings_file", default=None, type=str, required=True)
    parser.add_argument("--vocab_file", default=None, type=str, required=True)
    args = parser.parse_args()

    weights = np.load(args.embeddings_file)
    print('Loaded weights: {}'.format(weights.shape))

    vocab = []
    with open(args.vocab_file) as f:
        for line in f.readlines():
            vocab.append(line.strip())

    print('Loaded vocab: {} words'.format(len(vocab)))

    covid_idx = vocab.index('covid')
    covid19_idx = vocab.index('covid19')
    cow_idx = vocab.index('cow')
    cows_idx = vocab.index('cows')
    select_idx = vocab.index('select')
    choose_idx = vocab.index('choose')

    covid = weights[covid_idx]
    covid19 = weights[covid19_idx]
    cow = weights[cow_idx]
    cows = weights[cows_idx]
    select = weights[select_idx]
    choose = weights[choose_idx]

    print('||covid - covid19||', np.linalg.norm(covid - covid19))
    print('||covid - cow||', np.linalg.norm(covid - cow))
    print('||cow - cows||', np.linalg.norm(cow - cows))
    print('||select - choose||', np.linalg.norm(select - choose))


    w = torch.from_numpy(weights).cuda()
    w = w.unsqueeze(0).half()

    distances = torch.cdist(w, w)
    print('max distance', torch.max(distances))
    distances = torch.where(distances !=0, torch.zeros_like(distances), distances)
    print('min distance', torch.min(distances))

if __name__ == "__main__":
    main()
