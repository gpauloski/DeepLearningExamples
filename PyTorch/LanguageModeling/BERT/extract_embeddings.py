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
from run_pretraining import *
from modeling import BertModel, BertConfig

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--ckpt_file", default=None, type=str, required=True)
    parser.add_argument("--output_file", default=None, type=str, required=True)
    parser.add_argument("--config_file", default=None, type=str, required=True)
    args = parser.parse_args()

    # Prepare model
    config = BertConfig.from_json_file(args.config_file)

    # Padding for divisibility by 8
    if config.vocab_size % 8 != 0:
        config.vocab_size += 8 - (config.vocab_size % 8)

    model = modeling.BertForPreTraining(config)
    checkpoint = torch.load(args.ckpt_file, map_location="cpu")

    model.load_state_dict(checkpoint['model'], strict=False)

    embedding_layer = model.bert.embeddings.word_embeddings

    print('Embedding shape: {}'.format(embedding_layer.weight.shape))

    weights = embedding_layer.weight.detach().numpy()
    np.save(args.output_file, weights)

    print('Embeddings saved to {}'.format(args.output_file))

if __name__ == "__main__":
    main()
