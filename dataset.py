# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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

import logging
import os

from dataclasses import dataclass
from typing import Optional
import json
import copy
import numpy
import pickle
# import spacy
# import difflib
# from utils import coref_re,string_similar
from tqdm import tqdm

logger = logging.getLogger(__name__)


def norm_mask(input_mask):
    output_mask = numpy.zeros(input_mask.shape)
    for i in range(input_mask.shape[0]):
        for j in range(input_mask.shape[1]):
            if not numpy.all(input_mask[i][j] == 0):
                output_mask[i][j] = input_mask[i][j] / sum(input_mask[i][j])
    return output_mask

docred_rel2id = json.load(open('data/DocRED/rel2id.json', 'r'))


def docred_convert_examples_to_features(args,
    examples,
    model_type,
    tokenizer,
    max_length=1024,
    max_ent_cnt=42,
    label_map=None,
    pad_token=0,
):


    """
    Loads a data file into a list of ``InputFeatures``

    Args:
        examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.
        tokenizer: Instance of a tokenizer that will tokenize the examples
        max_length: Maximum example length
        task: GLUE task
        label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
        output_mode: String indicating the output mode. Either ``regression`` or ``classification``
        pad_on_left: If set to ``True``, the examples will be padded on the left rather than on the right (default)
        pad_token: Padding token
        pad_token_segment_id: The segment ID for the padding token (It is usually 0, but can vary such as for XLNet where it is 4)
        mask_padding_with_zero: If set to ``True``, the attention mask will be filled by ``1`` for actual values
            and by ``0`` for padded values. If set to ``False``, inverts it (``1`` for padded values, ``0`` for
            actual values)

    Returns:
        If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset``
        containing the task-specific features. If the input is a list of ``InputExamples``, will return
        a list of task-specific ``InputFeatures`` which can be fed to the model.

    """
    # 记录处理后超过max_length的文档
    more_len = []
    features = []
    pos_samples = 0
    neg_samples = 0

    ner_map = {'PAD':0, 'ORG':1, 'LOC':2, 'NUM':3, 'TIME':4, 'MISC':5, 'PER':6}
    distance_buckets = numpy.zeros((512), dtype='int64')
    distance_buckets[1] = 1
    distance_buckets[2:] = 2
    distance_buckets[4:] = 3
    distance_buckets[8:] = 4
    distance_buckets[16:] = 5
    distance_buckets[32:] = 6
    distance_buckets[64:] = 7
    distance_buckets[128:] = 8
    distance_buckets[256:] = 9

    for (ex_index, example) in enumerate(examples):

        len_examples = len(examples)


        if ex_index % 500 == 0:
            logger.info("Writing example %d/%d" % (ex_index, len_examples))



        input_tokens = []
        tok_to_sent = []
        tok_to_word = []
        sent_map = []
        sent_pos = [] # 记录文档每个句子对应sents的首尾下标
        # cls feature
        max_sent_cnt = 25
        cls_mask = numpy.zeros((max_sent_cnt, max_length), dtype='int64') # 记录每个句子首个token的起始下标

        entities = example['vertexSet']
        entity_start, entity_end = [], []  # 文档中实体提及的起始位置和终止位置
        for entity in entities:
            for mention in entity:
                sent_id = mention["sent_id"]  # 实体提及句子id
                pos = mention["pos"]  # 实体提及位置
                entity_start.append((sent_id, pos[0],))
                entity_end.append((sent_id, pos[1] - 1,))  # 结束下标
        sent_start = 0
        for i_s, sent in enumerate(example['sents']):
            new_map = {}

            for i_t, token in enumerate(sent):
                tokens_wordpiece = tokenizer.tokenize(token)
                if (i_s, i_t) in entity_start:
                    tokens_wordpiece = ["*"] + tokens_wordpiece
                if (i_s, i_t) in entity_end:
                    tokens_wordpiece = tokens_wordpiece + ["*"]
                new_map[i_t] = len(input_tokens)
                input_tokens.extend(tokens_wordpiece)
                tok_to_word += [i_t] * len(tokens_wordpiece)
                tok_to_sent += [i_s] * len(tokens_wordpiece)
            new_map[i_t + 1] = len(input_tokens) # 记录这句话结束的token下标
            sent_map.append(new_map)
            sent_end = len(input_tokens)
            sent_pos.append((sent_start,sent_end,))
            sent_start = sent_end

        # sents = sents[:max_seq_length - 2]
        # if len(input_tokens)>max_length-2:
        #     more_len.append((ex_index,len(input_tokens),))
        input_ids = tokenizer.convert_tokens_to_ids(input_tokens[:max_length - 2])
        input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)
        attention_mask = [1] * len(input_ids)
        token_type_ids = [0] * len(input_ids)



        for i_s in range(cls_mask.shape[0]):
            if i_s < len(sent_map):
                cls_mask[i_s][sent_map[i_s][0]] = 1

        # ent_mask & ner / coreference feature
        ent_mask = numpy.zeros((max_ent_cnt, max_length), dtype='int64')
        max_men_cnt = 27
        ent_men_mask = numpy.zeros((max_ent_cnt, max_men_cnt, max_length), dtype='int64')
        ent_ner = [0] * max_length  # 每个token对应实体类型
        ent_pos = [0] * max_length  # 每个tokne对应实体序号+1
        tok_to_ent = [-1] * max_length

        entity_pos = [] # 列表列表，文档实体提及在sents的位置下标元组
        for i_e, e in enumerate(entities):
            entity_pos.append([])
            for i_m, m in enumerate(e):
                if i_m >= max_men_cnt:
                    break
                # print(m["pos"])
                # print(m)
                start = sent_map[m["sent_id"]][m["pos"][0]]
                end = sent_map[m["sent_id"]][m["pos"][1]] # sent_map在每个句子key都加一个
                entity_pos[-1].append((start, end,))
                ent_mask[i_e][start: end] = 1
                ent_men_mask[i_e][i_m][start: end] = 1

                # ent_ner[start: end] = ner_map[e[0]['type']]
                # ent_pos[start: end] = i_e + 1
                # tok_to_ent[start: end] = i_e



        ents = example['vertexSet']

        label_ids = numpy.zeros((max_ent_cnt, max_ent_cnt, len(label_map.keys())), dtype='bool')
        # test file does not have "labels"
        if 'labels' in example: # 正例
            labels = example['labels']
            for label in labels:
                label_ids[label['h']][label['t']][label_map[label['r']]] = 1
        for h in range(len(ents)): # 负例
            for t in range(len(ents)):
                if numpy.all(label_ids[h][t] == 0):
                    label_ids[h][t][0] = 1
        # 标记可以关系抽取的实体对
        label_mask = numpy.zeros((max_ent_cnt, max_ent_cnt), dtype='bool')
        label_mask[:len(ents), :len(ents)] = 1
        for ent in range(len(ents)):
            label_mask[ent][ent] = 0
        for ent in range(len(ents)):
            if numpy.all(ent_mask[ent] == 0):
                label_mask[ent, :] = 0
                label_mask[:, ent] = 0

        # training triples with positive examples (entity pairs with labels)
        train_triple = {}

        if "labels" in example:
            for label in example['labels']:
                evidence = label['evidence']
                r = int(docred_rel2id[label['r']])

                # update training triples
                if (label['h'], label['t']) not in train_triple:
                    train_triple[(label['h'], label['t'])] = [
                        {'relation': r, 'evidence': evidence}]
                else:
                    train_triple[(label['h'], label['t'])].append(
                        {'relation': r, 'evidence': evidence})

        # hts, relations that ent pairs maker for ER loss, evidence sents for evety ent pairs
        relations, hts, sent_labels = [], [], []
        for i in range(len(entity_pos)):
            for j in range(len(entity_pos)):
                if i != j:
                    hts.append((i,j))

                    if (i,j) in train_triple:
                        relation = [0] * len(docred_rel2id)
                        sent_evi = [0] * len(sent_pos)

                        for mention in train_triple[i, j]:  # for each relation mention with head h and tail t
                            relation[mention["relation"]] = 1
                            for k in mention["evidence"]:
                                sent_evi[k] += 1

                        relations.append(relation)
                        sent_labels.append(sent_evi)
                        pos_samples += 1

                    else:
                        relation = [1] + [0] * (len(docred_rel2id) - 1)
                        sent_evi = [0] * len(sent_pos)
                        relations.append(relation)

                        sent_labels.append(sent_evi)
                        neg_samples += 1


        # ent_mask = norm_mask(ent_mask)
        ent_men_mask = norm_mask(ent_men_mask)


        feature = {
             "input_ids": input_ids,
             "ent_men_mask": ent_men_mask,
             "label": label_ids,
             "relations": relations,
             'sent_labels': sent_labels,
             'sent_pos': sent_pos,
             "label_mask": label_mask,
             "entity_pos": entity_pos,
             "title": example['title'],
             "hts": hts
        }
        features.append(feature)

    print("# of positive examples {}.".format(pos_samples))
    print("# of negative examples {}.".format(neg_samples))

    return features


class DocREDProcessor(object):
    """Processor for the DocRED data set."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return DocREDExample(
            tensor_dict["guid"].numpy(),
            tensor_dict["title"].numpy(),
            tensor_dict["vertexSet"].numpy(),
            tensor_dict["sents"].numpy(),
            tensor_dict["labels"].numpy(),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        with open(os.path.join(data_dir, "train_annotated.json"), 'r') as f:
            examples = json.load(f)
        return self._create_examples(examples, 'train')

    def get_distant_examples(self, data_dir):
        """See base class."""
        with open(os.path.join(data_dir, "train_distant.json"), 'r') as f:
            examples = json.load(f)
        return self._create_examples(examples, 'train')

    def get_dev_examples(self, data_dir):
        """See base class."""
        with open(os.path.join(data_dir, "dev.json"), 'r') as f:
            examples = json.load(f)
        return self._create_examples(examples, 'dev')

    def get_test_examples(self, data_dir):
        """See base class."""
        with open(os.path.join(data_dir, "test.json"), 'r') as f:
            examples = json.load(f)
        return self._create_examples(examples, 'test')

    def get_label_map(self, data_dir):
        """See base class."""
        with open(os.path.join(data_dir, "rel2id.json"), 'r') as f:
            label_map = json.load(f)
        return label_map

    def _create_examples(self, instances, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, ins) in enumerate(instances):
            guid = "%s-%s" % (set_type, i)
            examples.append(DocREDExample(guid=guid,
                                          title=ins['title'],
                                          vertexSet=ins['vertexSet'],
                                          sents=ins['sents'],
                                          labels=ins['labels'] if set_type!="test" else None))
        return examples

@dataclass(frozen=False)
class DocREDExample:

    guid: str
    title: str
    vertexSet: list
    sents: list
    labels: Optional[list] = None

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self), indent=2, sort_keys=True) + "\n"


class DocREDInputFeatures(object):
    """
    A single set of features of data.

    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
        token_type_ids: Segment token indices to indicate first and second portions of the inputs.
        label: Label corresponding to the input
    """

    def __init__(self, input_ids, attention_mask, token_type_ids, ent_men_mask, cls_mask, ent_ner, ent_pos, ent_distance, structure_mask, label=None, label_mask=None):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.ent_men_mask = ent_men_mask
        self.cls_mask = cls_mask
        self.ent_ner = ent_ner
        self.ent_pos = ent_pos
        self.ent_distance = ent_distance
        self.structure_mask = structure_mask
        self.label = label
        self.label_mask = label_mask

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"