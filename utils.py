import spacy
import os
import json
from tqdm import tqdm
from copy import deepcopy
"""
1.遍历doc生成的每个共指消解cluster，遍历每个实体提及集合搜索生成的共指提及cluster第一个提及；
2.如果名字相同，确定实体下标，实体类型。遍历该cluster每个共指提及的位置，如果该提及不与训练集实体提及冲突，添加该生成的共指提及信息
"""
import difflib

import argparse
import glob
import json
import logging
import os
import random

import numpy as np
from copy import deepcopy
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from evaluation import to_official, official_evaluate, get_prediction

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoTokenizer,
    AutoModel,
    get_constant_schedule_with_warmup,
    get_linear_schedule_with_warmup
)

from dataset import docred_convert_examples_to_features as convert_examples_to_features
from dataset import DocREDProcessor

from new_model import BertForDocRED

logger = logging.getLogger(__name__)
from torch.cuda.amp import autocast as autocast, GradScaler

def string_similar(s1, s2):
    return difflib.SequenceMatcher(None, s1, s2).quick_ratio()


def coref_re(text_dic):
    # 字符拼接
    text_dic = deepcopy(text_dic)
    sent = ''
    for i, s in enumerate(text_dic['sents']):
        for c in s:
            if sent == '':
                sent = c
            else:
                sent = sent + ' ' + c

    nlp = spacy.load('en_core_web_sm')
    # Add neural coref to SpaCy's pipe
    import neuralcoref
    neuralcoref.add_to_pipe(nlp, greedyness=0.5)
    doc = nlp(sent)

    coref_cluser = doc._.coref_clusters
    # for i in coref_cluser:
    #     print(i.mentions)
    doc_men_name = []
    doc_men_idx = [] # [[{s_id: (start,end)},],]
    for i_ent, ent in enumerate(text_dic['vertexSet']):
        doc_men_name.append([])
        doc_men_idx.append([])
        for i_men, men in enumerate(ent):
            doc_men_name[-1].append(men['name'])
            doc_men_idx[-1].append({'sent_id':men['sent_id'],'pos':men['pos']})
    # 遍历doc生成的每个共指消解cluster，遍历每个实体提及集合搜索生成的共指提及cluster第一个提及；
    for i_clu,cluster in enumerate(coref_cluser):
        label = True # cluster进行处理的标志位
        men_list = cluster.mentions
        main_men = men_list[0] # 第一个提及的名字
        for i_ent,ent in enumerate(doc_men_name):
            for i_men,men in enumerate(ent):
                if float(string_similar(str(main_men),men))>= 0.9 and label:
                    label = False
                    last_ent = i_ent
                    ent_type = text_dic['vertexSet'][last_ent][0]['type']
                    # 如果名字相同，确定实体下标，实体类型。遍历该cluster每个共指提及的位置，如果该提及不与训练集实体提及冲突，添加该生成的共指提及信息
                    idx_list = []
                    """
                    确定span起始下标，在截取sents中查找目标提及的下标，（直接在sents中截取，由列表去确定提及部分名称的位置），
                    由此确定提及位置
                    """
                    for i, men in enumerate(men_list):
                        temp_sent = []
                        # print(men.start)
                        for i_s, s in enumerate(text_dic['sents']):
                            if len(temp_sent) <= men.start < len(temp_sent)+len(s):
                                temp_sent+= s[:men.start-len(temp_sent)+1]
                                sent_i = i_s
                                break
                            temp_sent.extend(s)
                        # 如果候选提及之前存在特殊字符，且这之后临近存在首单词，如何？
                        # print(temp_sent)
                        # 检查生成的提及是否在同一个句子内
                        if str(men) in ' '.join(text_dic['sents'][sent_i]) or str(men) in ' '.join(
                                text_dic['sents'][sent_i - 1]):

                            # 如果候选提及之前存在特殊字符，且这之后临近存在首单词，如何？
                            men_name_list = str(men).split()
                            gol_str = men_name_list[0]
                            # print(gol_str)
                            if gol_str in temp_sent:

                                temp_sent_str = ''
                                for c in temp_sent:
                                    if temp_sent_str == '':
                                        temp_sent_str = c
                                    else:
                                        temp_sent_str = temp_sent_str + ' ' + c

                                num = temp_sent_str.encode('unicode_escape').decode('utf-8').count(
                                    '\\u2018') + temp_sent_str.encode('unicode_escape').decode('utf-8').count(
                                    '\\u2019')
                                if str(men) in ' '.join(temp_sent[-num:]) and num != 0:  # 针对提及长度很短，且有较大的特殊字符提前量
                                    # print(' '.join(temp_sent[-num:]))
                                    # print(num)
                                    words = temp_sent[-num:]
                                    short_words = str(men).split()
                                    positions = []
                                    if len(short_words) <= len(words):
                                        for i in range(len(words) - len(short_words) + 1):
                                            if words[i:i + len(short_words)] == short_words:
                                                positions.append(i)
                                    # print(positions)
                                    if len(positions) == 0:
                                        try:
                                            positions.append(words.index(gol_str))
                                        except ValueError:  # of Africa’
                                            print('doc {} span {} has err'.format(text_dic['title'], str(men)))
                                            continue
                                    final_idx = len(temp_sent) - (len(words) - positions[0])
                                else:
                                    candi_list = [idx for idx, str in enumerate(temp_sent) if str == gol_str]
                                    final_idx = candi_list[-1]
                                idx_list.append((men, final_idx))
                            else:
                                continue


                    # 根据每句话的长度，得出具体下标

                    men_idx = []  # [{s_id: (start, end)},]
                    sent_len = [len(s) for s in text_dic['sents']]
                    # print(sent_len)
                    for men, idx_men in idx_list:
                        start_id = 0
                        for idx_s, len_s in enumerate(sent_len):
                            if start_id <= idx_men < start_id + len_s:
                                start_id_men = idx_men - start_id
                                men_idx.append({'sent_id':idx_s, 'pos':[start_id_men, start_id_men + len(str(men).split())]})
                                break
                            start_id += len_s

                    for i_coref, idx_coref in enumerate(men_idx):
                        add_label =True
                        id_list = [idx_coref['pos'][0]+i for i in range(idx_coref['pos'][1]-idx_coref['pos'][0]+1)]
                        for gold_men in doc_men_idx[last_ent]:
                            if idx_coref['sent_id'] == gold_men['sent_id']:
                                id_list_gold = [gold_men['pos'][0] + i for i in range(gold_men['pos'][1] - gold_men['pos'][0] + 1)]
                                if len(set(id_list).intersection(set(id_list_gold))) !=0:
                                    add_label =False
                        if add_label:
                            name = idx_list[i_coref][0]
                            sent_id = idx_coref['sent_id']
                            pos = idx_coref['pos']
                            text_dic['vertexSet'][last_ent].append({'name':str(name), 'sent_id':int(sent_id),'pos':pos,'type':str(ent_type)})
    return text_dic


def check_max_men_len(dir):
    """

    :param dir:更新后的数据集地址
    :return: 训练集中所有文档最大提及长度,最大实体数
    """
    with open(dir, 'r') as fh:
        data = json.load(fh)
    idx_list_more = []
    doc_ent_num = []

    for i_doc, new_doc in enumerate(tqdm(data, desc="itration")):

        doc_ent_num.append(len(new_doc['vertexSet']))
        men_len_list = []
        for ent in new_doc['vertexSet']:
            men_len_list.append(len(ent))
        max_ent_len = max(men_len_list)

        idx_list_more.append((i_doc, max_ent_len,))
    max_ent_len = max([num for i, num in idx_list_more])
    max_ent_num = max(doc_ent_num)
    return max_ent_len, max_ent_num

def check_ave_men_num(dir):
    with open(dir,'r') as f:
        data = json.load(f)
    e_num , m_num = 0, 0
    for ex_i ,example in enumerate(data):
        e_num += len(example['vertexSet'])
        for e_i , e in enumerate(example['vertexSet']):
            m_num += len(e)
    return e_num , m_num, round(m_num/e_num, 2)

def check_max_sent(dataset):

    sent_num_list = []
    sent_idx_list = []
    for data in dataset:
        curr_sent_num = len(data['sent_pos'])
        sent_num_list.append(curr_sent_num)
        sent_idx_list.append(data['sent_pos'][-1][1])
    return max(sent_num_list), max(sent_idx_list)

def load_and_cache_examples(args, tokenizer, evaluate=False, predict=False):

    processor = DocREDProcessor()
    # Load data
    logger.info("Creating features from dataset file at %s", args.data_dir)
    label_map = processor.get_label_map(args.data_dir)

    if evaluate:

        dev_path = './data/dwie/dev.json'
        if os.path.exists(dev_path):
            with open(dev_path,'r') as f:
                examples = json.load(f)
        else:
            print("loading data with errors..")
        features = convert_examples_to_features(args,
                                                examples,
                                                args.model_type,
                                                tokenizer,
                                                max_length=args.max_seq_length,
                                                max_ent_cnt=args.max_ent_cnt,
                                                label_map=label_map
                                                )

    elif predict:

        test_path = './data/dwie/test.json'
        if os.path.exists(test_path):
            with open(test_path,'r') as f:
                examples = json.load(f)
        else:
            print("loading data with errors..")
        features = convert_examples_to_features(args,
                                                examples,
                                                args.model_type,
                                                tokenizer,
                                                max_length=args.max_seq_length,
                                                max_ent_cnt=args.max_ent_cnt,
                                                label_map=label_map
                                                )
    else:

        train_path = './data/dwie/train_annotated.json'
        if os.path.exists(train_path):
            with open(train_path,'r') as f:
                examples = json.load(f)
        else:
            print("loading data with errors..")
        features = convert_examples_to_features(args,
                                                examples,
                                                args.model_type,
                                                tokenizer,
                                                max_length=args.max_seq_length,
                                                max_ent_cnt=args.max_ent_cnt,
                                                label_map=label_map
                                                )

    return features


