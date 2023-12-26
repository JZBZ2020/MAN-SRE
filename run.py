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

from model import BertForDocRED
from utils import coref_re
logger = logging.getLogger(__name__)
from torch.cuda.amp import autocast as autocast, GradScaler

def iteration_coref(dataset):
    new_dataset = []
    for ex_index, ex in enumerate(tqdm(dataset,desc='coreference resolution')):
        new_ex = coref_re(ex)
        new_dataset.append(new_ex)
    return new_dataset

def process_pred(pred, label_mask):
    pred_array = pred.detach().cpu().numpy()
    mask = label_mask.detach().cpu().numpy()

    pred_out = []

    for i in range(len(pred_array)):
        # tmp_pred_out = []
        for j in range(len(pred_array[i])):
            for k in range(len(pred_array[i])):
                if mask[i][j][k] == 1:
                    pred_out.append(pred_array[i][j][k][:])
    return pred_out

def collate_fn(batch):
    max_sent = max([len(f["sent_pos"]) for f in batch])
    max_len = max([len(f["input_ids"]) for f in batch])
    input_ids = [f["input_ids"] + [0] * (max_len - len(f["input_ids"])) for f in batch] # 补0
    input_mask = [[1.0] * len(f["input_ids"]) + [0.0] * (max_len - len(f["input_ids"])) for f in batch]
    labels = torch.tensor([f["label"] for f in batch])

    relations = [f['relations'] for f in batch]
    relations = [torch.tensor(relation) for relation in relations]
    relations = torch.cat(relations, dim=0)
    sent_pos =  [f["sent_pos"] for f in batch]

    entity_pos = [f["entity_pos"] for f in batch]
    ent_men_mask = torch.tensor([f['ent_men_mask'] for f in batch], dtype=torch.float)
    hts = [f["hts"] for f in batch]
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    input_mask = torch.tensor(input_mask, dtype=torch.float)
    label_mask = torch.tensor([f['label_mask'] for f in batch],dtype=torch.bool)

    sent_labels = [f["sent_labels"] for f in batch if "sent_labels" in f]
    if sent_labels != [] and None not in sent_labels:
        sent_labels_tensor = []
        for sent_label in sent_labels:
            sent_label = np.array(sent_label)
            sent_labels_tensor.append(np.pad(sent_label, ((0, 0), (0, max_sent - sent_label.shape[1]))))
        sent_labels_tensor = torch.from_numpy(np.concatenate(sent_labels_tensor, axis=0))
    else:
        sent_labels_tensor = None

    if "negative_mask" in batch[0]:
        neg_masks = torch.stack([f['negative_mask'] for f in batch], dim=0)
    else: neg_masks = None

    output = (input_ids, input_mask, labels, relations, entity_pos, sent_pos, ent_men_mask, hts,sent_labels_tensor,neg_masks,label_mask)

    return output

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def get_random_mask(train_features, drop_prob):
    new_features = []
    n_e = 42
    for old_feature in tqdm(train_features,desc='getting negative masks..'):
        feature = deepcopy(old_feature)
        ent_n = len(feature['entity_pos'])
        neg_labels = torch.tensor(feature['relations'])[:, 0]
        neg_index = torch.where(neg_labels == 1)[0]
        pos_index = torch.where(neg_labels == 0)[0]
        perm = torch.randperm(neg_index.size(0))  # 负例个数的随机序列，结果用于索引列表值
        sampled_negative_index = neg_index[perm[:int(drop_prob * len(neg_index))]]
        sampled_negative_hts = torch.tensor(feature['hts'])[sampled_negative_index]
        neg_mask = torch.ones((ent_n,ent_n))
        neg_mask[sampled_negative_hts[:,0],sampled_negative_hts[:,1]] = 0
        pad_neg = torch.zeros((n_e,n_e))
        pad_neg[:ent_n, :ent_n] = neg_mask
        feature['negative_mask'] = pad_neg # [42,42] 真正实体对没有被采样到的负例的id被赋值为1，其他为0
        new_features.append(feature)
    return new_features


def train(args, train_dataset, dev_dataset, test_dataset, model, tokenizer):
    def Set_optimizer_for_tow_step():
        if args.fixed_encoder:
            args.encoder_lr = 0.0
        new_layer = ["extractor", "bilinear", "classifier","projector"]
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        component = ['encoder']
        grouped_params = [
            {
                'params': [p for n, p in model.named_parameters() if any(nd in n for nd in component)], # 在encoder
                'lr': args.encoder_lr
            },
            {
                'params': [p for n, p in model.named_parameters() if any(nd in n for nd in new_layer)], # 在new_layer
                'lr': args.learning_rate
            },
        ]

        optimizer = AdamW(grouped_params, lr=args.encoder_lr, eps=args.adam_epsilon)
        if args.fixed_encoder:
            optimizer = AdamW(grouped_params, lr=3e-5, eps=args.adam_epsilon)

        return optimizer

    def Set_optimizer_for_one_step():
        new_layer = ["extractor", "bilinear"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in new_layer)], },
            {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in new_layer)],
             "lr": args.learning_rate},
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=args.encoder_lr, eps=args.adam_epsilon)
        return optimizer

    def finetune(train_dataset, optimizer, num_epoch, num_steps,epoch_start = 0):
        best_f1 = 0
        best_results = None
        args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
        if not args.base_train:
            train_dataset = get_random_mask(train_dataset, args.drop_prob)
        train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn,drop_last=True)
        t_total = int(len(train_dataloader) * num_epoch // args.gradient_accumulation_steps)
        args.warmup_steps = int(t_total * args.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
        train_iterator = range(epoch_start,int(num_epoch))
        print("Total steps: {}".format(t_total))
        print("Warmup steps: {}".format(args.warmup_steps))
        scaler = GradScaler()
        for epoch in tqdm(train_iterator,desc="Epoch"):
            epoch_iterator = tqdm(train_dataloader, desc="Iteration" )
            for step, batch in enumerate(epoch_iterator):
                model.train()
                model.zero_grad()
                optimizer.zero_grad()
                inputs = {
                    "input_ids": batch[0].to(0),
                    "input_mask": batch[1].to(0),
                    'label': batch[2].to(0),
                    "entity_pos": batch[4],
                    "ent_men_mask": batch[6],
                    'hts': batch[7],
                    'sent_labels': batch[8].to(0),
                    'sent_pos': batch[5],
                    'relations': batch[3].to(0),
                    'negative_mask': batch[-2].to(0) if not args.base_train else batch[-2],
                    'label_mask': batch[-1]
                }

                outputs = model(**inputs)
                loss = outputs["loss"]["rel_loss"]

                if inputs["sent_labels"] != None and args.base_train:
                    loss += outputs["loss"]["evi_loss"] * args.evi_lambda
                if args.n_gpu > 1:
                    loss = loss.mean()
                loss = loss / args.gradient_accumulation_steps


                scaler.scale(loss).backward()
                # scaler.unscale_(optimizer)

                if step % args.gradient_accumulation_steps == 0:
                    if args.max_grad_norm > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    model.zero_grad()
                    num_steps += 1

                    if (step + 1) == len(train_dataloader) or (args.evaluation_steps > 0 and num_steps % args.evaluation_steps == 0 and step % args.gradient_accumulation_steps == 0):
                        scores, result = evaluate(args, dev_dataset, model, tokenizer)
                        learning_rate_scalar = scheduler.get_last_lr()[0]
                        logger.info({'learning_rate': learning_rate_scalar, 'step':num_steps})
                        if scores['f1'] > best_f1:
                            best_f1 = scores['f1']
                            best_scores = scores
                            best_results = result

                            logger.info("Saving best model checkpoint to %s", args.output_dir)

                            ckpt_file = os.path.join(args.output_dir, "best.ckpt")
                            print(f"saving model checkpoint into {ckpt_file} ...")
                            torch.save(model.state_dict(),ckpt_file)
                        if epoch == train_iterator[-1]:
                            print("best scores : {}".format(best_scores))
                            scores_dir = os.path.join(args.output_dir,'scores.json')
                            json.dump(best_scores,open(scores_dir,'w'))
                            result_dir = os.path.join(args.output_dir,'result.json')
                            with open(result_dir,'w') as fh :
                                json.dump(best_results, fh)

        return num_steps

    if (args.base_train):
        optimizer = Set_optimizer_for_one_step()
    else:
        optimizer = Set_optimizer_for_tow_step()
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    num_steps = 0
    set_seed(args)
    model.zero_grad()

    finetune(train_dataset, optimizer, args.num_train_epochs, num_steps, epoch_start=0)



def evaluate(args, eval_dataset, model, tokenizer, prefix=""):

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate_fn,drop_last=False)
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)
    eval_loss = 0.0
    nb_eval_steps = 0
    scores, topks = [], []
    preds,evi_preds = [], []
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        # batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {
                      "input_ids": batch[0].to(0),
                      "input_mask": batch[1].to(0),
                      'label': batch[2].to(0),
                      "entity_pos": batch[4],
                      "ent_men_mask": batch[6],
                      'hts': batch[7],
                      'sent_labels': batch[8].to(0),
                      'sent_pos': batch[5],
                      'relations': batch[3].to(0),
                      'negative_mask': batch[-2],
                      'label_mask': batch[-1]
                      }


            outputs = model(**inputs)
            if "scores" in outputs:
                scores.append(outputs["scores"].cpu().numpy())
                topks.append(outputs["topks"].cpu().numpy())
            logits = outputs['rel_pred']

            tmp_eval_loss = outputs['loss']['rel_loss']
            eval_loss += tmp_eval_loss.mean().item()
            if "evi_pred" in outputs: # relation extraction and evidence extraction
                evi_pred = outputs["evi_pred"]
                evi_pred = evi_pred.cpu().numpy()
                evi_preds.append(evi_pred)
            nb_eval_steps += 1
            if not args.base_train:
                pred = process_pred(logits,batch[-1])
            else:
                pred = logits.detach().cpu().numpy()
            preds.append(pred)

    preds = np.concatenate(preds, axis=0).astype(np.float32)
    if evi_preds != []:
        evi_preds = np.concatenate(evi_preds, axis=0)
    eval_loss = eval_loss / nb_eval_steps
    if scores != []: # 利用初步模型预测的实体对较大关系以及分数，用于之后的推导融合
        scores = np.concatenate(scores, axis=0)
        topks =  np.concatenate(topks, axis=0)
    official_results = to_official(preds, eval_dataset, evi_preds=evi_preds, scores=scores, topks=topks)
    if len(official_results) > 0:
        best_f1, evi_f1, best_f1_ign, _ = official_evaluate(official_results, args.data_dir)
    else:
        best_f1, best_f1_ign, evi_f1 = 0.0, 0.0, 0.0
    output = {
        "f1": best_f1 * 100,
        "ign_f1": best_f1_ign * 100,
        'evi_f1': evi_f1 * 100,
        "loss": eval_loss
    }
    del preds
    del evi_preds
    print(output)

    return output, official_results

def predict(args, eval_dataset, model, tokenizer, prefix=""):

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate_fn,drop_last=False)
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = []
    evi_preds = []
    scores, topks = [], []
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        # batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {
                      "input_ids": batch[0].to(0),
                      "input_mask": batch[1].to(0),
                      'label': batch[2].to(0),
                      "entity_pos": batch[4],
                      "ent_men_mask": batch[6],
                      'hts': batch[7],
                      'sent_labels': batch[8].to(0),
                      'sent_pos': batch[5],
                      'relations': batch[3].to(0),
                      'negative_mask': batch[-2],
                      'label_mask': batch[-1]
                      }
            outputs = model(**inputs)
            # tmp_eval_loss, logits = outputs[:2]
            # tmp_eval_loss = outputs['loss']['rel_loss']
            logits = outputs['rel_pred']

            # eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            # pred = process_pred(logits,batch[-1])
            pred = logits.detach().cpu().numpy()
            preds.append(pred)
            if "evi_pred" in outputs: # relation extraction and evidence extraction
                evi_pred = outputs["evi_pred"]
                evi_pred = evi_pred.cpu().numpy()
                evi_preds.append(evi_pred)
    preds = np.concatenate(preds, axis=0).astype(np.float32)
    if evi_preds != []:
        evi_preds = np.concatenate(evi_preds, axis=0)
    out_prediction = get_prediction(preds,eval_dataset,evi_preds,scores,topks)

    # eval_loss = eval_loss / nb_eval_steps
    # print("eval_loss: {}".format(eval_loss))
    # if not os.path.exists('./data/DocRED/'):
    #     os.makedirs('./data/DocRED')
    output_eval_file = os.path.join(args.output_dir, "result.json")
    with open(output_eval_file, 'w') as f:
        json.dump(out_prediction, f)


def load_and_cache_examples(args, tokenizer, evaluate=False, predict=False):

    processor = DocREDProcessor()
    # Load data
    logger.info("Creating features from dataset file at %s", args.data_dir)
    label_map = processor.get_label_map(args.data_dir)

    if evaluate:

        dev_path = './data/DocRED/dev.json'
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

        test_path = './data/DocRED/test.json'
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

        train_path = './data/DocRED/train_annotated.json'
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max_ent_cnt",
        default=42,
        type=int,
        help="The maximum entities considered.",
    )

    parser.add_argument("--entity_structure", default='biaffine', type=str, choices=['none', 'decomp', 'biaffine'],
                        help="whether and how do we incorporate entity structure in Transformer models.")
    parser.add_argument("--data_dir", default='./data/DocRED/', type=str, help="The input data dir. Should contain the .tsv files (or other data files) for the task.",)
    parser.add_argument(
        "--transformer_type",
        default='roberta',
        type=str,
        help="Model type",
    )
    parser.add_argument("--model_type", default="roberta", type=str)
    parser.add_argument(
        "--model_name_or_path",
        default='roberta-large',
        type=str,
        help="Path to pre-trained model or shortcut name",
    )
    parser.add_argument("--base_train", action='store_true', default=True, help="whether to do base train (evi extractor)") # 参数action与type不能共存


    parser.add_argument(
        "--output_dir",
        default='',
        type=str,
        required=False,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        default='checkpoints',
        type=str,
        required=False,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument('--load_pretrained',default='', type=str)
    parser.add_argument("--load_path", default="", type=str)

    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument("--processed_path", default="./data/processed_data/",type=str,help="save the processed data")

    parser.add_argument(
        "--max_seq_length",
        default=1024,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument("--do_train", default=False, type=bool, help="Whether to run training.")
    parser.add_argument("--do_eval", default=False, type=bool, help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict", action="store_true", default=True, help="Whether to run pred on the pred set.")
    parser.add_argument("--predict_thresh", default=0.5, type=float, help="pred thresh")
    parser.add_argument(
        "--do_lower_case", type=bool, help="Set this flag if you are using an uncased model.",
    )
    parser.add_argument(
        "--per_gpu_train_batch_size", default=4, type=int, help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=4, type=int, help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--evaluation_steps", default=-1, type=int,
                        help="Number of training steps between evaluations.")
    parser.add_argument("--num_labels",default=97,type=int)
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--logging_steps", type=int, default=1000, help="Log every X updates steps.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--warmup_ratio", default=0.06, type=float, help="Linear warmup ratio, overwriting warmup_steps.")

    parser.add_argument("--fixed_encoder", type=bool, default=True)
    parser.add_argument("--learning_rate", default=1e-4, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--encoder_lr", default=3e-5, type=float, help="The initial learning rate for encoder.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")


    parser.add_argument(
        "--num_train_epochs", default=60, type=float, help="Total number of training epochs to perform.",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")


    parser.add_argument('--weight_threshold', type=float, default=0.0, help='overlook those weight values that are smaller than threshold')
    parser.add_argument("--evi_lambda", default=0.1, type=float,
                        help="Weight of relation-agnostic evidence loss during training. ")
    parser.add_argument("--drop_prob", default=0.0, type=float,
                        help="Negative Sample Discard rate.")
    args = parser.parse_args()


    n_gpu = 0
    if torch.cuda.is_available():
        print("=====================GPU========================")
        device = torch.device("cuda")
        n_gpu = torch.cuda.device_count()
    else:
        print("=====================CPU========================")
        device = torch.device("cpu")
    print("device:", device, "\t", "n_gpu:", n_gpu)

    args.n_gpu = n_gpu
    args.device = device

    if args.output_dir != '':
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=args.num_labels,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
    )

    model = AutoModel.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
    )
    #
    config.cls_token_id = tokenizer.cls_token_id
    config.sep_token_id = tokenizer.sep_token_id
    config.transformer_type = args.transformer_type
    # Training
    set_seed(args)
    model = BertForDocRED(args, config, model, num_labels=args.num_labels, max_ent_cnt=args.max_ent_cnt,
                          weight_threshold=args.weight_threshold)
    model.to(args.device)

    train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False)
    dev_dataset = load_and_cache_examples(args, tokenizer, evaluate=True)
    test_dataset = load_and_cache_examples(args, tokenizer, predict=True)
    # 完成训练，开始预测
    if not args.base_train:
        print("loading model checkpoints from {} and begin training model".format(args.load_path))
        model.load_state_dict(torch.load(args.load_path), strict=False)
        train(args, train_dataset, dev_dataset, test_dataset, model, tokenizer)

    elif not args.do_train and args.do_predict:
        print("loading model checkpoints from {} and begin training model".format(args.load_path))
        model.load_state_dict(torch.load(args.load_path), strict=False)
        predict(args, test_dataset, model, tokenizer)
    else:
        train(args, train_dataset, dev_dataset, test_dataset, model, tokenizer)


if __name__ == "__main__":
    main()
