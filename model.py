import torch
import torch.nn as nn
from opt_einsum import contract
import numpy as np
import torch.nn.functional as F
from long_seq import process_long_input
from axial_attention import AxialAttention, AxialImageTransformer
from loss import ATLoss,ATLoss_1
import math

class AxialTransformer_by_entity(nn.Module):
    def __init__(self, emb_size=768, dropout=0.1, num_layers=2, dim_index=-1, heads=8, num_dimensions=2, ):
        super().__init__()
        self.num_layers = num_layers  # 6
        self.dim_index = dim_index
        self.heads = heads
        self.emb_size = emb_size
        self.dropout = dropout
        self.num_dimensions = num_dimensions
        self.axial_attns = nn.ModuleList(
            [AxialAttention(dim=self.emb_size, dim_index=dim_index, heads=heads, num_dimensions=num_dimensions, ) for i
             in range(num_layers)])
        self.ffns = nn.ModuleList([nn.Linear(self.emb_size, self.emb_size) for i in range(num_layers)])
        self.lns = nn.ModuleList([nn.LayerNorm(self.emb_size) for i in range(num_layers)])
        self.attn_dropouts = nn.ModuleList([nn.Dropout(dropout) for i in range(num_layers)])
        self.ffn_dropouts = nn.ModuleList([nn.Dropout(dropout) for i in range(num_layers)])

    def forward(self, x):
        for idx in range(self.num_layers):
            x = x + self.attn_dropouts[idx](self.axial_attns[idx](x))
            x = self.ffns[idx](x)
            x = self.ffn_dropouts[idx](x)
            x = self.lns[idx](x)
        return x


class AxialEntityTransformer(nn.Module):
    def __init__(self, emb_size=768, dropout=0.1, num_layers=2, dim_index=-1, heads=8, num_dimensions=2, ):
        super().__init__()
        self.num_layers = num_layers
        self.dim_index = dim_index
        self.heads = heads
        self.emb_size = emb_size
        self.dropout = dropout
        self.num_dimensions = num_dimensions
        self.axial_img_transformer = AxialImageTransformer()
        self.axial_attns = nn.ModuleList(
            [AxialAttention(dim=self.emb_size, dim_index=dim_index, heads=heads, num_dimensions=num_dimensions, ) for i
             in range(num_layers)])
        self.ffns = nn.ModuleList([nn.Linear(self.emb_size, self.emb_size) for i in range(num_layers)])
        self.lns = nn.ModuleList([nn.LayerNorm(self.emb_size) for i in range(num_layers)])
        self.attn_dropouts = nn.ModuleList([nn.Dropout(dropout) for i in range(num_layers)])
        self.ffn_dropouts = nn.ModuleList([nn.Dropout(dropout) for i in range(num_layers)])

    def forward(self, x):
        for idx in range(self.num_layers):
            x = x + self.attn_dropouts[idx](self.axial_attns[idx](x))
            x = self.ffns[idx](x)
            x = self.ffn_dropouts[idx](x)
            x = self.lns[idx](x)
        return x

class simpleAttention(nn.Module):
    """
    q: [b,max_ent, reduce_dim]
    kv: [b, max_ent, max_men, reduce_dim]
    ment_mask: [b, max_ent, max_men,1]
    """
    def __init__(self,k_dim, weight_threshold):
        super().__init__()
        self.k_dim = k_dim
        self.reduce_dim = k_dim
        self.threshold = weight_threshold
        self.to_q_projector = nn.Linear(self.reduce_dim, k_dim, bias=False)
        self.to_k_projector = nn.Linear(self.reduce_dim, k_dim, bias = False)
        self.to_v_projector = nn.Linear(self.reduce_dim, self.reduce_dim, bias=False)


    def forward(self,q,kv, ment_mask):
        q = self.to_q_projector(q) # [b,max_ent,k_dim]
        k = self.to_k_projector(kv)# [b,max_ent,max_men,k_dim]
        v = self.to_v_projector(kv) # [b,max_ent,max_men,k_dim]

        assert q.shape[-1] == k.shape[-1]
        assert k.shape[-2] == v.shape[-2]
        # [b,(max_ent),max_ent,k_dim] * [b,max_ent,k_dim,max_men] -> [b,(max_ent),max_ent,max_men]
        # dots = torch.matmul(q.unsqueeze(1).repeat(1,q.shape[1],1,1), k.transpose(2,3).contiguous() ) * (self.k_dim ** -0.5)

        # 权重计算方法2
        dots = torch.matmul(k, (q.unsqueeze(2).repeat(1,1,q.shape[1],1)).permute(0,1,3,2).contiguous()) * (self.k_dim ** -0.5) # [b, max_ent, max_men, max_ent]
        dots = dots + ment_mask

        # dots = dots.permute(0,1,3,2).contiguous() + ment_mask # [b, max_ent, max_men, max_ent] ment_mask是否合理
        dots = nn.Softmax(dim=-2)(dots).transpose(2,3).contiguous() # [b, max_ent, max_ent, max_men]
        # 通过阈值判断是否忽略权重小的提及,即时实体提及表示为0，但影响k,v
        mask = dots >= self.threshold # [b, max_e, max_e, max_m]
        dots = dots * mask # 小于阈值权重的提及被忽略，其他保留权重
        # 小于阈值的提及被忽略，其他的提及拥有相同的权重
        # simple_dots = torch.zeros(dots.shape).to(dots)
        # simple_dots[mask] = 1.0
        # dots = simple_dots / torch.sum(simple_dots, dim = -1, keepdim= True)


        value = torch.matmul(dots,v) # [b, max_ent,max_ent,reduce_dim]
        output = (value,dots)
        return output

class multi_head_Attention(nn.Module):
    """
    利用多头注意力计算指定实体的实体表示，无法得出实体对应实体的每个提及的权重
    这里不采用men_mask，因为不存在的提及表示已经是0向量，计算结果权重自然为0
    """
    def __init__(self, dim, heads, dim_heads = None):
        super().__init__()
        self.dim_heads = (dim // heads) if dim_heads is None else dim_heads
        dim_hidden = self.dim_heads * heads

        self.heads = heads
        self.to_q = nn.Linear(dim, dim_hidden, bias = False)
        self.to_kv = nn.Linear(dim, 2 * dim_hidden, bias = False)
        self.to_out = nn.Linear(dim_hidden, dim)

    def forward(self, x, kv = None,):
        (b,ne,hidden) = x.size()
        (b,ne,nm,hidden) = kv.size()
        kv = x if kv is None else kv
        q, k, v = (self.to_q(x), *self.to_kv(kv).chunk(2, dim=-1))

        b, t, d, h, e = *q.shape, self.heads, self.dim_heads # bs, 42, 768, 8 ,96
        q = q.reshape(b,-1,h,e).transpose(1,2).reshape(b*h,-1,e)
        merge_heads = lambda x: x.reshape(b, x.shape[1],x.shape[2], h, e).permute(0,3,1,2,4).reshape(b * h, ne,nm, e)
        k, v = map(merge_heads, (k, v))
        dots = torch.matmul(q.unsqueeze(1).repeat(1,q.shape[1],1,1), k.transpose(2,3).contiguous() ) * (e ** -0.5)
        dots = dots.permute(0,1,3,2).contiguous()  # [b*h, max_ent, max_men, max_ent]
        dots = nn.Softmax(dim=-2)(dots).transpose(2,3).contiguous() # [b*h, max_ent, max_ent, max_men]
        value = torch.matmul(dots,v)
        value = value.reshape(b,h,ne,ne,e).permute(0,2,3,1,4).reshape(b,ne,ne,d)
        value = self.to_out(value)
        # merge_heads = lambda x: x.reshape(b, -1, h, e).transpose(1, 2).reshape(b * h, -1, e) # q: [b*8, 42, 96] Kv:[b*8, 42*nm, 96]
        # q, k, v = map(merge_heads, (q, k, v)) # [bs*42*8, 42, 96]
        # 加权求和
        # dots = torch.einsum('bie,bje->bij', q, k) * (e ** -0.5) # 通过点积计算i对于j的权重
        # dots = dots.softmax(dim=-1)
        # out = torch.einsum('bij,bje->bie', dots, v)
        #
        # out = out.reshape(b, h, -1, e).transpose(1, 2).reshape(b, -1, d)
        # out = self.to_out(out)

        return value

class BertForDocRED(nn.Module):
    def __init__(self, args,config,model, num_labels, max_ent_cnt, with_naive_feature=False, entity_structure=False, weight_threshold = 0.0, max_sent_num = 25, evi_thresh = 0.2):
        super().__init__()
        self.args = args
        self.num_labels = num_labels
        self.max_ent_cnt = max_ent_cnt
        self.config = config
        self.ne = 42
        self.encoder = model
        self.max_sent_num = max_sent_num
        self.with_naive_feature = with_naive_feature
        self.reduced_dim = config.hidden_size
        self.relation_dim = 256
        self.evi_thresh = evi_thresh
        self.loss_fnt_evi = nn.KLDivLoss(reduction="batchmean")

        # self.bert = BertModel(config, with_naive_feature, entity_structure)
        self.dropout = nn.Dropout(0.1)
        self.dim_reduction = nn.Linear(config.hidden_size, self.reduced_dim)
        self.feature_size = self.reduced_dim

        self.hidden_size = config.hidden_size
        self.attention_weight = nn.Linear(self.feature_size, self.relation_dim)
        self.attention_net = nn.Parameter(torch.randn(self.num_labels, self.relation_dim))
        self.classifier = nn.Parameter(torch.randn(self.num_labels, self.reduced_dim*2, self.reduced_dim*2))
        self.classifier_bais = nn.Parameter(torch.randn(self.num_labels))
        nn.init.uniform_(self.classifier,a=-math.sqrt(1/(2*self.reduced_dim)), b=math.sqrt(1/(2*self.reduced_dim)))
        nn.init.uniform_(self.classifier_bais, a=-math.sqrt(1 / (2*self.reduced_dim)), b=math.sqrt(1 / (2*self.reduced_dim)))
        nn.init.xavier_normal_(self.attention_net)

        """
        设计共指消解模型为可训练
        设计阈值确定指定实体的提及分布，相当于模型聚焦于实体对的重要提及；
        采用atlop上下文注意力分布，抓取实体对的上下文信息；
        根据实体对相互提及在文档的分布，确定实体对的证据集，计算ER损失
        """
        # ent or men att rep
        self.ent_projector = nn.Linear(2* self.config.hidden_size, self.config.hidden_size, bias=False)
        self.men_projector = nn.Linear(2 * self.config.hidden_size, self.config.hidden_size, bias=False)

        self.simpleAttention = simpleAttention(self.reduced_dim, weight_threshold)
        self.rel_classifier_1 = nn.Linear(self.reduced_dim * 2 , self.num_labels)
        self.rel_classifier_2 = nn.Linear(self.reduced_dim, self.num_labels)

        # head or tail extractor for base train
        self.head_extractor_1 = nn.Linear(2 * config.hidden_size, config.hidden_size)
        self.tail_extractor_1 = nn.Linear(2 * config.hidden_size, config.hidden_size)

        # 分类之前，首尾实体表示重塑
        self.head_extractor_2 = nn.Linear(2 * config.hidden_size, config.hidden_size)
        self.tail_extractor_2 = nn.Linear(2 * config.hidden_size, config.hidden_size)

        # bilinear classifier
        self.emb_size = self.reduced_dim
        self.block_size = 64
        self.bilinear = nn.Linear(self.emb_size * self.block_size, self.num_labels)
        self.last_classifier = nn.Linear(config.hidden_size, config.num_labels)
        # 实体对表示映射
        self.projector = nn.Linear(self.emb_size * self.block_size, config.hidden_size, bias=False)

        # atloss
        self.max_num_labels = 4
        self.loss_fnt = ATLoss()
        self.loss_fnt_1 = ATLoss_1()

        self.axial_transformer = AxialTransformer_by_entity(emb_size=config.hidden_size, dropout=0.0, num_layers=6,
                                                            heads=8)



    def encode(self, input_ids, attention_mask):
        config = self.config
        if config.transformer_type == "bert":
            start_tokens = [config.cls_token_id] #[101]

            end_tokens = [config.sep_token_id]#[102]

        elif config.transformer_type == "roberta":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id, config.sep_token_id]
        sequence_output, attention = process_long_input(self.encoder, input_ids, attention_mask, start_tokens, end_tokens)
        return sequence_output, attention

    def get_hrt_evi(self, sequence_output, attention, entity_pos, hts, offset):

        '''
        Get head, tail, context embeddings from token embeddings.
        Inputs:
            :sequence_output: (batch_size, doc_len, hidden_dim)
            :attention: (batch_size, num_attn_heads, doc_len, doc_len)
            :entity_pos: list of list. Outer length = batch size, inner length = number of entities each batch.
            :hts: list of list. Outer length = batch size, inner length = number of combination of entity pairs each batch.
            :offset: 1 for bert and roberta. Offset caused by [CLS] token.
        Outputs:
            :hss: (num_ent_pairs_all_batches, emb_size)
            :tss: (num_ent_pairs_all_batches, emb_size)
            :rss: (num_ent_pairs_all_batches, emb_size)
            :ht_atts: (num_ent_pairs_all_batches, doc_len)
            :rels_per_batch: list of length = batch size. Each entry represents the number of entity pairs of the batch.
        '''

        n, h, _, c = attention.size()
        hss, tss, rss = [], [], []
        ht_atts = []

        for i in range(len(entity_pos)):  # for each batch
            entity_embs, entity_atts = [], []

            # obtain entity embedding from mention embeddings.
            for eid, e in enumerate(entity_pos[i]):  # for each entity
                if len(e) > 1:
                    e_emb, e_att = [], []
                    for mid, (start, end) in enumerate(e):  # for every mention
                        if start + offset < c:
                            # In case the entity mention is truncated due to limited max seq length.
                            e_emb.append(sequence_output[i, start + offset])
                            e_att.append(attention[i, :, start + offset])

                    if len(e_emb) > 0:
                        e_emb = torch.logsumexp(torch.stack(e_emb, dim=0), dim=0)
                        e_att = torch.stack(e_att, dim=0).mean(0)
                    else:
                        e_emb = torch.zeros(self.config.hidden_size).to(sequence_output)
                        e_att = torch.zeros(h, c).to(attention)
                else:
                    start, end = e[0]
                    if start + offset < c:
                        e_emb = sequence_output[i, start + offset]
                        e_att = attention[i, :, start + offset]
                    else:
                        e_emb = torch.zeros(self.config.hidden_size).to(sequence_output)
                        e_att = torch.zeros(h, c).to(attention)

                entity_embs.append(e_emb)
                entity_atts.append(e_att)

            entity_embs = torch.stack(entity_embs, dim=0)  # [n_e, d]
            entity_atts = torch.stack(entity_atts, dim=0)  # [n_e, h, seq_len]

            ht_i = torch.LongTensor(hts[i]).to(sequence_output.device)

            # obtain subject/object (head/tail) embeddings from entity embeddings.
            hs = torch.index_select(entity_embs, 0, ht_i[:, 0])
            ts = torch.index_select(entity_embs, 0, ht_i[:, 1])

            h_att = torch.index_select(entity_atts, 0, ht_i[:, 0])
            t_att = torch.index_select(entity_atts, 0, ht_i[:, 1])

            ht_att = (h_att * t_att).mean(1)  # average over all heads
            ht_att = ht_att / (ht_att.sum(1, keepdim=True) + 1e-30)
            ht_atts.append(ht_att)

            # obtain local context embeddings.
            rs = contract("ld,rl->rd", sequence_output[i], ht_att)

            hss.append(hs)
            tss.append(ts)
            rss.append(rs)

        rels_per_batch = [len(b) for b in hss]  # 每个文档实体对数
        hss = torch.cat(hss, dim=0)  # (num_ent_pairs_all_batches, emb_size)
        tss = torch.cat(tss, dim=0)  # (num_ent_pairs_all_batches, emb_size)
        rss = torch.cat(rss, dim=0)  # (num_ent_pairs_all_batches, emb_size)
        ht_atts = torch.cat(ht_atts, dim=0)  # (num_ent_pairs_all_batches, max_doc_len)

        return hss, rss, tss, ht_atts, rels_per_batch

    def get_hrt(self, sequence_output, attention, entity_pos, ent_men_mask):
        """
        :param sequence_output: [b, max_len, hidden_size]
        :param attention:  [b, h, max_len, max_len]
        :param entity_pos:
        :param hts:
        :return:
        """
        offset = 1 if self.config.transformer_type in ["bert", "roberta"] else 0
        n, h, _, c = attention.size()
        (bs, ent_num, men_num, max_len) = ent_men_mask.shape
        ent_s = []
        ent_rep = torch.zeros((n, ent_num, self.config.hidden_size)).to(sequence_output)
        men_rep = torch.zeros((n, ent_num, men_num, self.config.hidden_size)).to(sequence_output)
        # ent or men attention
        ent_att = torch.zeros((n, ent_num, c)).to(attention)
        men_att = torch.zeros((n, ent_num, men_num, c)).to(attention)
        # ent or men att rep
        ent_att_rep = torch.zeros((n, ent_num, self.config.hidden_size)).to(sequence_output)
        men_att_rep = torch.zeros((n, ent_num, men_num, self.config.hidden_size)).to(sequence_output)

        for i in range(len(entity_pos)):
            # 遍历一个文档中所有实体
            entity_embs= []
            for e_id,e in enumerate(entity_pos[i]):
                if len(e) > 1: # 多次出现的实体
                    e_emb, e_att = [], []
                    # 实体所有提及
                    for m_id,(start, end) in enumerate(e):
                        if start + offset < c:
                            # In case the entity mention is truncated due to limited max seq length.
                            e_emb.append(sequence_output[i, start + offset])
                            men_rep[i][e_id][m_id][:] = sequence_output[i, start + offset]
                            e_att.append(attention[i, :, start + offset])
                            men_att[i][e_id][m_id][:] = attention[i, :, start + offset].mean(0)
                            # men_att_rep[i][e_id][m_id] = \
                            #     torch.tanh(self.men_projector(torch.cat([sequence_output[i, start + offset],contract("ld,rl->rd", sequence_output[i], men_att[i][e_id][m_id].unsqueeze(0)).squeeze(0)], dim=0)))
                            # when long is more than 1024 ?
                        else:
                            men_rep[i][e_id][m_id][:] = torch.zeros(self.config.hidden_size).to(sequence_output)
                            e_emb.append(torch.zeros(self.config.hidden_size).to(sequence_output))
                            e_att.append(torch.zeros(h,c).to(attention))
                            men_att[i][e_id][m_id][:] = torch.zeros(c).to(attention)
                            # men_att_rep[i][e_id][m_id] = torch.zeros(self.config.hidden_size).to(sequence_output)

                    if len(e_emb) > 0: # 公式2
                        e_emb = torch.logsumexp(torch.stack(e_emb, dim=0), dim=0)
                        # torch.mean()
                        e_att = torch.stack(e_att, dim=0).mean(0) # 采取最后一层的注意力
                        ent_att[i][e_id][:]=e_att.mean(0)
                    else: # 当所有提及都在1024外，实体嵌入赋值0
                        e_emb = torch.zeros(self.config.hidden_size).to(sequence_output)
                        e_att = torch.zeros(h, c).to(attention)
                        ent_att[i][e_id][:] = torch.zeros(c).to(attention)
                else:
                    start, end = e[0]
                    if start + offset < c:
                        e_emb = sequence_output[i, start + offset]
                        men_rep[i][e_id][0][:] = e_emb
                        e_att = attention[i, :, start + offset].mean(0)
                        ent_att[i][e_id][:] = e_att
                        men_att[i][e_id][0][:] = e_att
                        # men_att_rep[i][e_id][0] = \
                        #     torch.tanh(self.men_projector(torch.cat([sequence_output[i, start + offset],contract("ld,rl->rd", sequence_output[i], e_att.unsqueeze(0)).squeeze(0)], dim=0)))
                    else:
                        e_emb = torch.zeros(self.config.hidden_size).to(sequence_output)
                        men_rep[i][e_id][0][:] = e_emb
                        e_att = torch.zeros(h, c).to(attention)
                        ent_att[i][e_id][:] = torch.zeros(c).to(attention)
                        men_att[i][e_id][0][:]=torch.zeros(c).to(attention)
                        # men_att_rep[i][e_id][0] = torch.zeros(self.config.hidden_size).to(sequence_output)
                ent_rep[i][e_id][:] = e_emb
                # ent_att_rep[i][e_id] = \
                #     torch.tanh(self.ent_projector(torch.cat([e_emb,contract("ld,rl->rd", sequence_output[i], ent_att[i][e_id].unsqueeze(0)).squeeze(0)], dim=0)))

                entity_embs.append(e_emb)
                # entity_atts.append(e_att)

            entity_embs = torch.stack(entity_embs, dim=0)  # [n_e, d]
            # entity_atts = torch.stack(entity_atts, dim=0)  # [n_e, h, seq_len]
            ent_s.append(entity_embs)
        all_ent_s = torch.cat(ent_s, dim=0) # [b, ent_num, hidden_size]
        # 根据实体或提及的注意力，利用可训练全连接层，得出融合注意力信息的实体或提及表示
        # for i in range(len(entity_pos)):
        #     for e_i in range(ent_num):
        #         ent_att_rep[i][e_i] = \
        #             torch.tanh(self.ent_projector(torch.cat([ent_rep[i][e_i],contract("ld,rl->rd", sequence_output[i], ent_att[i][e_i].unsqueeze(0)).squeeze(0)], dim=0)))
        #         for m_i in range(men_num):
        #             men_att_rep[i][e_i][m_i] = \
        #                 torch.tanh(self.men_projector(torch.cat([men_rep[i][e_i][m_i],contract("ld,rl->rd", sequence_output[i], men_att[i][e_i][m_i].unsqueeze(0)).squeeze(0)], dim=0)))
        return ent_rep, men_rep, all_ent_s, ent_att_rep, men_att_rep, men_att

    def forward_evi(self, doc_attn, sent_pos, batch_rel, offset):
        '''
        Forward computation for ER.
        Inputs:
            :doc_attn: (num_ent_pairs_all_batches, doc_len), attention weight of each token for computing localized context pooling.
            :sent_pos: list of list. The outer length = batch size. The inner list contains (start, end) position of each sentence in each batch.
            :batch_rel: list of length = batch size. Each entry represents the number of entity pairs of the batch.
            :offset: 1 for bert and roberta. Offset caused by [CLS] token.
        Outputs:
            :s_attn:  (num_ent_pairs_all_batches, max_sent_all_batch), sentence-level evidence distribution of each entity pair.
        '''

        max_sent_num = max([len(sent) for sent in sent_pos])
        rel_sent_attn = []
        for i in range(len(sent_pos)):  # for each batch
            # the relation ids corresponds to document in batch i is [sum(batch_rel[:i]), sum(batch_rel[:i+1]))
            curr_attn = doc_attn[sum(batch_rel[:i]):sum(batch_rel[:i + 1])]  # [ent_pairs, c]
            curr_sent_pos = [torch.arange(s[0], s[1]).to(curr_attn.device) + offset for s in sent_pos[i]]  # + offset

            curr_attn_per_sent = [curr_attn.index_select(-1, sent) for sent in
                                  curr_sent_pos]  # [sent_n, ht_n, sent_tokens]
            curr_attn_per_sent += [torch.zeros_like(curr_attn_per_sent[0])] * (
                        max_sent_num - len(curr_attn_per_sent))  # [max_sents,hts,sent_token]
            sum_attn = torch.stack([attn.sum(dim=-1) for attn in curr_attn_per_sent],
                                   dim=-1)  # sum across those attentions, [hts,max_sents,]
            rel_sent_attn.append(sum_attn)

        s_attn = torch.cat(rel_sent_attn, dim=0)
        return s_attn

    def evi_masks(self,evi_logits,hts,batch_rel,ent_men_mask):
        (ht_n, max_sent_n) = evi_logits.size()
        (bs,max_ent,max_men,doc_l) = ent_men_mask.size()
        evi_masks = torch.zeros((bs,max_ent,max_ent)).to(evi_logits)

        start = 0
        for i in range(bs):
            end = start + batch_rel[i]
            curr_evi_logits = evi_logits[start:end]
            curr_hts = hts[start:end]
            start = end
            mask = torch.where(((curr_evi_logits.sum(-1))>1)==1)[0] # [hts] 当前文档所有实体对证据句子大于1的实体对标记
            select_hts = curr_hts[mask]
            evi_masks[i][select_hts[:,0],select_hts[:,1]] = 1
        return evi_masks

    def forward(self,
                input_ids= None,
                input_mask= None,
                label= None,
                entity_pos= None,
                ent_men_mask= None,
                hts= None,
                sent_labels = None,
                sent_pos = None,
                relations = None,
                negative_mask = None,
                label_mask= None):
        output = {}
        offset = 1 if self.config.transformer_type in ["bert", "roberta"] else 0
        sequence_output, attention = self.encode(input_ids, input_mask)
        # get sequence outputs
        hs, rs, ts, doc_attn, batch_rel = self.get_hrt_evi(sequence_output, attention, entity_pos, hts, offset)

        # evi pred and loss
        if sent_labels != None: # human-annotated evidence available
            s_attn = self.forward_evi(doc_attn, sent_pos, batch_rel, offset) # [all_hts,max_sents] 每个文档每个实体对对每个句子的重要性
            output['evi_pred'] = F.pad(s_attn > self.evi_thresh, (0, self.max_sent_num - s_attn.shape[-1])) # [all_hts, 25]

        if sent_labels != None:  # supervised training with human evidence

            idx_used = torch.nonzero(relations[:, 1:].sum(dim=-1)).view(-1)
            # evidence retrieval loss (kldiv loss)
            s_attn = s_attn[idx_used]
            sent_labels = sent_labels[idx_used]
            norm_s_labels = sent_labels / (sent_labels.sum(dim=-1, keepdim=True) + 1e-30)
            norm_s_labels[norm_s_labels == 0] = 1e-30
            s_attn[s_attn == 0] = 1e-30
            evi_loss = self.loss_fnt_evi(s_attn.log(), norm_s_labels)
            output["loss"] = {'evi_loss': evi_loss.to(sequence_output)}

        if self.args.base_train:
            hs = torch.tanh(self.head_extractor_1(torch.cat([hs, rs], dim=-1)))
            ts = torch.tanh(self.tail_extractor_1(torch.cat([ts, rs], dim=-1)))
            # split into several groups.
            b1 = hs.view(-1, self.emb_size // self.block_size, self.block_size)
            b2 = ts.view(-1, self.emb_size // self.block_size, self.block_size)

            bl = (b1.unsqueeze(3) * b2.unsqueeze(2)).view(-1, self.emb_size * self.block_size)
            logits = self.bilinear(bl)
            output["rel_pred"] = self.loss_fnt_1.get_label(logits, num_labels=self.max_num_labels) # 注意区分用于预测结果的标签数和用于分类器的关系类别数
            # 利用模型预测的较大四个关系类型作为预测结果
            scores_topk = self.loss_fnt_1.get_score(logits, self.max_num_labels)
            output["scores"] = scores_topk[0]
            output["topks"] = scores_topk[1]

            loss = self.loss_fnt_1(logits.float(), relations.float())
            output["loss"]["rel_loss"] = loss.to(sequence_output)
        else: # 利用训练的基础模型，根据预测的证据句子数对指定实体的实体对表示重新进行轴注意力。
            ent_rep_1, men_rep, all_ent_s, ent_att_rep, men_att_rep, men_att = self.get_hrt(sequence_output, attention, entity_pos, ent_men_mask)

            # get entity pairs maker in every doc according to having one more evidence sentences
            hts = [torch.tensor(ht) for ht in hts]
            hts = torch.cat(hts,dim=0)
            evi_hts_mask = self.evi_masks(output['evi_pred'],hts,batch_rel,ent_men_mask) # 需要添加其他实体对信息的实体对id掩码

            # get mention mask | ent_men_mask:[bs, max_ent_cnt, max_men_cnt,max_seq_len]
            men_mask = torch.sum(ent_men_mask, dim=-1, keepdim=True)  # shape:[bs, max_ent_cnt, max_men_cnt, 1]
            men_mask = (1.0 - men_mask) * -1000000.0
            men_mask = men_mask.to(sequence_output)
            # using simple attention for entity rep of entity_spec
            ent_rep_spec, men_weights = self.simpleAttention(ent_rep_1, men_rep, men_mask)
            # 根据simpleAttention得出dots，这里不采用得出的指定实体嵌入：
            ent_att_spec = torch.matmul(men_weights, men_att)  # [bs,max_ent,max_ent,c]
            ent_att_spec_rev = ent_att_spec.transpose(1, 2).contiguous()
            ht_att = (ent_att_spec * ent_att_spec_rev) / ((ent_att_spec * ent_att_spec_rev).sum(-1, keepdim=True) + 1e-5)  # [bs,max_ent,max_ent,c]
            # 根据指定实体注意力得出对应上下文表示
            ht_rep = torch.zeros((ht_att.shape[0], ht_att.shape[1], ht_att.shape[2], sequence_output.shape[-1])).to(
                ent_rep_spec)
            for i in range(ht_rep.shape[0]):
                rep = torch.einsum('nl,ld->nd', [ht_att[i].view(-1, ht_att.shape[-1]), sequence_output[i]])
                rep = rep.view(ht_att.shape[1], ht_att.shape[2], -1)
                ht_rep[i] = rep
            ent_rep_spec_rev = ent_rep_spec.transpose(1, 2).contiguous()
            # 拼接ht_att和ent_rep_spec
            ent_rep_spec = torch.tanh(self.head_extractor_2(torch.cat([ent_rep_spec, ht_rep], dim=-1)))
            ent_rep_spec_rev = torch.tanh(self.tail_extractor_2(torch.cat([ent_rep_spec_rev, ht_rep], dim=-1)))
            # bilinear classifier
            b1 = ent_rep_spec.view(ent_rep_spec.shape[0], ent_rep_spec.shape[1], ent_rep_spec.shape[2],
                                   self.emb_size // self.block_size, self.block_size)
            b2 = ent_rep_spec_rev.view(ent_rep_spec.shape[0], ent_rep_spec.shape[1], ent_rep_spec.shape[2],
                                       self.emb_size // self.block_size, self.block_size)
            bl = (b1.unsqueeze(5) * b2.unsqueeze(4)).view(ent_rep_spec.shape[0], ent_rep_spec.shape[1],
                                                          ent_rep_spec.shape[2], self.emb_size * self.block_size)
            if negative_mask is not None:
                bl_e = bl * negative_mask.unsqueeze(-1)
            else:
                bl_e = bl

            feature = self.projector(bl_e)  # [bs,ne,ne,hidden_size]
            feature_last = feature.clone()
            feature_last[evi_hts_mask] = feature.clone()[evi_hts_mask] + self.axial_transformer(feature).clone()[
                evi_hts_mask]  # 指定实体对添加其他实体对的信息

            logits = self.last_classifier(feature_last)
            # atloss
            output["rel_pred"] = self.loss_fnt.get_label(logits, num_labels=self.max_num_labels)
            if label is not None:
                loss = self.loss_fnt(logits.float(), label, label_mask.to(logits))
                output['loss']['rel_loss'] = loss.to(sequence_output)

        return output