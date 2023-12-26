import torch
import torch.nn as nn
import torch.nn.functional as F

class ATLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, labels, label_mask):
        # TH label
        """
        :param logits: [b,max_ent, max_ent, num_labels]
        :param labels: [b,max_ent, max_ent, num_labels]
        :return:
        """
        th_label = torch.zeros_like(labels, dtype=torch.float).to(labels)
        th_label[:, :, :, 0] = 1.0
        labels[:, :, :, 0] = 0.0

        p_mask = labels + th_label
        # n_mask = 1 - labels
        n_mask = ~ labels
        # Rank positive classes to TH
        logit1 = logits - (~ p_mask) * 1e30 # 正实体对保留th和正关系的logits，负实体对只保留th
        loss1 = -(F.log_softmax(logit1, dim=-1) * labels).sum(3) # 公式前半部分正实体对所有正关系之和，负实体对0

        # Rank TH to negative classes
        logit2 = logits - (~ n_mask) * 1e30  # 公式后半部分，只保留th和非黄金关系logit
        loss2 = -(F.log_softmax(logit2, dim=-1) * th_label).sum(3)

        # Sum two parts
        loss = loss1 + loss2
        # label_mask: [bs, max_ent_cnt, max_ent_cnt]
        loss_per_example = torch.sum(loss * label_mask, dim=[1, 2]) / torch.sum(label_mask, dim=[1, 2])
        loss = torch.mean(loss_per_example)
        return loss

    def get_label(self, logits, num_labels=-1):
        th_logit = logits[:, :, :, 0].unsqueeze(-1)
        output = torch.zeros_like(logits).to(logits)
        mask = (logits > th_logit)
        if num_labels > 0:
            top_v, _ = torch.topk(logits, num_labels, dim=-1)
            top_v = top_v[:, :, :, -1]
            mask = (logits >= top_v.unsqueeze(-1)) & mask #分数大于th且大于第四个大的分数
        output[mask] = 1.0
        output[:, :, :, 0] = (output.sum(-1) == 0.).to(logits) # 预测没有正例，则记录关系为NA
        return output


class ATLoss_1(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, logits, labels):
        # TH label
        th_label = torch.zeros_like(labels, dtype=torch.float).to(labels)
        th_label[:, 0] = 1.0
        labels[:, 0] = 0.0

        p_mask = labels + th_label
        n_mask = 1 - labels

        # Rank positive classes to TH
        logit1 = logits - (1 - p_mask) * 1e30
        loss1 = -(F.log_softmax(logit1, dim=-1) * labels).sum(1) # 第一类损失，正实体对包含所有正关系，负实体对0
        # Rank TH to negative classes
        logit2 = logits - (1 - n_mask) * 1e30
        loss2 = -(F.log_softmax(logit2, dim=-1) * th_label).sum(1) # 第二类损失，正实体对以及负实体对只包含th
        # Sum two parts
        loss = loss1 + loss2
        loss = loss.mean()
        return loss

    def get_label(self, logits, num_labels=-1):

        th_logit = logits[:, 0].unsqueeze(1)  # theshold is norelation
        output = torch.zeros_like(logits).to(logits)
        mask = (logits > th_logit)
        if num_labels > 0:
            top_v, _ = torch.topk(logits, num_labels, dim=1)
            top_v = top_v[:, -1] # smallest logits among the num_labels
            # predictions are those logits > thresh and logits >= smallest
            mask = (logits >= top_v.unsqueeze(1)) & mask
        output[mask] = 1.0
        # if no such relation label exist: set its label to 'Nolabel'
        output[:, 0] = (output.sum(1) == 0.).to(logits)
        return output

    def get_score(self, logits, num_labels=-1):

        if num_labels > 0:
            return torch.topk(logits, num_labels, dim=1)
        else:
            return logits[:, 1] - logits[:, 0], 0
