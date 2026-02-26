import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



class Custom_loss(nn.Module):
    def __init__(self, margin=0.7, triplet_weight=1.0, contrastive_weight=2.0, init_temp=0.07):
        super().__init__()
        self.margin = margin
        self.triplet_weight = triplet_weight
        self.contrastive_weight = contrastive_weight
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.tensor(1 / init_temp).log())

    def _get_batch_similarity_matrix(self, v_img, t_text, mask):
        """
        Computes NxN similarity matrix for all images vs all texts (for InfoNCE)
        """
        
        #Einsum for batched dot products: [B, B, 16, L]
        sim_all = torch.einsum('ipd, jld -> ijpl', v_img, t_text) 
        word_scores_all, _ = torch.max(sim_all, dim=2)
        
        mask_float = mask.float().unsqueeze(0)
        word_scores_all = word_scores_all * mask_float
        
        valid_words = torch.sum(mask.float(), dim=1).clamp(min=1.0).unsqueeze(0) # [1, B]
        return torch.sum(word_scores_all, dim=2) / valid_words # [B, B]

    def _get_paired_similarity(self, v_img, t_text, mask):
        """
        similarity for parallel elements in a batch (for Triplet Loss)
        """
        
        sim_matrix = torch.bmm(v_img, t_text.transpose(1, 2))
        word_scores, _ = torch.max(sim_matrix, dim=1)
        
        word_scores = word_scores * mask.float()
        valid_words = torch.sum(mask.float(), dim=1).clamp(min=1.0)
        return torch.sum(word_scores, dim=1) / valid_words

    def forward(self, v_main, v_aug, t_pos, t_neg, m_pos, m_neg):
        with torch.no_grad():
            self.logit_scale.clamp_(0, 4.6052)
            
        s = self.logit_scale.exp() 
        labels = torch.arange(v_main.size(0), device=v_main.device)

        ################################################
        #Contrastive loss: Main motor of training. Pushing all samples in batch apart except of the one
        ################################################
        logits_main = self._get_batch_similarity_matrix(v_main, t_pos, m_pos) * s
        loss_i2t_main = F.cross_entropy(logits_main, labels)
        loss_t2i_main = F.cross_entropy(logits_main.T, labels)
        loss_main = (loss_i2t_main + loss_t2i_main) / 2.0

        logits_aug = self._get_batch_similarity_matrix(v_aug, t_pos, m_pos) * s
        loss_i2t_aug = F.cross_entropy(logits_aug, labels)
        loss_t2i_aug = F.cross_entropy(logits_aug.T, labels)
        loss_aug = (loss_i2t_aug + loss_t2i_aug) / 2.0
        
        loss_contrastive = (loss_main + loss_aug) / 2.0

        ################################################
        #Triplet loss: for the samples with the hard, single or small word changes
        ################################################
        sim_pos_main = self._get_paired_similarity(v_main, t_pos, m_pos)
        sim_pos_aug  = self._get_paired_similarity(v_aug, t_pos, m_pos)
        
        sim_neg_main = self._get_paired_similarity(v_main, t_neg, m_neg)
        sim_neg_aug  = self._get_paired_similarity(v_aug, t_neg, m_neg)


        loss_triplet_main = F.relu(self.margin - sim_pos_main + sim_neg_main).mean()
        loss_triplet_aug  = F.relu(self.margin - sim_pos_aug + sim_neg_aug).mean()
        
        loss_triplet = (loss_triplet_main + loss_triplet_aug) / 2.0

        return (self.triplet_weight * loss_triplet) + (self.contrastive_weight * loss_contrastive)
               
               
               
               

def calibrate_threshold(pos_scores, neg_scores):
    """
    Finds the threshold that maximizes training Balanced Accuracy
    """
    
    best_balanced_acc = 0.0
    best_t = 0.5
    
    #100 steps resolution
    for t in np.linspace(0, 1, 101):
        tp = (pos_scores > t).sum()
        fn = (pos_scores <= t).sum()
        
        tn = (neg_scores < t).sum()
        fp = (neg_scores >= t).sum()
        
        #Recall and sensitiviy for bal_acc
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # Balanced Accuracy
        balanced_acc = (recall + specificity) / 2
        
        if balanced_acc > best_balanced_acc:
            best_balanced_acc = balanced_acc
            best_t = t
            
    return best_t               
               
               
               
def calculate_metrics(pos_scores, neg_scores, threshold):
    
    tp = (pos_scores > threshold).sum()
    
    fn = (pos_scores <= threshold).sum()
    
    tn = (neg_scores < threshold).sum()

    fp = (neg_scores >= threshold).sum()
    
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    balanced_acc = (recall + specificity) / 2
    
    return balanced_acc, recall, specificity               
               
               
               
               
               
               
               