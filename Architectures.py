import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import os

class Image_encoder(nn.Module):
    def __init__(self, embed_dim, weights_path="Models/Pretrained/resnet50_weights.pth"):
        """
        Image encoder for extraction of feature vector from image to compare 
        it with the feature vector from the text.
        
        It sues resnet backbone and then follows by dense layers. Transfer learning applied
        """    
        super().__init__()
        
        resnet = models.resnet50(weights=None)
        if os.path.exists(weights_path):
            state_dict = torch.load(weights_path, map_location='cpu', weights_only=True)
            resnet.load_state_dict(state_dict)
            print("Loaded ResNet50 backbone (Offline)")

        #Removing original avgf and dense layers
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        
        #Adding small conv layers to our training model
        self.spatial_neck = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=1), 
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        #No dense layer now. Just the projection without non-linear activation
        self.local_proj = nn.Sequential(
            nn.Linear(512, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
    #we keep the 7x7 info (more spatial. and do process it in the way spatial info could be potentially compared to text embedding
    def forward(self, x):
        x = self.backbone(x)        
        x = self.spatial_neck(x)   
        
        B, C, H, W = x.shape
        x = x.view(B, C, H * W).permute(0, 2, 1) 
        
        return self.local_proj(x)  

    def train(self, mode=True):
        """
        
        Overwrites the default train() to ensure BN layers in the backbone
        remain in eval mode (frozen stats) even during fine-tuning.
        We keep the statistics learned in the ImageNet Resnet training
        
        """
        super().train(mode)
        for module in self.backbone.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()


class Text_encoder(nn.Module):
    """
    Text encoder for the same reason as image encoder but inverted. Now we are vectorizing the text
    
    Initially I tried to use the LSTM but got poor results - at least in first epochs of training.
    
    #LLM helped me with the concept of the bidirectional Residual bidirectional LSTM and the small attention layer

    """
    class ResidualLSTM(nn.Module):
        def __init__(self, dim, dropout=0.1):
            super().__init__()
            #setting the bidirectional for better coverage of the sequence.
            #It is especially usefull when only some small word is changed in the caption
            #and the rest of the caption seems to be well describing the picture. But in reality caption is false
            self.lstm = nn.LSTM(dim, dim // 2, num_layers=1, 
                                batch_first=True, bidirectional=True)
            self.norm = nn.LayerNorm(dim)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.norm(x + self.dropout(out))

    def __init__(self, vocab_size, word_dim, hidden_dim, embed_dim, depth=3):
        super().__init__()
        
        #Embedding tokens + projection for lstm
        self.embedding = nn.Embedding(vocab_size, word_dim, padding_idx=0)
        self.proj = nn.Linear(word_dim, hidden_dim)
        
        #LSTM layers
        self.layers = nn.ModuleList([self.ResidualLSTM(hidden_dim) for _ in range(depth)])
        
        #
        #Projections into the embedded dimmension so its comparable with the image vector
        self.token_proj = nn.Sequential(
            nn.Linear(hidden_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.LeakyReLU(0.02),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, x):
        mask = (x != 0) 
        x = self.proj(self.embedding(x)) 
        
        for layer in self.layers:
            x = layer(x)
        
        
        out = self.token_proj(x)
        return out, mask
    
    
class Siamese_model(nn.Module):
    """
    Model which merges 2 models, text and image encoder, and then compares their produced vectors to determine
    if the text describes the image well, and if there are not any mismatches in the caption
    """
    def __init__(self, Image_model, Text_model, device):
        super().__init__()
        self.img_enc = Image_model
        self.txt_enc = Text_model
        self.device = device

    def move_to_device(self, device=None):
        target = device if device else self.device
        self.img_enc.to(target)
        self.txt_enc.to(target)

    def train_mode(self):
        self.img_enc.train()
        self.txt_enc.train()
            
    def eval_mode(self):
        self.img_enc.eval()
        self.txt_enc.eval()        

    def forward(self, img, aug_img, pos_cap, neg_cap):
        #Image
        v_main = F.normalize(self.img_enc(img), p=2, dim=-1, eps=1e-8)
        v_aug  = F.normalize(self.img_enc(aug_img), p=2, dim=-1, eps=1e-8)

        #Text
        t_pos, m_pos = self.txt_enc(pos_cap)
        t_neg, m_neg = self.txt_enc(neg_cap)
        
        t_pos = F.normalize(t_pos, p=2, dim=-1, eps=1e-8)
        t_neg = F.normalize(t_neg, p=2, dim=-1, eps=1e-8)

        return v_main, v_aug, t_pos, t_neg, m_pos, m_neg

    def predict(self, image_tensor, text_string, threshold=0.8):
        """
        Inference mode: Computes 1-to-1 similarity directly.
        
        Suprisingly the treshold was in the range in 0.4 for best bal_acc and 0.51 for best specificity (small loss only 0.01 of bal_acc)
        Put it at the 0.5
        """
        self.eval_mode()
        #If single image only, then unsqueeze it to batch 1
        if image_tensor.ndimension() == 3:
            image_tensor = image_tensor.unsqueeze(0)
            
        #Transforming text to tokens by the tokenizer
        caption_tokens = self.tokenizer.encode(text_string)
        #We also move to device to match image tensor
        caption_tensor = torch.tensor(caption_tokens).unsqueeze(0).to(image_tensor.device)
        
        with torch.no_grad():
            v_img = self.img_enc(image_tensor)
            v_img = F.normalize(v_img, p=2, dim=-1, eps=1e-8)
            
            t_text, mask = self.txt_enc(caption_tensor)
            t_text = F.normalize(t_text, p=2, dim=-1, eps=1e-8)
            
            # 1-to-1 cross-attention
            #Computes cosine similarity between every image patch and every word token
            sim_matrix = torch.bmm(v_img, t_text.transpose(1, 2))
            
            #takes the highest patch similarity score for each individual word
            word_scores, _ = torch.max(sim_matrix, dim=1)
            
            # Mask out padding tokens so they do not influence the score
            word_scores = word_scores * mask.float()
            
            valid_words = torch.sum(mask.float(), dim=1).clamp(min=1.0)
            
            similarity = torch.sum(word_scores, dim=1) / valid_words
            is_match = (similarity > threshold).float()
            
            
            return is_match, similarity 
    
    
    
    
    
    
    
    
    