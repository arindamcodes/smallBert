import torch
from torch import nn
from torch.nn import functional as F



class SelfAttention(nn.Module):
    def __init__(self, head_size, d_k): # d_k dimension of embed vector
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(d_k, head_size)
        self.keys = nn.Linear(d_k, head_size)
        self.values = nn.Linear(d_k, head_size)

    
    def forward(self, x):
        B, C, d_k = x.shape  # B--> batch size, C --> con_length
        q = self.query(x) #(B, C, H) H--> Head size
        k = self.query(x) #(B, C, H)
        v = self.values(x) #(B, C, H)

        score = q @ k.transpose(-2, -1) * (d_k ** -0.5)  #(B, C, H)
        prob = F.softmax(score, dim=-1)
        out = prob @ v # (B, C, C) @ (B, C, H) = (B, C, H)

        return out
    

class MultiheadAttention(nn.Module):
    
    def __init__(self, head_size, d_k, n_heads):
        super(MultiheadAttention, self).__init__()
        self.heads = nn.ModuleList(SelfAttention(head_size, d_k) for _ in range(n_heads))
        self.res_fc = nn.Linear(n_heads * head_size, d_k)
    def forward(self, x):
        heads = [h(x) for h in self.heads]
        heads_concat = torch.concat(heads, dim=-1)  #(B, C, n_heads * head_size)
        out = self.res_fc(heads_concat)
        return out
    

class FC(nn.Module):

    def __init__(self, d_k):
        super(FC, self).__init__()
        self.fc = nn.Linear(d_k, 4 * d_k) # 4 is in the original paper
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(4 * d_k, d_k)

    def forward(self, x):
        out = self.fc2(self.gelu(self.fc(x)))
        return out
    

class EncoderBlock(nn.Module):

    def __init__(self, d_k, n_heads):
        super(EncoderBlock, self).__init__()
        self.encoder = MultiheadAttention(head_size=d_k // n_heads, d_k=d_k, n_heads=n_heads)
        self.layernorm_pre_encoder = nn.LayerNorm(d_k)
        self.fc = FC(d_k)
        self.layernorm_pre_fc = nn.LayerNorm(d_k)

    def forward(self, x):
        x = x + self.encoder(self.layernorm_pre_encoder(x))    # Adding residual connection
        x = x + self.fc(self.layernorm_pre_fc(x))        # Adding residual connection
        return x


class Bert(nn.Module):

    def __init__(self, 
                 vocab_size, 
                 context_length, 
                 d_k, 
                 n_heads,
                 n_layers, 
                 device):

        super(Bert, self).__init__()
        self.device = device
        self.embeddings = nn.Embedding(vocab_size, d_k)
        self.positional_embeddings = nn.Embedding(context_length, d_k)
        self.encoder = nn.Sequential(*[EncoderBlock(d_k, n_heads) for _ in range(n_layers)])
        self.output = nn.Linear(d_k, vocab_size)

    
    def forward(self, x):
        # x --> (B, C)
        B, C = x.shape
        token_embed = self.embeddings(x) #(B, C, d_k)
        pos_embed = self.positional_embeddings(torch.arange(C, device=self.device)) #(C, d_k)
        x = token_embed + pos_embed # (B, C, d_k)
        x = self.encoder(x) 
        logits = self.output(x)

        return logits




