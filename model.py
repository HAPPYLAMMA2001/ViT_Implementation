import torch
import torch.nn as nn
import math

class PatchEmbedding(nn.Module): # Yes
    def __init__(self, image_size: int, patch_size: int, in_channels: int, d_model: int):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        
        # Linear projection of flattened patches
        self.proj = nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)
        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
    def forward(self, x):
        B = x.shape[0]
        # Project and flatten patches
        x = self.proj(x)  # (B, d_model, H/p, W/p)
        x = x.flatten(2)  # (B, d_model, N)
        x = x.transpose(1, 2)  # (B, N, d_model)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        return x

    
class PositionalEncoding(nn.Module):  # Yes
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self. d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # create a matri of shape [seq_len, d_model]
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # shape [seq_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)#.transpose(0, 1)

        self.register_buffer('pe', pe) # stores it as buffer meaning final vector will be saved in the model state_dict

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).to(x.device).requires_grad_(False) # add positional encoding to the input embeddings
        return self.dropout(x)
    
class LayerNorm(nn.Module): # Yes
    def __init__(self, eps: float = 1e-6, d_model: int = 512) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        x = x.float()
        mean = x.mean(dim = -1, keepdim=True)
        std = x.std(dim = -1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias # layer normalization formula
    

class FeedForwardBlock(nn.Module): # Yes
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.l1 = nn.Linear(d_model, d_ff) # first linear layer
        self.dropout = nn.Dropout(dropout)
        self.l2 = nn.Linear(d_ff,d_model) # second linear layer

    def forward(self, x): 
        x = self.l1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.l2(x)
        return x 
    

class MultiHeadAttention(nn.Module): # Yes
    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads


        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_k = d_model // n_heads # dimension of each head

        self.wq = nn.Linear(d_model, d_model) # linear layer for query
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)

        self.wo = nn.Linear(d_model, d_model) # linear layer for output
        self.dropout = nn.Dropout(dropout)
    
    @staticmethod
    def attention(query, key, value, mask, dropout=nn.Dropout):
        d_k = query.shape[-1]

        attention_score = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_score = attention_score.masked_fill(mask == 0, -1e9) 

        attention_score = torch.softmax(attention_score, dim=-1) # softmax on the last dimension
        if dropout is not None:
            attention_score = dropout(attention_score)

        output = (attention_score @ value)
        return output, attention_score

    def forward(self, q, k, v, mask):
        query = self.wq(q)
        key = self.wk(k)
        value = self.wv(v)


        query = query.view(query.shape[0], query.shape[1], self.n_heads, self.d_k).transpose(1, 2) # shape [batch_size, n_heads, seq_len, d_k]
        key = key.view(key.shape[0], key.shape[1], self.n_heads, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.n_heads, self.d_k).transpose(1, 2)

        x, self.attention_scores = MultiHeadAttention.attention(query, key, value, mask, self.dropout)

        x = x.transpose(1,2)
        x = x.contiguous()
        x = x.view(x.shape[0], -1, self.n_heads * self.d_k)
        return self.wo(x)
    

class residualConnection(nn.Module): # Yes
    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNorm()
    
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x))) # residual connection formula
    

class EncoderBlock(nn.Module): # Yes
    def __init__(self, self_attention_block: MultiHeadAttention, feed_forward_block: FeedForwardBlock, d_model: int, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection1 = residualConnection(dropout)
        self.residual_connection2 = residualConnection(dropout)

    def forward(self, x, src_mask):
        x = self.residual_connection1(x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connection2(x, self.feed_forward_block)
        return x
    
class Encoder(nn.Module): # yes
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNorm()

    def forward(self, x, src_mask):
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)
    

class MLPHead(nn.Module): #Yes
    def __init__(self, d_model: int, num_classes: int):
        super().__init__()
        self.norm = LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, num_classes)
        )
    
    def forward(self, x):
        x = x[:, 0]  # Take CLS token
        x = self.norm(x)
        x = self.mlp(x)
        return x

class VisionTransformer(nn.Module):
    def __init__(self, encoder: Encoder, patch_embed: PatchEmbedding, pos_embed: PositionalEncoding, mlp_head: MLPHead):
        super().__init__()
        self.patch_embed = patch_embed
        self.pos_embed = pos_embed
        self.encoder = encoder
        self.mlp_head = mlp_head

    def forward(self, x):
        # Patch embedding
        x = self.patch_embed(x)
        # Add positional encoding
        x = self.pos_embed(x)
        # Encode
        x = self.encoder(x, None)  # No mask needed for ViT
        # Classify
        x = self.mlp_head(x)
        return x

def build_vit(image_size: int, patch_size: int, in_channels: int, num_classes: int, d_model: int = 768, N: int = 12, 
              h: int = 12, dropout: float = 0.1, d_ff: int = 3072) -> VisionTransformer:
    
    # Create patch embedding layer
    patch_embed = PatchEmbedding(image_size, patch_size, in_channels, d_model)
    
    # Calculate sequence length (number of patches + 1 for CLS token)
    seq_len = (image_size // patch_size) ** 2 + 1
    
    # Create positional encoding
    pos_embed = PositionalEncoding(d_model, seq_len, dropout)
    
    # Create encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttention(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, d_model, dropout)
        encoder_blocks.append(encoder_block)
    
    # Create encoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    
    # Create MLP head for classification
    mlp_head = MLPHead(d_model, num_classes)
    
    # Create ViT
    vit = VisionTransformer(encoder, patch_embed, pos_embed, mlp_head)
    
    # Initialize weights
    for p in vit.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return vit

