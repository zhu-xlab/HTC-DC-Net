import torch
import torch.nn as nn


class PatchTransformerEncoder(nn.Module):
    def __init__(self, in_channels, patch_size=10, embedding_dim=128, num_heads=4):
        super(PatchTransformerEncoder, self).__init__()
        encoder_layers = nn.TransformerEncoderLayer(embedding_dim, num_heads, dim_feedforward=1024)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=4)  # takes shape S,N,E

        self.embedding_convPxP = nn.Conv2d(in_channels, embedding_dim,
                                           kernel_size=patch_size, stride=patch_size, padding=0)

        self.positional_encodings = nn.Parameter(torch.rand(1200, embedding_dim), requires_grad=True) #changed from 500 to 1200

    def forward(self, x):
        embeddings = self.embedding_convPxP(x).flatten(2)  # .shape = n,c,s = n, embedding_dim, s
        # embeddings = nn.functional.pad(embeddings, (1,0))  # extra special token at start ?
        embeddings = embeddings + self.positional_encodings[:embeddings.shape[2]].T.unsqueeze(0)

        # change to S,N,E format required by transformer
        embeddings = embeddings.permute(2, 0, 1)
        x = self.transformer_encoder(embeddings)  # .shape = S, N, E
        return x

class PatchTransformerEncoderGlobal(nn.Module):
    def __init__(self, in_channels, patch_size=10, embedding_dim=128, num_heads=4, cfgs={}):
        super(PatchTransformerEncoderGlobal, self).__init__()
        encoder_layers = nn.TransformerEncoderLayer(embedding_dim, num_heads, dim_feedforward=1024)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=4)  # takes shape S,N,E

        self.embedding_convPxP = nn.Conv2d(in_channels, embedding_dim,
                                           kernel_size=patch_size, stride=patch_size, padding=0)

        self.positional_encodings = nn.Parameter(torch.rand(4000, embedding_dim), requires_grad=True)

        self.pooling = cfgs.get("pooling", False)
        if self.pooling:
            self.poolavg = nn.AdaptiveAvgPool1d(output_size=128)
            self.poolmax = nn.AdaptiveMaxPool1d(output_size=128)
            #self.conv_pool = nn.Conv1d(256, 256, 1)

    def forward(self, x):
        x_center = x[0]
        embedding = self.embedding_convPxP(x_center).flatten(2)
        embedding = embedding + self.positional_encodings[:embedding.shape[2]].T.squeeze(0)
        padding_mask = torch.zeros(embedding.shape[0], embedding.shape[2]).to(embedding.device)
        
        embeddings = []
        if self.pooling:
            for i, xx in enumerate(x[1:]):
                if xx is not None:
                    emb = self.embedding_convPxP(xx).flatten(2)
                    embeddings.append(emb + self.positional_encodings[(i+1)*emb.shape[2]:(i+2)*emb.shape[2]].T.squeeze(0))
            if len(embeddings):
                embeddings = torch.cat(embeddings, dim=2)
                embeddings_avg = self.poolavg(embeddings)
                embeddings_max = self.poolmax(embeddings)
                embeddings = torch.cat((embeddings_avg, embeddings_max), dim=2).permute(0, 2, 1).contiguous()
                #embeddings = self.conv_pool(embeddings).permute(0, 2, 1).contiguous()
                embeddings = embeddings.permute(0, 2, 1).contiguous()
                padding_masks = torch.zeros(embeddings.shape[0], embeddings.shape[2]).to(embeddings.device)
                embeddings = torch.cat((embedding, embeddings), dim=2)
                padding_masks = torch.cat((padding_mask, padding_masks), dim=1)
            else:
                embeddings = embedding
                padding_masks = padding_mask
        else:
            embeddings.append(embedding)
            padding_masks = [padding_mask] 
            for i, xx in enumerate(x[1:]):
                if xx is not None:
                    emb = self.embedding_convPxP(xx).flatten(2)
                    embeddings.append(emb + self.positional_encodings[(i+1)*emb.shape[2]:(i+2)*emb.shape[2]].T.squeeze(0))
                    padding_masks.append(torch.zeros(emb.shape[0], emb.shape[2]).to(emb.device))
                else:
                    emb = torch.zeros_like(embedding).to(embedding.device)
                    embeddings.append(emb)
                    padding_masks.append(torch.ones(emb.shape[0], emb.shape[2]).to(emb.device))
            embeddings = torch.cat(embeddings, dim=2)
            padding_masks = torch.cat(padding_masks, dim=1)

        # change to S,N,E format required by transformer
        embeddings = embeddings.permute(2, 0, 1)

        x = self.transformer_encoder(embeddings, src_key_padding_mask=padding_masks)  # .shape = S, N, E
        return x

class PixelWiseDotProduct(nn.Module):
    def __init__(self):
        super(PixelWiseDotProduct, self).__init__()

    def forward(self, x, K):
        n, c, h, w = x.size()
        _, cout, ck = K.size()
        assert c == ck, "Number of channels in x and Embedding dimension (at dim 2) of K matrix must match"
        y = torch.matmul(x.view(n, c, h * w).permute(0, 2, 1), K.permute(0, 2, 1))  # .shape = n, hw, cout
        return y.permute(0, 2, 1).view(n, cout, h, w)
