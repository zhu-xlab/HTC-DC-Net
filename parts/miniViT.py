import torch
import torch.nn as nn

from .layers import PatchTransformerEncoder, PatchTransformerEncoderGlobal, PixelWiseDotProduct


class mViT(nn.Module):
    def __init__(self, in_channels, n_query_channels=128, patch_size=16, dim_out=256,
                 embedding_dim=128, num_heads=4, norm='linear', adanum=False):
        super(mViT, self).__init__()
        self.norm = norm
        self.adanum = adanum
        self.n_query_channels = n_query_channels
        self.patch_transformer = PatchTransformerEncoder(in_channels, patch_size, embedding_dim, num_heads)
        self.dot_product_layer = PixelWiseDotProduct()

        self.conv3x3 = nn.Conv2d(in_channels, embedding_dim, kernel_size=3, stride=1, padding=1)
        self.regressor = nn.Sequential(nn.Linear(embedding_dim, 256),
                                       nn.LeakyReLU(),
                                       nn.Linear(256, 256),
                                       nn.LeakyReLU(),
                                       nn.Linear(256, dim_out))

    def forward(self, x):
        # n, c, h, w = x.size()
        tgt = self.patch_transformer(x.clone())  # .shape = S, N, E

        x = self.conv3x3(x)

        regression_head, queries = tgt[0, ...], tgt[1:self.n_query_channels + 1, ...]

        # Change from S, N, E to N, S, E
        queries = queries.permute(1, 0, 2)
        range_attention_maps = self.dot_product_layer(x, queries)  # .shape = n, n_query_channels, h, w

        y = self.regressor(regression_head)  # .shape = N, dim_out
        if self.adanum:
            return y, range_attention_maps
        else:
            if self.norm == 'linear':
                y = torch.relu(y)
                eps = 0.1
                y = y + eps
            elif self.norm == 'softmax':
                return torch.softmax(y, dim=1), range_attention_maps
            else:
                y = torch.sigmoid(y)
            y = y / y.sum(dim=1, keepdim=True)
        return y, range_attention_maps


class mViTHTC(nn.Module):
    def __init__(self, in_channels, n_query_channels=128, patch_size=16, dim_out=256,
                 embedding_dim=128, num_heads=4, norm='linear', adanum=False):
        super(mViTHTC, self).__init__()
        self.norm = norm
        self.adanum = adanum
        self.n_query_channels = n_query_channels
        self.patch_transformer = PatchTransformerEncoder(in_channels, patch_size, embedding_dim, num_heads)
        self.dot_product_layer = PixelWiseDotProduct()

        self.conv3x3 = nn.Conv2d(in_channels, embedding_dim, kernel_size=3, stride=1, padding=1)
        self.conv3x3_bg = nn.Conv2d(in_channels, embedding_dim, kernel_size=3, stride=1, padding=1)
        self.regressor = nn.Sequential(nn.Linear(embedding_dim, 256),
                                       nn.LeakyReLU(),
                                       nn.Linear(256, 256),
                                       nn.LeakyReLU(),
                                       nn.Linear(256, dim_out))

    def forward(self, x):
        # n, c, h, w = x.size()
        tgt = self.patch_transformer(x.clone())  # .shape = S, N, E

        x_fg = self.conv3x3(x)
        x_bg = self.conv3x3_bg(x)

        regression_head, queries, queries_bg = tgt[0, ...], tgt[1:self.n_query_channels + 1, ...], tgt[self.n_query_channels+1: 2*self.n_query_channels+1, ...]

        # Change from S, N, E to N, S, E
        queries = queries.permute(1, 0, 2)
        queries_bg = queries_bg.permute(1, 0, 2)

        range_attention_maps_fg = self.dot_product_layer(x_fg, queries)  # .shape = n, n_query_channels, h, w
        range_attention_maps_bg = self.dot_product_layer(x_bg, queries_bg)

        y = self.regressor(regression_head)  # .shape = N, dim_out
        if self.adanum:
            return y, range_attention_maps_fg, range_attention_maps_bg
        else:
            if self.norm == 'linear':
                y = torch.relu(y)
                eps = 0.1
                y = y + eps
            elif self.norm == 'softmax':
                return torch.softmax(y, dim=1), range_attention_maps_fg, range_attention_maps_bg
            else:
                y = torch.sigmoid(y)
            y = y / y.sum(dim=1, keepdim=True)
        return y, range_attention_maps_fg, range_attention_maps_bg


class mViTGlobal(nn.Module):
    def __init__(self, in_channels, n_query_channels=128, patch_size=16, dim_out=256,
                 embedding_dim=128, num_heads=4, norm='linear', cfgs={}):
        super(mViTGlobal, self).__init__()
        self.norm = norm
        self.n_query_channels = n_query_channels
        self.patch_transformer = PatchTransformerEncoderGlobal(in_channels, patch_size, embedding_dim, num_heads, cfgs=cfgs)
        self.dot_product_layer = PixelWiseDotProduct()

        self.conv3x3 = nn.Conv2d(in_channels, embedding_dim, kernel_size=3, stride=1, padding=1)
        self.regressor = nn.Sequential(nn.Linear(embedding_dim, 256),
                                       nn.LeakyReLU(),
                                       nn.Linear(256, 256),
                                       nn.LeakyReLU(),
                                       nn.Linear(256, dim_out))

    def forward(self, x):
        # n, c, h, w = x.size()
        tgt = self.patch_transformer(x.copy())  # .shape = S, N, E

        x = self.conv3x3(x[0])

        regression_head, queries = tgt[0, ...], tgt[1:self.n_query_channels + 1, ...]

        # Change from S, N, E to N, S, E
        queries = queries.permute(1, 0, 2)
        range_attention_maps = self.dot_product_layer(x, queries)  # .shape = n, n_query_channels, h, w

        y = self.regressor(regression_head)  # .shape = N, dim_out
        if self.norm == 'linear':
            y = torch.relu(y)
            eps = 0.1
            y = y + eps
        elif self.norm == 'softmax':
            return torch.softmax(y, dim=1), range_attention_maps
        else:
            y = torch.sigmoid(y)
        y = y / y.sum(dim=1, keepdim=True)
        return y, range_attention_maps
