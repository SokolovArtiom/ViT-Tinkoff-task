import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import ndimage

from Transformer import TransformerModel
from ResNet import ResNetV2


class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_dim,
        patch_dim,
        out_dim,
        num_channels,
        embedding_dim,
        num_heads,
        num_layers,
        hidden_dim,
        dropout_rate,
        warmed_up,
        hybrid
    ):
        super(VisionTransformer, self).__init__()

        assert embedding_dim % num_heads == 0
        assert img_dim % patch_dim == 0

        self.hybrid = hybrid

        num_patches = int((img_dim // patch_dim) ** 2)

        if hybrid:
            patch_dim = img_dim//16//14
            self.hybrid_model = ResNetV2(block_units=(3,4,9), width_factor=1)
            num_channels = self.hybrid_model.width*16

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches+1, embedding_dim))

        self.pe_dropout = nn.Dropout(p=dropout_rate)

        self.transformer = TransformerModel(
            embedding_dim,
            num_layers,
            num_heads,
            hidden_dim,
            dropout_rate,
            warmed_up
        )

        self.pre_head_ln = nn.LayerNorm(embedding_dim)

        if not warmed_up:
            self.mlp_head = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, out_dim),
            )
        else:
            self.mlp_head = nn.Linear(embedding_dim, out_dim)

        self.conv_x = nn.Conv2d(
            num_channels,
            embedding_dim,
            kernel_size=patch_dim,
            stride=patch_dim,
        )

        self.to_cls_token = nn.Identity()

        if warmed_up:

            if hybrid:
                weights = np.load("imagenet21k_R50+ViT-B_16.npz")
            else:
                weights = np.load("imagenet21k_ViT-B_16.npz")
            
            with torch.no_grad():
                self.pre_head_ln.weight.copy_(torch.from_numpy(weights["Transformer/encoder_norm/scale"]))
                self.pre_head_ln.bias.copy_(torch.from_numpy(weights["Transformer/encoder_norm/bias"]))
                self.cls_token.copy_(torch.from_numpy(weights["cls"]))
                pe_weights = weights["embedding/kernel"]
                pe_weights = torch.from_numpy(pe_weights.transpose([3, 2, 0, 1]))
                self.conv_x.weight.copy_(pe_weights)
                self.conv_x.bias.copy_(torch.from_numpy(weights["embedding/bias"]))
                nn.init.zeros_(self.mlp_head.weight)
                nn.init.zeros_(self.mlp_head.bias)

                posemb = torch.from_numpy(weights["Transformer/posembed_input/pos_embedding"])
                posemb_new = self.position_embeddings
                if posemb.size() == posemb_new.size():
                    self.position_embeddings.copy_(posemb)
                else:
                    ntok_new = posemb_new.size(1)

                    posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
                    ntok_new -= 1

                    gs_old = int(np.sqrt(len(posemb_grid)))
                    gs_new = int(np.sqrt(ntok_new))
                    posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)

                    zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                    posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)
                    posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                    posemb = np.concatenate([posemb_tok, posemb_grid], axis=1)
                    self.position_embeddings.copy_(torch.from_numpy(posemb))

                for bname, block in self.transformer.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(weights, n_block=uname)

                if hybrid:
                    root_conv_weight = weights["conv_root/kernel"]
                    root_conv_weight = torch.from_numpy(root_conv_weight.transpose([3, 2, 0, 1]))
                    self.hybrid_model.root.conv.weight.copy_(root_conv_weight)
                    gn_weight = torch.from_numpy(weights["gn_root/scale"]).view(-1)
                    gn_bias = torch.from_numpy(weights["gn_root/bias"]).view(-1)
                    self.hybrid_model.root.gn.weight.copy_(gn_weight)
                    self.hybrid_model.root.gn.bias.copy_(gn_bias)

                    for bname, block in self.hybrid_model.body.named_children():
                        for uname, unit in block.named_children():
                            unit.load_from(weights, n_block=bname, n_unit=uname)

    def forward(self, x):
        
        if self.hybrid:
            x = self.hybrid_model(x)
        x = self.conv_x(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)

        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.position_embeddings
        x = self.pe_dropout(x)

        x = self.transformer(x)
        x = self.pre_head_ln(x)
        x = self.mlp_head(x[:, 0])

        return x
