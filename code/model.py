import os
import sys
import math
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import OneHotEncoder

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from config import args



import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SelectiveScan(nn.Module):

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(d_model * expand)


        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)


        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias=True
        )


        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)


        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(self.d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))


        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def selective_scan(self, x, dt, A, B, C, D):

        batch_size, seq_len, d_inner = x.shape
        n = A.shape[1]


        dt = dt.unsqueeze(-1)
        A = A.unsqueeze(0).unsqueeze(0)
        dA = torch.exp(dt * A)

        B = B.unsqueeze(2)
        dB = dt * B


        h = torch.zeros(batch_size, d_inner, n, device=x.device, dtype=x.dtype)
        ys = []

        for t in range(seq_len):
            h = dA[:, t] * h + dB[:, t] * x[:, t].unsqueeze(-1)
            y_t = (h * C[:, t].unsqueeze(1)).sum(dim=-1)
            ys.append(y_t)

        y = torch.stack(ys, dim=1)
        y = y + x * D.unsqueeze(0).unsqueeze(0)

        return y

    def forward(self, x):

        batch_size, seq_len, _ = x.shape


        xz = self.in_proj(x)
        x_part, z = xz.chunk(2, dim=-1)


        x_conv = x_part.transpose(1, 2)
        x_conv = self.conv1d(x_conv)[:, :, :seq_len]
        x_conv = x_conv.transpose(1, 2)
        x_conv = F.silu(x_conv)


        ssm_params = self.x_proj(x_conv)
        B_param = ssm_params[:, :, :self.d_state]
        C_param = ssm_params[:, :, self.d_state:2*self.d_state]
        dt_param = ssm_params[:, :, -1:]


        dt = F.softplus(self.dt_proj(dt_param))


        A = -torch.exp(self.A_log)


        y = self.selective_scan(x_conv, dt, A, B_param, C_param, self.D)


        y = y * F.silu(z)


        y = self.out_proj(y)
        y = self.dropout(y)

        return y


class MambaBlock(nn.Module):

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = SelectiveScan(d_model, d_state, d_conv, expand, dropout)

    def forward(self, x):

        residual = x
        x = self.norm(x)
        x = self.mamba(x)
        return x + residual


class GEGLU(nn.Module):

    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return x * F.gelu(gate)


class MambaWithFFN(nn.Module):

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.0, n_layers=2):
        super().__init__()
        n_layers = max(int(n_layers), 1)
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.ModuleDict({
                'mamba': MambaBlock(d_model, d_state, d_conv, expand, dropout),
                'ffn': nn.Sequential(
                    nn.LayerNorm(d_model),
                    nn.Linear(d_model, d_model * 2),
                    GEGLU(),
                    nn.Dropout(dropout),
                )
            }))

    def forward(self, x):

        for layer in self.layers:
            x = layer['mamba'](x)
            x = layer['ffn'](x)
        return x



import torch
import torch.nn as nn
import torch.nn.functional as F

def _drop_meta_paths(module, mp_embeds):
    drop_prob = float(getattr(module, 'meta_path_drop', 0.0))
    if (not module.training) or drop_prob <= 0.0 or len(mp_embeds) <= 1:
        return mp_embeds

    keep_prob = 1.0 - drop_prob
    device = mp_embeds[0].device
    keep_mask = torch.rand(len(mp_embeds), device=device) < keep_prob
    if not bool(keep_mask.any()):
        keep_mask[torch.randint(len(mp_embeds), (1,), device=device).item()] = True
    return [embed for embed, keep in zip(mp_embeds, keep_mask) if bool(keep)]


class SemanticAttention(nn.Module):

    def __init__(self, hidden_dim, attn_drop=0.0):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, hidden_dim, bias=True)
        nn.init.xavier_normal_(self.fc.weight, gain=1.414)
        self.tanh = nn.Tanh()
        self.att = nn.Parameter(torch.empty(hidden_dim, 1), requires_grad=True)
        nn.init.xavier_normal_(self.att.data, gain=1.414)
        self.attn_drop = nn.Dropout(attn_drop) if attn_drop > 0 else nn.Identity()

    def forward(self, embeds_list):

        if len(embeds_list) == 1:
            return embeds_list[0]

        stacked = torch.stack(embeds_list, dim=1)
        scores = torch.matmul(self.attn_drop(self.tanh(self.fc(stacked))), self.att).squeeze(-1)
        beta = torch.softmax(scores, dim=1).unsqueeze(-1)
        return (stacked * beta).sum(dim=1)


class GlobalSemanticAttention(nn.Module):

    def __init__(self, hidden_dim, attn_drop=0.0):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, hidden_dim, bias=True)
        nn.init.xavier_normal_(self.fc.weight, gain=1.414)
        self.tanh = nn.Tanh()
        self.att = nn.Parameter(torch.empty(hidden_dim, 1), requires_grad=True)
        nn.init.xavier_normal_(self.att.data, gain=1.414)
        self.attn_drop = nn.Dropout(attn_drop) if attn_drop > 0 else nn.Identity()

    def forward(self, embeds_list):

        if len(embeds_list) == 1:
            return embeds_list[0]

        stacked = torch.stack(embeds_list, dim=1)
        node_scores = torch.matmul(
            self.attn_drop(self.tanh(self.fc(stacked))), self.att
        ).squeeze(-1)
        path_scores = node_scores.mean(dim=0)
        beta = torch.softmax(path_scores, dim=0).view(1, -1, 1)
        return (stacked * beta).sum(dim=1)


class MultiResolutionFusion(nn.Module):

    def __init__(self, hidden_dim, num_hops, dropout=0.0):
        super().__init__()
        self.num_hops = num_hops
        self.depthwise = nn.Conv1d(
            hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim
        )
        self.pointwise = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        self.fuse = nn.Linear(hidden_dim * 3, hidden_dim)
        self.readout = nn.Linear(hidden_dim, 1)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        nn.init.xavier_normal_(self.pointwise.weight, gain=1.414)
        nn.init.xavier_normal_(self.fuse.weight, gain=1.414)
        nn.init.xavier_normal_(self.readout.weight, gain=1.414)

    def forward(self, hop_embeds):

        if isinstance(hop_embeds, (list, tuple)):
            x = torch.stack(hop_embeds, dim=1)
        else:
            x = hop_embeds

        local = self.depthwise(x.transpose(1, 2))
        local = F.gelu(self.pointwise(local)).transpose(1, 2)
        global_context = x.mean(dim=1, keepdim=True).expand_as(x)

        fused = self.fuse(torch.cat([x, local, global_context], dim=-1))
        fused = F.gelu(fused)
        fused = self.norm(x + self.dropout(fused))

        readout_score = self.readout(fused).squeeze(-1)
        readout_weight = torch.softmax(readout_score, dim=1).unsqueeze(-1)
        return (fused * readout_weight).sum(dim=1)


class SemanticMetaPathMamba(nn.Module):

    def __init__(self, num_metapaths, hidden_dim, num_hops=3,
                 d_state=16, d_conv=4, expand=2, dropout=0.0, n_mamba_layers=2,
                 meta_path_drop=0.0, semantic_fusion_mode='node'):
        super().__init__()
        self.num_metapaths = num_metapaths
        self.num_hops = num_hops
        self.hidden_dim = hidden_dim
        self.meta_path_drop = float(meta_path_drop)
        self.semantic_fusion_mode = (semantic_fusion_mode or 'node').lower()


        self.mpam_list = nn.ModuleList([
            MambaWithFFN(hidden_dim, d_state, d_conv, expand, dropout, n_mamba_layers)
            for _ in range(num_metapaths)
        ])


        self.mrf_list = nn.ModuleList([
            MultiResolutionFusion(hidden_dim, num_hops, dropout)
            for _ in range(num_metapaths)
        ])


        self.pos_encoding = nn.Parameter(torch.randn(1, num_hops, hidden_dim) * 0.02)


        if self.semantic_fusion_mode in ('node', 'nodewise'):
            self.semantic_attention = SemanticAttention(hidden_dim, dropout)
        elif self.semantic_fusion_mode in ('global', 'paper', 'original'):
            self.semantic_attention = GlobalSemanticAttention(hidden_dim, dropout)
        else:
            raise ValueError("semantic_fusion_mode must be one of: node, global, paper, original")

    def forward(self, multi_hop_features):

        mp_embeds = []
        for i in range(self.num_metapaths):
            x = multi_hop_features[i]

            x = x + self.pos_encoding[:, :x.size(1), :]

            u = self.mpam_list[i](x)

            z = self.mrf_list[i](u)
            mp_embeds.append(z)


        mp_embeds = _drop_meta_paths(self, mp_embeds)
        z_mp = self.semantic_attention(mp_embeds)
        return z_mp


class _TokenMLPBlock(nn.Module):

    def __init__(self, hidden_dim, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.in_proj = nn.Linear(hidden_dim, hidden_dim * 4)
        self.out_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):



        a, gate = self.in_proj(self.norm(x)).chunk(2, dim=-1)
        return x + self.dropout(self.out_proj(F.gelu(a) * gate))


class MatchedMLPMetaPathEncoder(nn.Module):

    def __init__(self, num_metapaths, hidden_dim, num_hops=3,
                 dropout=0.0, n_layers=1, meta_path_drop=0.0,
                 semantic_fusion_mode='node'):
        super().__init__()
        self.num_metapaths = num_metapaths
        self.num_hops = num_hops
        self.hidden_dim = hidden_dim
        self.meta_path_drop = float(meta_path_drop)
        self.pos_encoding = nn.Parameter(torch.randn(1, num_hops, hidden_dim) * 0.02)
        self.token_mlp_list = nn.ModuleList([
            nn.Sequential(*[_TokenMLPBlock(hidden_dim, dropout) for _ in range(n_layers)])
            for _ in range(num_metapaths)
        ])
        self.mrf_list = nn.ModuleList([
            MultiResolutionFusion(hidden_dim, num_hops, dropout)
            for _ in range(num_metapaths)
        ])
        if (semantic_fusion_mode or 'node').lower() in ('node', 'nodewise'):
            self.semantic_attention = SemanticAttention(hidden_dim, dropout)
        elif (semantic_fusion_mode or 'node').lower() in ('global', 'paper', 'original'):
            self.semantic_attention = GlobalSemanticAttention(hidden_dim, dropout)
        else:
            raise ValueError("semantic_fusion_mode must be one of: node, global, paper, original")

    def forward(self, multi_hop_features):
        mp_embeds = []
        for i in range(self.num_metapaths):
            x = multi_hop_features[i] + self.pos_encoding[:, :multi_hop_features[i].size(1), :]
            u = self.token_mlp_list[i](x)
            mp_embeds.append(self.mrf_list[i](u))
        return self.semantic_attention(_drop_meta_paths(self, mp_embeds))


class MeanMetaPathEncoder(nn.Module):

    def __init__(self, num_metapaths, hidden_dim, num_hops=3,
                 dropout=0.0, meta_path_drop=0.0, semantic_fusion_mode='node'):
        super().__init__()
        self.num_metapaths = num_metapaths
        self.num_hops = num_hops
        self.hidden_dim = hidden_dim
        self.meta_path_drop = float(meta_path_drop)
        self.semantic_fusion_mode = (semantic_fusion_mode or 'node').lower()

        if self.semantic_fusion_mode in ('node', 'nodewise'):
            self.semantic_attention = SemanticAttention(hidden_dim, dropout)
        elif self.semantic_fusion_mode in ('global', 'paper', 'original'):
            self.semantic_attention = GlobalSemanticAttention(hidden_dim, dropout)
        else:
            raise ValueError("semantic_fusion_mode must be one of: node, global, paper, original")

    def forward(self, multi_hop_features):


        mp_embeds = [
            multi_hop_features[i].mean(dim=1)
            for i in range(self.num_metapaths)
        ]
        mp_embeds = _drop_meta_paths(self, mp_embeds)
        return self.semantic_attention(mp_embeds)


class MLPMetaPathEncoder(nn.Module):

    def __init__(self, num_metapaths, hidden_dim, num_hops=3,
                 dropout=0.0, meta_path_drop=0.0):
        super().__init__()
        self.num_metapaths = num_metapaths
        self.num_hops = num_hops
        self.hidden_dim = hidden_dim
        self.meta_path_drop = float(meta_path_drop)
        bottleneck_dim = max(8, hidden_dim // 4)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(bottleneck_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, multi_hop_features):

        mp_embeds = [
            self.norm(self.mlp(multi_hop_features[i].mean(dim=1)))
            for i in range(self.num_metapaths)
        ]
        mp_embeds = _drop_meta_paths(self, mp_embeds)
        return torch.stack(mp_embeds, dim=0).mean(dim=0)


class StrictMLPMetaPathEncoder(nn.Module):

    def __init__(self, num_metapaths, hidden_dim, num_hops=3,
                 dropout=0.0, meta_path_drop=0.0):
        super().__init__()
        self.num_metapaths = num_metapaths
        self.num_hops = num_hops
        self.hidden_dim = hidden_dim
        self.meta_path_drop = float(meta_path_drop)
        bottleneck_dim = max(4, hidden_dim // 8)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(bottleneck_dim, hidden_dim),
        )

    def forward(self, multi_hop_features):
        mp_embeds = [
            self.mlp(multi_hop_features[i].mean(dim=1))
            for i in range(self.num_metapaths)
        ]
        mp_embeds = _drop_meta_paths(self, mp_embeds)
        return torch.stack(mp_embeds, dim=0).mean(dim=0)


class UltraStrictMLPMetaPathEncoder(nn.Module):

    def __init__(self, num_metapaths, hidden_dim, num_hops=3,
                 dropout=0.0, meta_path_drop=0.0):
        super().__init__()
        self.num_metapaths = num_metapaths
        self.num_hops = num_hops
        self.hidden_dim = hidden_dim
        self.meta_path_drop = float(meta_path_drop)
        bottleneck_dim = max(2, hidden_dim // 16)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(bottleneck_dim, hidden_dim),
        )

    def forward(self, multi_hop_features):
        mp_embeds = [
            self.mlp(multi_hop_features[i].mean(dim=1))
            for i in range(self.num_metapaths)
        ]
        mp_embeds = _drop_meta_paths(self, mp_embeds)
        return torch.stack(mp_embeds, dim=0).mean(dim=0)


class SimpleMetaPathMamba(nn.Module):

    def __init__(self, num_metapaths, hidden_dim, num_hops=3,
                 d_state=16, d_conv=4, expand=2, dropout=0.0, n_mamba_layers=1,
                 hop_pool='mean', meta_pool='mean', meta_path_drop=0.0):
        super().__init__()
        if hop_pool not in ('mean', 'last'):
            raise ValueError("hop_pool must be 'mean' or 'last'")
        if meta_pool != 'mean':
            raise ValueError("SimpleMetaPathMamba currently supports meta_pool='mean' only")
        self.num_metapaths = num_metapaths
        self.num_hops = num_hops
        self.hidden_dim = hidden_dim
        self.meta_path_drop = float(meta_path_drop)
        self.hop_pool = hop_pool
        self.meta_pool = meta_pool
        self.mpam_list = nn.ModuleList([
            MambaWithFFN(hidden_dim, d_state, d_conv, expand, dropout, n_mamba_layers)
            for _ in range(num_metapaths)
        ])
        self.pos_encoding = nn.Parameter(torch.randn(1, num_hops, hidden_dim) * 0.02)
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(self, multi_hop_features):
        mp_embeds = []
        for i in range(self.num_metapaths):
            x = multi_hop_features[i]
            x = x + self.pos_encoding[:, :x.size(1), :]
            u = self.mpam_list[i](x)
            if self.hop_pool == 'last':
                z = u[:, -1, :]
            else:
                z = u.mean(dim=1)
            mp_embeds.append(z)
        mp_embeds = _drop_meta_paths(self, mp_embeds)
        z_mp = torch.stack(mp_embeds, dim=0).mean(dim=0)
        return self.out_norm(z_mp)


class LightweightMetaPathMamba(nn.Module):

    def __init__(self, num_metapaths, hidden_dim, num_hops=3,
                 d_state=16, d_conv=4, expand=2, dropout=0.0, n_mamba_layers=1,
                 use_hop_gate=True, use_meta_attn=True,
                 meta_attn_temperature=2.0, meta_attn_strength=0.5,
                 meta_path_drop=0.0):
        super().__init__()
        self.num_metapaths = num_metapaths
        self.num_hops = num_hops
        self.hidden_dim = hidden_dim
        self.use_hop_gate = use_hop_gate
        self.use_meta_attn = use_meta_attn
        self.meta_attn_temperature = float(meta_attn_temperature)
        self.meta_attn_strength = float(meta_attn_strength)
        self.meta_path_drop = float(meta_path_drop)

        self.mpam_list = nn.ModuleList([
            MambaWithFFN(hidden_dim, d_state, d_conv, expand, dropout, n_mamba_layers)
            for _ in range(num_metapaths)
        ])
        self.pos_encoding = nn.Parameter(torch.randn(1, num_hops, hidden_dim) * 0.02)

        if self.use_hop_gate:
            self.hop_logits = nn.Parameter(torch.zeros(num_hops))
        if self.use_meta_attn:
            self.meta_query = nn.Parameter(torch.empty(hidden_dim))
            nn.init.normal_(self.meta_query, mean=0.0, std=hidden_dim ** -0.5)
            self.meta_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.out_norm = nn.LayerNorm(hidden_dim)

    def _pool_hops(self, u):
        if not self.use_hop_gate:
            return u.mean(dim=1)
        hop_weights = torch.softmax(self.hop_logits[:u.size(1)], dim=0)
        return (u * hop_weights.view(1, -1, 1)).sum(dim=1)

    def _pool_metapaths(self, mp_embeds):
        stacked = torch.stack(mp_embeds, dim=1)
        mean_z = stacked.mean(dim=1)
        if not self.use_meta_attn or stacked.size(1) == 1:
            return mean_z

        query = self.meta_dropout(self.meta_query)
        scores = torch.matmul(stacked, query) / max(self.meta_attn_temperature, 1e-6)
        beta = torch.softmax(scores, dim=1).unsqueeze(-1)
        attn_z = (stacked * beta).sum(dim=1)
        return (1.0 - self.meta_attn_strength) * mean_z + self.meta_attn_strength * attn_z

    def forward(self, multi_hop_features):
        mp_embeds = []
        for i in range(self.num_metapaths):
            x = multi_hop_features[i]
            x = x + self.pos_encoding[:, :x.size(1), :]
            u = self.mpam_list[i](x)
            mp_embeds.append(self._pool_hops(u))
        mp_embeds = _drop_meta_paths(self, mp_embeds)
        z_mp = self._pool_metapaths(mp_embeds)
        return self.out_norm(z_mp)



import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class RelationIntraAttention(nn.Module):

    def __init__(self, hidden_dim, attn_drop,
                 use_log_degree_penalty=False, degree_penalty_bias=math.e):
        super().__init__()
        self.target_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.nei_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.att = nn.Parameter(torch.empty(size=(hidden_dim, 1)), requires_grad=True)
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.attn_drop = nn.Dropout(attn_drop) if attn_drop > 0 else nn.Identity()
        self.use_log_degree_penalty = use_log_degree_penalty
        self.degree_penalty_bias = degree_penalty_bias

        nn.init.xavier_normal_(self.target_proj.weight, gain=1.414)
        nn.init.xavier_normal_(self.nei_proj.weight, gain=1.414)
        nn.init.xavier_normal_(self.att.data, gain=1.414)

    def forward(self, target_feat, nei_feat, nei_weight=None, nei_degree=None):

        if nei_feat.size(0) == 0:
            return target_feat

        q = self.target_proj(target_feat).unsqueeze(0).expand_as(nei_feat)
        k = self.nei_proj(nei_feat)
        score = self.leakyrelu(q + k).matmul(self.att).squeeze(-1)

        if self.use_log_degree_penalty and nei_degree is not None:
            penalty = torch.log(nei_degree.to(score.dtype) + self.degree_penalty_bias)
            score = score / (penalty + 1e-12)

        score = score - score.max()
        alpha = torch.softmax(score, dim=0)

        if nei_weight is not None:
            alpha = alpha * nei_weight.to(alpha.dtype)
            alpha = alpha / (alpha.sum() + 1e-12)

        alpha = self.attn_drop(alpha)
        out = (alpha.unsqueeze(-1) * nei_feat).sum(dim=0)
        return out


class RelationViewGAT(nn.Module):

    def __init__(self, hidden_dim, attn_drop):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.att = nn.Parameter(torch.empty(size=(2 * hidden_dim, 1)), requires_grad=True)
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.attn_drop = nn.Dropout(attn_drop) if attn_drop > 0 else nn.Identity()

        nn.init.xavier_normal_(self.proj.weight, gain=1.414)
        nn.init.xavier_normal_(self.att.data, gain=1.414)

    def forward(self, embeds):

        tokens = torch.stack(embeds, dim=1)
        h = self.proj(tokens)
        n_nodes, n_tokens, hidden_dim = h.shape

        h_i = h.unsqueeze(2).expand(n_nodes, n_tokens, n_tokens, hidden_dim)
        h_j = h.unsqueeze(1).expand(n_nodes, n_tokens, n_tokens, hidden_dim)
        pair = torch.cat([h_i, h_j], dim=-1)
        score = self.leakyrelu(torch.matmul(pair, self.att).squeeze(-1))
        score = score - score.max(dim=-1, keepdim=True).values
        alpha = torch.softmax(score, dim=-1)
        alpha = self.attn_drop(alpha)
        out = torch.matmul(alpha, h)
        return out.mean(dim=1)


class RelationTypeAttention(nn.Module):

    def __init__(self, hidden_dim, attn_drop):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.context = nn.Parameter(torch.empty(size=(1, hidden_dim)), requires_grad=True)
        self.tanh = nn.Tanh()
        self.attn_drop = nn.Dropout(attn_drop) if attn_drop > 0 else nn.Identity()

        nn.init.xavier_normal_(self.fc.weight, gain=1.414)
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)
        nn.init.xavier_normal_(self.context.data, gain=1.414)

    def forward(self, embeds):
        tokens = torch.stack(embeds, dim=1)
        score = self.tanh(self.fc(tokens)).matmul(self.context.t()).squeeze(-1)
        score = score - score.max(dim=1, keepdim=True).values
        beta = torch.softmax(score, dim=1)
        beta = self.attn_drop(beta)
        return (beta.unsqueeze(-1) * tokens).sum(dim=1)


class LoEncoder(nn.Module):

    def __init__(self, hidden_dim, sample_rate, nei_num, attn_drop,
                 use_log_degree_penalty=False, degree_penalty_bias=math.e,
                 second_layer='type_attn', output_drop=0.0, self_weight=1.0,
                 bottleneck_dim=0, relation_drop=0.0):
        super().__init__()
        self.intra = nn.ModuleList([
            RelationIntraAttention(
                hidden_dim,
                attn_drop,
                use_log_degree_penalty=use_log_degree_penalty,
                degree_penalty_bias=degree_penalty_bias,
            )
            for _ in range(nei_num)
        ])
        if second_layer == 'relation_gat':
            self.relation_fusion = RelationViewGAT(hidden_dim, attn_drop)
        elif second_layer == 'type_attn':
            self.relation_fusion = RelationTypeAttention(hidden_dim, attn_drop)
        else:
            raise ValueError(f"Unsupported short second layer: {second_layer}")
        self.sample_rate = sample_rate
        self.nei_num = nei_num
        self.output_drop = nn.Dropout(output_drop) if output_drop > 0 else nn.Identity()
        self.self_weight = float(self_weight)
        self.relation_drop = float(relation_drop)
        bottleneck_dim = int(bottleneck_dim or 0)
        if 0 < bottleneck_dim < hidden_dim:
            self.output_bottleneck = nn.Sequential(
                nn.Linear(hidden_dim, bottleneck_dim),
                nn.ReLU(),
                nn.Linear(bottleneck_dim, hidden_dim),
            )
        else:
            self.output_bottleneck = nn.Identity()
        self.output_norm = nn.LayerNorm(hidden_dim) if 0 < bottleneck_dim < hidden_dim else nn.Identity()

    def _drop_relation_tokens(self, relation_embeds):

        if (not self.training) or self.relation_drop <= 0 or len(relation_embeds) <= 1:
            return relation_embeds

        kept = []
        for rel_embed in relation_embeds:
            if torch.rand((), device=rel_embed.device) >= self.relation_drop:
                kept.append(rel_embed)
        if not kept:
            keep_idx = int(torch.randint(len(relation_embeds), (1,), device=relation_embeds[0].device).item())
            kept = [relation_embeds[keep_idx]]
        scale = float(len(relation_embeds)) / float(len(kept))
        return [rel_embed * scale for rel_embed in kept]

    def _sample_neighbors(self, per_node_nei, sample_cap, nei_degree=None):
        if len(per_node_nei) == 0:
            return np.array([], dtype=np.int64)
        per_node_nei = np.asarray(per_node_nei, dtype=np.int64)
        if len(per_node_nei) > sample_cap:
            if nei_degree is not None:


                degree = nei_degree[torch.as_tensor(per_node_nei, dtype=torch.long, device=nei_degree.device)]
                order = np.argsort(degree.detach().cpu().numpy(), kind='mergesort')
                return per_node_nei[order[:sample_cap]]

            return per_node_nei[:sample_cap]
        return per_node_nei

    def forward(self, nei_h, nei_index, nei_weights=None, nei_degrees=None):
        device = nei_h[0].device
        relation_embeds = []
        target_feat = nei_h[0]

        for rel_id in range(self.nei_num):
            sample_cap = self.sample_rate[rel_id]
            one_type_weight = None if nei_weights is None else nei_weights[rel_id]
            one_type_degree = None if nei_degrees is None else nei_degrees[rel_id]
            h_nei = nei_h[rel_id + 1]

            outputs = []
            for node_id, per_node_nei in enumerate(nei_index[rel_id]):
                sampled = self._sample_neighbors(per_node_nei, sample_cap, one_type_degree)
                if sampled.size == 0:
                    outputs.append(target_feat[node_id])
                    continue

                idx = torch.as_tensor(sampled, dtype=torch.long, device=device)
                nei_feat = h_nei[idx]
                rel_weight = None if one_type_weight is None else one_type_weight[idx]
                rel_degree = None if one_type_degree is None else one_type_degree[idx]
                outputs.append(
                    F.elu(self.intra[rel_id](target_feat[node_id], nei_feat, rel_weight, rel_degree))
                )

            relation_embeds.append(torch.stack(outputs, dim=0))

        relation_embeds = self._drop_relation_tokens(relation_embeds)
        self_token = target_feat * self.self_weight
        fused = self.relation_fusion([self_token] + relation_embeds)
        return self.output_drop(self.output_norm(self.output_bottleneck(fused)))

import torch
import torch.nn as nn
import torch.nn.functional as F


class Contrast(nn.Module):
    def __init__(self, hidden_dim, tau, lam,
                 neg_mode='none', meow_floor=0.2, meow_power=1.0,
                 align_mode='symmetric'):
        super(Contrast, self).__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.tau = tau
        self.lam = lam
        self.neg_mode = neg_mode
        self.meow_floor = meow_floor
        self.meow_power = meow_power
        self.align_mode = align_mode
        for model in self.proj:
            if isinstance(model, nn.Linear):
                nn.init.xavier_normal_(model.weight, gain=1.414)

    def sim(self, z1, z2):
        z1_norm = torch.norm(z1, dim=-1, keepdim=True)
        z2_norm = torch.norm(z2, dim=-1, keepdim=True)
        dot_numerator = torch.mm(z1, z2.t())
        dot_denominator = torch.mm(z1_norm, z2_norm.t())
        sim_matrix = torch.exp(dot_numerator / dot_denominator / self.tau)
        return sim_matrix

    def _meow_inspired_neg_weight(self, z1, z2, pos_dense):

        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        sim11 = torch.mm(z1, z1.t()).clamp(min=0.0, max=1.0)
        sim22 = torch.mm(z2, z2.t()).clamp(min=0.0, max=1.0)
        semantic_sim = 0.5 * (sim11 + sim22)

        weights = 1.0 - semantic_sim
        weights = torch.clamp(weights, min=0.0, max=1.0)
        if self.meow_power != 1.0:
            weights = torch.pow(weights, self.meow_power)
        weights = self.meow_floor + (1.0 - self.meow_floor) * weights
        weights = weights * (1.0 - pos_dense) + pos_dense
        return weights

    def forward(self, z_mp, z_sc, pos):
        if self.align_mode == 'teacher_long':
            z_mp = z_mp.detach()
        elif self.align_mode == 'teacher_short':
            z_sc = z_sc.detach()
        z_proj_mp = self.proj(z_mp)
        z_proj_sc = self.proj(z_sc)
        matrix_mp2sc = self.sim(z_proj_mp, z_proj_sc)
        matrix_sc2mp = matrix_mp2sc.t()

        pos_dense = pos.to_dense()
        if self.neg_mode == 'meow_weighted':
            neg_weights = self._meow_inspired_neg_weight(z_proj_mp, z_proj_sc, pos_dense)
            matrix_mp2sc = matrix_mp2sc * neg_weights
            matrix_sc2mp = matrix_sc2mp * neg_weights.t()

        matrix_mp2sc = matrix_mp2sc/(torch.sum(matrix_mp2sc, dim=1).view(-1, 1) + 1e-8)
        lori_mp = -torch.log(matrix_mp2sc.mul(pos_dense).sum(dim=-1)).mean()

        matrix_sc2mp = matrix_sc2mp / (torch.sum(matrix_sc2mp, dim=1).view(-1, 1) + 1e-8)
        lori_sc = -torch.log(matrix_sc2mp.mul(pos_dense.t()).sum(dim=-1)).mean()
        return self.lam * lori_mp + (1 - self.lam) * lori_sc



import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class DHMMDFIEncoderCore(nn.Module):
    def __init__(self, hidden_dim, graph_feat_dim_list, feat_drop,
                 P_drug, P_fc, num_hops, long_hidden_dim=None, d_state=16, d_conv=4, expand=2,
                 P_food=None,
                 mamba_dropout=0.1, n_mamba_layers=2,
                 long_encoder_mode='simple',
                 sample_rate_drug=None, sample_rate_fc=None, sample_rate_food=None,
                 nei_num_drug=2, nei_num_fc=2, nei_num_food=2, attn_drop=0.3,
                  tau=0.8, lam=0.5, fusion_mode='gated', ablation=None,
                  short_use_log_degree_penalty=False,
                  short_degree_penalty_bias=2.718281828,
                  short_second_layer='type_attn',
                  short_output_drop=0.0,
                  short_self_weight=1.0,
                  short_bottleneck_dim=0,
                  short_relation_drop=0.0,
                  long_output_drop=0.0,
                  long_meta_path_drop=0.0,
                  view_norm=True,
                  cross_view_residual=True,
                  cross_view_residual_scale=0.5,
                  cl_align_mode='teacher_short',
                  view_decorrelation_weight=0.0,
                  gate_init_long_weight=0.25):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.long_hidden_dim = int(long_hidden_dim or hidden_dim)
        self.ablation = ablation or 'none'
        self.fusion_mode = fusion_mode
        self.gate_init_long_weight = float(gate_init_long_weight)
        self.view_decorrelation_weight = float(view_decorrelation_weight)
        self.long_output_drop = nn.Dropout(long_output_drop) if long_output_drop > 0 else nn.Identity()
        self.long_meta_path_drop = float(long_meta_path_drop)
        self.view_norm_enabled = bool(view_norm)
        self.cross_view_residual_enabled = bool(cross_view_residual)
        self.cross_view_residual_scale = float(cross_view_residual_scale)
        if long_encoder_mode not in ('mean', 'mlp', 'mlp_matched', 'mlp_strict', 'mlp_ultra', 'simple', 'light_hop', 'light_meta', 'light_both', 'semantic'):
            raise ValueError("long_encoder_mode must be one of: mean, mlp, mlp_matched, mlp_strict, mlp_ultra, simple, light_hop, light_meta, light_both, semantic")

        self.graph_fc_list = nn.ModuleList([
            nn.Linear(feats_dim, hidden_dim, bias=True)
            for feats_dim in graph_feat_dim_list
        ])
        for fc in self.graph_fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)
            if fc.bias is not None:
                nn.init.zeros_(fc.bias)

        self.feat_drop = nn.Dropout(feat_drop) if feat_drop > 0 else nn.Identity()

        def make_long_bottleneck():
            if self.long_hidden_dim >= hidden_dim:
                return nn.Identity()
            return nn.Sequential(
                nn.Linear(hidden_dim, self.long_hidden_dim),
                nn.ReLU(),
                nn.Linear(self.long_hidden_dim, hidden_dim),
            )

        self.long_bottleneck_drug = make_long_bottleneck()
        self.long_bottleneck_fc = make_long_bottleneck()

        def make_long_encoder(num_metapaths):
            if long_encoder_mode == 'mean':
                return MeanMetaPathEncoder(
                    num_metapaths=num_metapaths, hidden_dim=hidden_dim, num_hops=num_hops,
                    dropout=mamba_dropout, meta_path_drop=self.long_meta_path_drop
                )
            if long_encoder_mode == 'mlp':
                return MLPMetaPathEncoder(
                    num_metapaths=num_metapaths, hidden_dim=hidden_dim, num_hops=num_hops,
                    dropout=mamba_dropout, meta_path_drop=self.long_meta_path_drop
                )
            if long_encoder_mode == 'mlp_matched':
                return MatchedMLPMetaPathEncoder(
                    num_metapaths=num_metapaths, hidden_dim=hidden_dim, num_hops=num_hops,
                    dropout=mamba_dropout, n_layers=n_mamba_layers,
                    meta_path_drop=self.long_meta_path_drop
                )
            if long_encoder_mode == 'mlp_strict':
                return StrictMLPMetaPathEncoder(
                    num_metapaths=num_metapaths, hidden_dim=hidden_dim, num_hops=num_hops,
                    dropout=mamba_dropout, meta_path_drop=self.long_meta_path_drop
                )
            if long_encoder_mode == 'mlp_ultra':
                return UltraStrictMLPMetaPathEncoder(
                    num_metapaths=num_metapaths, hidden_dim=hidden_dim, num_hops=num_hops,
                    dropout=mamba_dropout, meta_path_drop=self.long_meta_path_drop
                )
            if long_encoder_mode == 'semantic':
                return SemanticMetaPathMamba(
                    num_metapaths=num_metapaths, hidden_dim=hidden_dim, num_hops=num_hops,
                    d_state=d_state, d_conv=d_conv, expand=expand,
                    dropout=mamba_dropout, n_mamba_layers=n_mamba_layers,
                    meta_path_drop=self.long_meta_path_drop
                )
            if long_encoder_mode == 'simple':
                return SimpleMetaPathMamba(
                    num_metapaths=num_metapaths, hidden_dim=hidden_dim, num_hops=num_hops,
                    d_state=d_state, d_conv=d_conv, expand=expand,
                    dropout=mamba_dropout, n_mamba_layers=n_mamba_layers,
                    meta_path_drop=self.long_meta_path_drop
                )
            return LightweightMetaPathMamba(
                num_metapaths=num_metapaths, hidden_dim=hidden_dim, num_hops=num_hops,
                d_state=d_state, d_conv=d_conv, expand=expand,
                dropout=mamba_dropout, n_mamba_layers=n_mamba_layers,
                use_hop_gate=long_encoder_mode in ('light_hop', 'light_both'),
                use_meta_attn=long_encoder_mode in ('light_meta', 'light_both'),
                meta_attn_temperature=2.0,
                meta_attn_strength=0.5,
                meta_path_drop=self.long_meta_path_drop,
            )

        self.mamba_drug = make_long_encoder(P_drug)
        self.lo_drug = LoEncoder(
            hidden_dim=hidden_dim,
            sample_rate=sample_rate_drug or [5, 5],
            nei_num=nei_num_drug,
            attn_drop=attn_drop,
            use_log_degree_penalty=short_use_log_degree_penalty,
            degree_penalty_bias=short_degree_penalty_bias,
            second_layer=short_second_layer,
            output_drop=short_output_drop,
            self_weight=short_self_weight,
            bottleneck_dim=short_bottleneck_dim,
            relation_drop=short_relation_drop,
        )
        self.contrast_drug = Contrast(hidden_dim, tau, lam, align_mode=cl_align_mode)

        self.mamba_fc = make_long_encoder(P_fc)
        self.lo_fc = LoEncoder(
            hidden_dim=hidden_dim,
            sample_rate=sample_rate_fc or [5, 5],
            nei_num=nei_num_fc,
            attn_drop=attn_drop,
            use_log_degree_penalty=short_use_log_degree_penalty,
            degree_penalty_bias=short_degree_penalty_bias,
            second_layer=short_second_layer,
            output_drop=short_output_drop,
            self_weight=short_self_weight,
            bottleneck_dim=short_bottleneck_dim,
            relation_drop=short_relation_drop,
        )
        self.contrast_fc = Contrast(hidden_dim, tau, lam, align_mode=cl_align_mode)

        self.drug_gate = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Sigmoid())
        self.fc_gate = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Sigmoid())
        self.drug_long_norm = nn.LayerNorm(hidden_dim)
        self.drug_short_norm = nn.LayerNorm(hidden_dim)
        self.fc_long_norm = nn.LayerNorm(hidden_dim)
        self.fc_short_norm = nn.LayerNorm(hidden_dim)
        self.drug_view_residual = self._make_view_residual(hidden_dim)
        self.fc_view_residual = self._make_view_residual(hidden_dim)
        self.drug_view_synergy = self._make_view_synergy(hidden_dim)
        self.fc_view_synergy = self._make_view_synergy(hidden_dim)
        self.drug_view_calibrator = self._make_view_calibrator(hidden_dim)
        self.fc_view_calibrator = self._make_view_calibrator(hidden_dim)
        self.has_food_branch = P_food is not None
        if self.has_food_branch:
            self.long_bottleneck_food = make_long_bottleneck()
            self.mamba_food = make_long_encoder(P_food)
            self.lo_food = LoEncoder(
                hidden_dim=hidden_dim,
                sample_rate=sample_rate_food or [5, 5],
                nei_num=nei_num_food,
                attn_drop=attn_drop,
                use_log_degree_penalty=short_use_log_degree_penalty,
                degree_penalty_bias=short_degree_penalty_bias,
                second_layer=short_second_layer,
                output_drop=short_output_drop,
                self_weight=short_self_weight,
                bottleneck_dim=short_bottleneck_dim,
                relation_drop=short_relation_drop,
            )
            self.contrast_food = Contrast(hidden_dim, tau, lam, align_mode=cl_align_mode)
            self.food_gate = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Sigmoid())
            self.food_long_norm = nn.LayerNorm(hidden_dim)
            self.food_short_norm = nn.LayerNorm(hidden_dim)
            self.food_view_residual = self._make_view_residual(hidden_dim)
            self.food_view_synergy = self._make_view_synergy(hidden_dim)
            self.food_view_calibrator = self._make_view_calibrator(hidden_dim)

        gate_layers = [self.drug_gate[0], self.fc_gate[0]]
        if self.has_food_branch:
            gate_layers.append(self.food_gate[0])
        for gate_layer in gate_layers:
            nn.init.zeros_(gate_layer.weight)
            if gate_layer.bias is not None:
                init_weight = min(max(self.gate_init_long_weight, 1e-4), 1.0 - 1e-4)
                nn.init.constant_(gate_layer.bias, math.log(init_weight / (1.0 - init_weight)))

        for module in [self.drug_view_residual, self.fc_view_residual] + (
            [self.food_view_residual] if self.has_food_branch else []
        ):
            final_layer = module[-1]
            nn.init.zeros_(final_layer.weight)
            if final_layer.bias is not None:
                nn.init.zeros_(final_layer.bias)

        for module in [self.drug_view_synergy, self.fc_view_synergy] + (
            [self.food_view_synergy] if self.has_food_branch else []
        ):
            for layer in module.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_normal_(layer.weight, gain=1.414)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

        for module in [self.drug_view_calibrator, self.fc_view_calibrator] + (
            [self.food_view_calibrator] if self.has_food_branch else []
        ):
            for layer in module.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_normal_(layer.weight, gain=1.414)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
            final_layer = module[-1]
            nn.init.zeros_(final_layer.weight)
            if final_layer.bias is not None:
                nn.init.zeros_(final_layer.bias)

        bottleneck_modules = [self.long_bottleneck_drug, self.long_bottleneck_fc]
        if self.has_food_branch:
            bottleneck_modules.append(self.long_bottleneck_food)
        for module in bottleneck_modules:
            for layer in module.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_normal_(layer.weight, gain=1.414)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def project_features(self, graph_feats, smiles_feats=None):
        return [
            F.elu(self.feat_drop(self.graph_fc_list[i](graph_feats[i])))
            for i in range(len(graph_feats))
        ]

    def _long_bottleneck(self, z_mamba, side):
        if side == 'drug':
            return self.long_output_drop(self.long_bottleneck_drug(z_mamba))
        if side == 'fc':
            return self.long_output_drop(self.long_bottleneck_fc(z_mamba))
        if side == 'food':
            return self.long_output_drop(self.long_bottleneck_food(z_mamba))
        raise ValueError("side must be 'drug', 'food', or 'fc'")

    def forward(self, graph_feats, smiles_feats, pos_drug, pos_fc,
                multi_hop_drug, multi_hop_fc, nei_index_drug, nei_index_fc):
        h_all = self.project_features(graph_feats, smiles_feats)
        h_drug, h_food, h_fc = h_all[0], h_all[1], h_all[2]

        z_mamba_drug = self._long_bottleneck(self.mamba_drug(multi_hop_drug), 'drug')
        z_lo_drug = self.lo_drug([h_drug, h_food, h_fc], nei_index_drug)
        loss_drug = self.contrast_drug(z_mamba_drug, z_lo_drug, pos_drug)

        z_mamba_fc = self._long_bottleneck(self.mamba_fc(multi_hop_fc), 'fc')
        z_lo_fc = self.lo_fc([h_fc, h_drug, h_food], nei_index_fc)
        loss_fc = self.contrast_fc(z_mamba_fc, z_lo_fc, pos_fc)
        return loss_drug + loss_fc

    def _combine(self, z_mamba, z_lo, side=None):
        if self.view_norm_enabled:
            z_mamba, z_lo = self._normalize_views(z_mamba, z_lo, side)
        if self.ablation == 'wo_long':
            return z_lo
        if self.ablation == 'wo_short':
            return z_mamba
        if self.cross_view_residual_enabled:
            z_mamba, z_lo = self._apply_cross_view_residual(z_mamba, z_lo, side)
        if self.fusion_mode in ('concat', 'pair_cross'):
            return torch.cat([z_mamba, z_lo], dim=-1)
        if self.fusion_mode == 'concat_calibrated':
            z_mamba, z_lo = self._apply_view_calibrator(z_mamba, z_lo, side)
            return torch.cat([z_mamba, z_lo], dim=-1)
        if self.fusion_mode == 'concat_synergy':
            z_syn = self._apply_view_synergy(z_mamba, z_lo, side)
            return torch.cat([z_mamba, z_lo, z_syn], dim=-1)
        if self.fusion_mode == 'add':
            return z_mamba + z_lo
        if self.fusion_mode == 'mean':
            return 0.5 * (z_mamba + z_lo)
        if self.fusion_mode != 'gated':
            raise ValueError(f"Unsupported fusion_mode: {self.fusion_mode}")
        if side == 'drug':
            g = self.drug_gate(torch.cat([z_mamba, z_lo], dim=-1))
        elif side == 'fc':
            g = self.fc_gate(torch.cat([z_mamba, z_lo], dim=-1))
        elif side == 'food':
            g = self.food_gate(torch.cat([z_mamba, z_lo], dim=-1))
        else:
            raise ValueError("side must be 'drug', 'food', or 'fc'")
        return g * z_mamba + (1.0 - g) * z_lo

    @staticmethod
    def _make_view_residual(hidden_dim):
        return nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    @staticmethod
    def _make_view_synergy(hidden_dim):
        return nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    @staticmethod
    def _make_view_calibrator(hidden_dim):
        return nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
        )

    def _apply_view_synergy(self, z_mamba, z_lo, side):
        if side == 'drug':
            synergy = self.drug_view_synergy
        elif side == 'fc':
            synergy = self.fc_view_synergy
        elif side == 'food':
            synergy = self.food_view_synergy
        else:
            raise ValueError("side must be 'drug', 'food', or 'fc'")
        return synergy(torch.cat([z_mamba, z_lo, z_mamba * z_lo, torch.abs(z_mamba - z_lo)], dim=-1))

    def _apply_view_calibrator(self, z_mamba, z_lo, side):
        if side == 'drug':
            calibrator = self.drug_view_calibrator
        elif side == 'fc':
            calibrator = self.fc_view_calibrator
        elif side == 'food':
            calibrator = self.food_view_calibrator
        else:
            raise ValueError("side must be 'drug', 'food', or 'fc'")
        delta_long, delta_short = calibrator(
            torch.cat([z_mamba, z_lo, z_mamba * z_lo, torch.abs(z_mamba - z_lo)], dim=-1)
        ).chunk(2, dim=-1)
        return z_mamba + delta_long, z_lo + delta_short

    def _apply_cross_view_residual(self, z_mamba, z_lo, side):
        if side == 'drug':
            residual = self.drug_view_residual
        elif side == 'fc':
            residual = self.fc_view_residual
        elif side == 'food':
            residual = self.food_view_residual
        else:
            raise ValueError("side must be 'drug', 'food', or 'fc'")
        delta = residual(torch.cat([z_mamba, z_lo], dim=-1))


        scale = self.cross_view_residual_scale
        return z_mamba + scale * delta, z_lo + scale * delta

    def _normalize_views(self, z_mamba, z_lo, side):
        if side == 'drug':
            long_norm = self.drug_long_norm
            short_norm = self.drug_short_norm
        elif side == 'fc':
            long_norm = self.fc_long_norm
            short_norm = self.fc_short_norm
        elif side == 'food':
            long_norm = self.food_long_norm
            short_norm = self.food_short_norm
        else:
            raise ValueError("side must be 'drug', 'food', or 'fc'")
        if z_mamba is not None:
            z_mamba = long_norm(z_mamba)
        if z_lo is not None:
            z_lo = short_norm(z_lo)
        return z_mamba, z_lo

    def _view_decorrelation_loss(self, z_long, z_short):
        if self.view_decorrelation_weight <= 0:
            return z_long.new_tensor(0.0)
        if z_long.size(0) < 2:
            return z_long.new_tensor(0.0)
        z1 = (z_long - z_long.mean(dim=0, keepdim=True)) / (z_long.std(dim=0, keepdim=True) + 1e-6)
        z2 = (z_short - z_short.mean(dim=0, keepdim=True)) / (z_short.std(dim=0, keepdim=True) + 1e-6)
        corr = torch.mm(z1.t(), z2) / float(z1.size(0))
        off_diag = corr - torch.diag(torch.diag(corr))
        return off_diag.pow(2).mean() * self.view_decorrelation_weight

    @property
    def embed_dim(self):
        if self.ablation not in ('wo_long', 'wo_short'):
            if self.fusion_mode == 'concat_synergy':
                return self.hidden_dim * 3
            if self.fusion_mode in ('concat', 'concat_calibrated', 'pair_cross'):
                return self.hidden_dim * 2
        return self.hidden_dim

    def forward_joint(self, graph_feats, smiles_feats, pos_drug, pos_fc,
                      multi_hop_drug, multi_hop_fc,
                      nei_index_drug, nei_index_fc,
                      nei_weight_drug=None, nei_weight_fc=None,
                      nei_degree_drug=None, nei_degree_fc=None,
                      target_side='fc', pos_food=None, multi_hop_food=None,
                      nei_index_food=None, nei_weight_food=None, nei_degree_food=None):
        h_all = self.project_features(graph_feats, smiles_feats)
        h_drug, h_food, h_fc = h_all[0], h_all[1], h_all[2]

        cl_loss = torch.tensor(0.0, device=graph_feats[0].device)

        if self.ablation != 'wo_long':
            z_mamba_drug = self._long_bottleneck(self.mamba_drug(multi_hop_drug), 'drug')
            z_mamba_drug_cl = z_mamba_drug
        else:
            z_mamba_drug = None
            z_mamba_drug_cl = None

        if self.ablation != 'wo_short':
            z_lo_drug = self.lo_drug([h_drug, h_food, h_fc], nei_index_drug, nei_weight_drug, nei_degree_drug)
            z_lo_drug_cl = z_lo_drug
        else:
            z_lo_drug = None
            z_lo_drug_cl = None

        if self.ablation not in ('wo_cl', 'wo_long', 'wo_short'):
            cl_loss = cl_loss + self.contrast_drug(z_mamba_drug_cl, z_lo_drug_cl, pos_drug)
            cl_loss = cl_loss + self._view_decorrelation_loss(z_mamba_drug_cl, z_lo_drug_cl)

        z_drug = self._combine(z_mamba_drug, z_lo_drug, side='drug')
        if target_side == 'fc':
            if self.ablation != 'wo_long':
                z_mamba_right = self._long_bottleneck(self.mamba_fc(multi_hop_fc), 'fc')
                z_mamba_right_cl = z_mamba_right
            else:
                z_mamba_right = None
                z_mamba_right_cl = None

            if self.ablation != 'wo_short':
                z_lo_right = self.lo_fc([h_fc, h_drug, h_food], nei_index_fc, nei_weight_fc, nei_degree_fc)
                z_lo_right_cl = z_lo_right
            else:
                z_lo_right = None
                z_lo_right_cl = None

            if self.ablation not in ('wo_cl', 'wo_long', 'wo_short'):
                cl_loss = cl_loss + self.contrast_fc(z_mamba_right_cl, z_lo_right_cl, pos_fc)
                cl_loss = cl_loss + self._view_decorrelation_loss(z_mamba_right_cl, z_lo_right_cl)
            z_right = self._combine(z_mamba_right, z_lo_right, side='fc')
        elif target_side == 'food':
            if not self.has_food_branch:
                raise ValueError('Food branch is not initialized but target_side=food')
            if self.ablation != 'wo_long':
                z_mamba_right = self._long_bottleneck(self.mamba_food(multi_hop_food), 'food')
                z_mamba_right_cl = z_mamba_right
            else:
                z_mamba_right = None
                z_mamba_right_cl = None

            if self.ablation != 'wo_short':
                z_lo_right = self.lo_food([h_food, h_drug, h_fc], nei_index_food, nei_weight_food, nei_degree_food)
                z_lo_right_cl = z_lo_right
            else:
                z_lo_right = None
                z_lo_right_cl = None

            if self.ablation not in ('wo_cl', 'wo_long', 'wo_short'):
                cl_loss = cl_loss + self.contrast_food(z_mamba_right_cl, z_lo_right_cl, pos_food)
                cl_loss = cl_loss + self._view_decorrelation_loss(z_mamba_right_cl, z_lo_right_cl)
            z_right = self._combine(z_mamba_right, z_lo_right, side='food')
        else:
            raise ValueError("target_side must be 'fc' or 'food'")
        return cl_loss, z_drug, z_right

    def get_embeds(self, graph_feats, smiles_feats, multi_hop_drug, multi_hop_fc,
                   nei_index_drug, nei_index_fc,
                   nei_weight_drug=None, nei_weight_fc=None,
                   nei_degree_drug=None, nei_degree_fc=None,
                   target_side='fc', multi_hop_food=None,
                   nei_index_food=None, nei_weight_food=None, nei_degree_food=None):
        h_all = self.project_features(graph_feats, smiles_feats)
        h_drug, h_food, h_fc = h_all[0], h_all[1], h_all[2]

        with torch.no_grad():
            if self.ablation != 'wo_long':
                z_mamba_drug = self._long_bottleneck(self.mamba_drug(multi_hop_drug), 'drug')
                if target_side == 'fc':
                    z_mamba_right = self._long_bottleneck(self.mamba_fc(multi_hop_fc), 'fc')
                elif target_side == 'food':
                    z_mamba_right = self._long_bottleneck(self.mamba_food(multi_hop_food), 'food')
                else:
                    raise ValueError("target_side must be 'fc' or 'food'")
            else:
                z_mamba_drug = z_mamba_right = None

            if self.ablation != 'wo_short':
                z_lo_drug = self.lo_drug([h_drug, h_food, h_fc], nei_index_drug, nei_weight_drug, nei_degree_drug)
                if target_side == 'fc':
                    z_lo_right = self.lo_fc([h_fc, h_drug, h_food], nei_index_fc, nei_weight_fc, nei_degree_fc)
                elif target_side == 'food':
                    z_lo_right = self.lo_food([h_food, h_drug, h_fc], nei_index_food, nei_weight_food, nei_degree_food)
                else:
                    raise ValueError("target_side must be 'fc' or 'food'")
            else:
                z_lo_drug = z_lo_right = None

        z_drug = self._combine(z_mamba_drug, z_lo_drug, side='drug')
        z_right = self._combine(z_mamba_right, z_lo_right, side=target_side)
        return z_drug.detach(), z_right.detach()


class LinkPredictor(nn.Module):

    def __init__(self, embed_dim, hidden_dim=32, dropout=0.3, pair_feat_dim=0, use_edge_ops=False):
        super().__init__()
        self.pair_feat_dim = int(pair_feat_dim)
        self.use_edge_ops = bool(use_edge_ops)
        input_dim = embed_dim * (4 if self.use_edge_ops else 2) + self.pair_feat_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=1.414)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, z_drug, z_fc, edges, pair_feat=None):
        h_drug = z_drug[edges[:, 0]]
        h_fc = z_fc[edges[:, 1]]
        inputs = [h_drug, h_fc]
        if self.use_edge_ops:
            inputs.extend([h_drug * h_fc, torch.abs(h_drug - h_fc)])
        if self.pair_feat_dim > 0:
            if pair_feat is None:
                pair_feat = torch.zeros(
                    edges.size(0),
                    self.pair_feat_dim,
                    device=edges.device,
                    dtype=h_drug.dtype,
                )
            inputs.append(pair_feat.to(dtype=h_drug.dtype, device=h_drug.device))
        return self.mlp(torch.cat(inputs, dim=1)).squeeze(-1)


class PairCrossLinkPredictor(nn.Module):

    def __init__(self, hidden_dim, hidden_mlp=32, dropout=0.3, pair_feat_dim=0,
                 ablation='none', main_weight=1.0, long_weight=1.0,
                 short_weight=0.25, learn_weights=False):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.pair_feat_dim = int(pair_feat_dim)
        self.ablation = ablation or 'none'
        input_dim = self.hidden_dim * 10 + self.pair_feat_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_mlp),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_mlp, max(hidden_mlp // 2, 1)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(max(hidden_mlp // 2, 1), 1),
        )
        branch_dim = self.hidden_dim * 4
        self.long_head = nn.Sequential(
            nn.Linear(branch_dim, hidden_mlp),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_mlp, 1),
        )
        self.short_head = nn.Sequential(
            nn.Linear(branch_dim, hidden_mlp),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_mlp, 1),
        )
        init_weights = torch.tensor(
            [float(main_weight), float(long_weight), float(short_weight)],
            dtype=torch.float32,
        )
        if learn_weights:
            self.logit_weights = nn.Parameter(init_weights)
        else:
            self.register_buffer('logit_weights', init_weights)

        for m in list(self.mlp) + list(self.long_head) + list(self.short_head):
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=1.414)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _split_views(self, h):
        if h.size(1) == self.hidden_dim * 2:
            return h[:, :self.hidden_dim], h[:, self.hidden_dim:]
        if h.size(1) != self.hidden_dim:
            raise ValueError(
                f"PairCrossLinkPredictor expected embedding dim {self.hidden_dim} "
                f"or {self.hidden_dim * 2}, got {h.size(1)}"
            )
        zeros = torch.zeros_like(h)
        if self.ablation == 'wo_short':
            return h, zeros
        if self.ablation == 'wo_long':
            return zeros, h
        return h, h

    def forward(self, z_drug, z_fc, edges, pair_feat=None):
        h_drug = z_drug[edges[:, 0]]
        h_fc = z_fc[edges[:, 1]]
        d_long, d_short = self._split_views(h_drug)
        r_long, r_short = self._split_views(h_fc)

        long_inputs = [d_long, r_long, d_long * r_long, torch.abs(d_long - r_long)]
        short_inputs = [d_short, r_short, d_short * r_short, torch.abs(d_short - r_short)]




        if self.ablation == 'wo_short':
            return self.long_head(torch.cat(long_inputs, dim=1)).squeeze(-1)
        if self.ablation == 'wo_long':
            return self.short_head(torch.cat(short_inputs, dim=1)).squeeze(-1)

        inputs = [
            d_long,
            r_long,
            d_short,
            r_short,
            long_inputs[2],
            short_inputs[2],
            long_inputs[3],
            short_inputs[3],
            d_long * r_short,
            d_short * r_long,
        ]
        if self.pair_feat_dim > 0:
            if pair_feat is None:
                pair_feat = torch.zeros(
                    edges.size(0),
                    self.pair_feat_dim,
                    device=edges.device,
                    dtype=h_drug.dtype,
                )
            inputs.append(pair_feat.to(dtype=h_drug.dtype, device=h_drug.device))
        main_logit = self.mlp(torch.cat(inputs, dim=1)).squeeze(-1)
        long_logit = self.long_head(torch.cat(long_inputs, dim=1)).squeeze(-1)
        short_logit = self.short_head(torch.cat(short_inputs, dim=1)).squeeze(-1)
        weights = self.logit_weights.to(dtype=main_logit.dtype, device=main_logit.device)
        return weights[0] * main_logit + weights[1] * long_logit + weights[2] * short_logit



import torch
import torch.nn as nn

class DHMMDFIEncoder(DHMMDFIEncoderCore):
    def __init__(self, *args, short_view_source='target', **kwargs):
        super().__init__(*args, **kwargs)
        self.short_view_source = short_view_source

    def _drug_short_inputs(self, h_drug, h_food, h_fc, target_side):
        if target_side == 'fc':
            if self.short_view_source == 'target':
                return [h_drug, h_fc]
            if self.short_view_source == 'bridge':
                return [h_drug, h_food]
            if self.short_view_source == 'all':
                return [h_drug, h_fc, h_food]
        else:
            if self.short_view_source == 'target':
                return [h_drug, h_food]
            if self.short_view_source == 'bridge':
                return [h_drug, h_fc]
            if self.short_view_source == 'all':
                return [h_drug, h_food, h_fc]
        raise ValueError("short_view_source must be 'target', 'bridge', or 'all'")

    def _right_short_inputs(self, h_drug, h_food, h_fc, target_side):
        if target_side == 'fc':
            if self.short_view_source == 'target':
                return [h_fc, h_drug]
            if self.short_view_source == 'bridge':
                return [h_fc, h_food]
            if self.short_view_source == 'all':
                return [h_fc, h_drug, h_food]
        else:
            if self.short_view_source == 'target':
                return [h_food, h_drug]
            if self.short_view_source == 'bridge':
                return [h_food, h_fc]
            if self.short_view_source == 'all':
                return [h_food, h_drug, h_fc]
        raise ValueError("target_side must be 'fc' or 'food'")

    def forward_joint(self, graph_feats, smiles_feats, pos_drug, pos_fc,
                      multi_hop_drug, multi_hop_fc,
                      nei_index_drug, nei_index_fc,
                      nei_weight_drug=None, nei_weight_fc=None,
                      nei_degree_drug=None, nei_degree_fc=None,
                      target_side='fc', pos_food=None, multi_hop_food=None,
                      nei_index_food=None, nei_weight_food=None, nei_degree_food=None):
        h_drug, h_food, h_fc = self.project_features(graph_feats, smiles_feats)
        device = graph_feats[0].device
        cl_loss = torch.tensor(0.0, device=device)

        z_mamba_drug = None
        z_lo_drug = None
        if self.ablation != 'wo_long':
            z_mamba_drug = self._long_bottleneck(self.mamba_drug(multi_hop_drug), 'drug')
        if self.ablation != 'wo_short':
            z_lo_drug = self.lo_drug(
                self._drug_short_inputs(h_drug, h_food, h_fc, target_side),
                nei_index_drug,
                nei_weight_drug,
                nei_degree_drug,
            )
        if self.ablation not in ('wo_cl', 'wo_long', 'wo_short'):
            cl_loss = cl_loss + self.contrast_drug(z_mamba_drug, z_lo_drug, pos_drug)
            cl_loss = cl_loss + self._view_decorrelation_loss(z_mamba_drug, z_lo_drug)
        z_drug = self._combine(z_mamba_drug, z_lo_drug, side='drug')

        z_mamba_right = None
        z_lo_right = None
        if target_side == 'fc':
            if self.ablation != 'wo_long':
                z_mamba_right = self._long_bottleneck(self.mamba_fc(multi_hop_fc), 'fc')
            if self.ablation != 'wo_short':
                z_lo_right = self.lo_fc(
                    self._right_short_inputs(h_drug, h_food, h_fc, target_side),
                    nei_index_fc,
                    nei_weight_fc,
                    nei_degree_fc,
                )
            if self.ablation not in ('wo_cl', 'wo_long', 'wo_short'):
                cl_loss = cl_loss + self.contrast_fc(z_mamba_right, z_lo_right, pos_fc)
                cl_loss = cl_loss + self._view_decorrelation_loss(z_mamba_right, z_lo_right)
            z_right = self._combine(z_mamba_right, z_lo_right, side='fc')
        elif target_side == 'food':
            if not self.has_food_branch:
                raise ValueError('Food branch is not initialized but target_side=food')
            if self.ablation != 'wo_long':
                z_mamba_right = self._long_bottleneck(self.mamba_food(multi_hop_food), 'food')
            if self.ablation != 'wo_short':
                z_lo_right = self.lo_food(
                    self._right_short_inputs(h_drug, h_food, h_fc, target_side),
                    nei_index_food,
                    nei_weight_food,
                    nei_degree_food,
                )
            if self.ablation not in ('wo_cl', 'wo_long', 'wo_short'):
                cl_loss = cl_loss + self.contrast_food(z_mamba_right, z_lo_right, pos_food)
                cl_loss = cl_loss + self._view_decorrelation_loss(z_mamba_right, z_lo_right)
            z_right = self._combine(z_mamba_right, z_lo_right, side='food')
        else:
            raise ValueError("target_side must be 'fc' or 'food'")

        return cl_loss, z_drug, z_right

    def get_embeds(self, graph_feats, smiles_feats, multi_hop_drug, multi_hop_fc,
                   nei_index_drug, nei_index_fc,
                   nei_weight_drug=None, nei_weight_fc=None,
                   nei_degree_drug=None, nei_degree_fc=None,
                   target_side='fc', multi_hop_food=None,
                   nei_index_food=None, nei_weight_food=None, nei_degree_food=None):
        h_drug, h_food, h_fc = self.project_features(graph_feats, smiles_feats)

        with torch.no_grad():
            z_mamba_drug = None
            z_lo_drug = None
            if self.ablation != 'wo_long':
                z_mamba_drug = self._long_bottleneck(self.mamba_drug(multi_hop_drug), 'drug')
            if self.ablation != 'wo_short':
                z_lo_drug = self.lo_drug(
                    self._drug_short_inputs(h_drug, h_food, h_fc, target_side),
                    nei_index_drug,
                    nei_weight_drug,
                    nei_degree_drug,
                )

            z_mamba_right = None
            z_lo_right = None
            if target_side == 'fc':
                if self.ablation != 'wo_long':
                    z_mamba_right = self._long_bottleneck(self.mamba_fc(multi_hop_fc), 'fc')
                if self.ablation != 'wo_short':
                    z_lo_right = self.lo_fc(
                        self._right_short_inputs(h_drug, h_food, h_fc, target_side),
                        nei_index_fc,
                        nei_weight_fc,
                        nei_degree_fc,
                    )
                side = 'fc'
            elif target_side == 'food':
                if self.ablation != 'wo_long':
                    z_mamba_right = self._long_bottleneck(self.mamba_food(multi_hop_food), 'food')
                if self.ablation != 'wo_short':
                    z_lo_right = self.lo_food(
                        self._right_short_inputs(h_drug, h_food, h_fc, target_side),
                        nei_index_food,
                        nei_weight_food,
                        nei_degree_food,
                    )
                side = 'food'
            else:
                raise ValueError("target_side must be 'fc' or 'food'")

        z_drug = self._combine(z_mamba_drug, z_lo_drug, side='drug')
        z_right = self._combine(z_mamba_right, z_lo_right, side=side)
        return z_drug.detach(), z_right.detach()





class ComplementaryLinkPredictor(nn.Module):


    def __init__(self, hidden_dim, hidden_mlp=16, dropout=0.3,
                 ablation='none', long_weight=0.8, short_weight=0.8,
                 synergy_weight=1.0, learn_weights=False):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.ablation = ablation or 'none'

        branch_dim = self.hidden_dim * 4
        synergy_dim = self.hidden_dim * 4
        self.long_head = self._make_head(branch_dim, hidden_mlp, dropout)
        self.short_head = self._make_head(branch_dim, hidden_mlp, dropout)
        self.synergy_head = self._make_head(synergy_dim, hidden_mlp, dropout)

        init = torch.tensor([float(long_weight), float(short_weight), float(synergy_weight)])
        if learn_weights:
            self.logit_weights = nn.Parameter(init)
        else:
            self.register_buffer('logit_weights', init)

    @staticmethod
    def _make_head(input_dim, hidden_mlp, dropout):
        hidden_mlp = max(int(hidden_mlp), 4)
        return nn.Sequential(
            nn.Linear(input_dim, hidden_mlp),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_mlp, 1),
        )

    def _split_views(self, h):
        if h.size(1) == self.hidden_dim:
            return h, h
        if h.size(1) != self.hidden_dim * 2:
            raise ValueError(
                f'ComplementaryLinkPredictor expects hidden_dim or 2*hidden_dim, got {h.size(1)}'
            )
        return h[:, :self.hidden_dim], h[:, self.hidden_dim:]

    def forward(self, z_drug, z_right, edges, pair_feat=None):
        del pair_feat
        h_drug = z_drug[edges[:, 0]]
        h_right = z_right[edges[:, 1]]
        d_long, d_short = self._split_views(h_drug)
        r_long, r_short = self._split_views(h_right)

        long_feat = torch.cat([d_long, r_long, d_long * r_long, torch.abs(d_long - r_long)], dim=1)
        short_feat = torch.cat([d_short, r_short, d_short * r_short, torch.abs(d_short - r_short)], dim=1)
        cross_feat = torch.cat([
            d_long * r_short,
            d_short * r_long,
            torch.abs(d_long - r_short),
            torch.abs(d_short - r_long),
        ], dim=1)

        if self.ablation == 'wo_short':
            return self.long_head(long_feat).squeeze(-1)
        if self.ablation == 'wo_long':
            return self.short_head(short_feat).squeeze(-1)

        long_logit = self.long_head(long_feat).squeeze(-1)
        short_logit = self.short_head(short_feat).squeeze(-1)
        synergy_logit = self.synergy_head(cross_feat).squeeze(-1)
        weights = self.logit_weights.to(dtype=long_logit.dtype, device=long_logit.device)
        return weights[0] * long_logit + weights[1] * short_logit + weights[2] * synergy_logit



import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))



class DHMMDFI(DHMMDFIEncoder):

    pass


def build_training_modules(fold_data, device):
    nd = fold_data['graph_feats'][0].shape[0]
    nfc = fold_data['graph_feats'][2].shape[0]
    target_side = fold_data.get('target_type', 'fc')
    graph_feat_dim_list = [feat.shape[1] for feat in fold_data['graph_feats']]
    nei_num_drug = len(fold_data['nei_drug'])
    nei_num_fc = len(fold_data['nei_fc'])
    nei_num_food = len(fold_data.get('nei_food', []))
    sample_rate_drug = list(args.sample_rate_drug[:nei_num_drug])
    sample_rate_fc = list(args.sample_rate_fc[:nei_num_fc])
    sample_rate_food = list(args.sample_rate_drug[:nei_num_food])
    if len(sample_rate_drug) < nei_num_drug:
        sample_rate_drug.extend([sample_rate_drug[-1] if sample_rate_drug else 8] * (nei_num_drug - len(sample_rate_drug)))
    if len(sample_rate_fc) < nei_num_fc:
        sample_rate_fc.extend([sample_rate_fc[-1] if sample_rate_fc else 8] * (nei_num_fc - len(sample_rate_fc)))
    if len(sample_rate_food) < nei_num_food:
        sample_rate_food.extend([sample_rate_food[-1] if sample_rate_food else 8] * (nei_num_food - len(sample_rate_food)))



    encoder_ablation = args.ablation
    if getattr(args, 'logit_level_ablation', False) and args.ablation in ('wo_long', 'wo_short'):
        encoder_ablation = 'none'

    model = DHMMDFI(
        hidden_dim=args.hidden_dim,
        graph_feat_dim_list=graph_feat_dim_list,
        feat_drop=args.feat_drop,
        P_drug=fold_data['P_drug'],
        P_fc=fold_data['P_fc'],
        P_food=fold_data.get('P_food') if target_side == 'food' else None,
        num_hops=args.num_hops,
        long_hidden_dim=args.long_hidden_dim,
        d_state=args.d_state,
        d_conv=args.d_conv,
        expand=args.expand,
        mamba_dropout=args.mamba_dropout,
        n_mamba_layers=args.n_mamba_layers,
        long_encoder_mode=args.long_encoder_mode,
        sample_rate_drug=sample_rate_drug,
        sample_rate_fc=sample_rate_fc,
        sample_rate_food=sample_rate_food,
        nei_num_drug=nei_num_drug,
        nei_num_fc=nei_num_fc,
        nei_num_food=nei_num_food,
        attn_drop=args.attn_drop,
        tau=args.tau,
        lam=args.lam,
        fusion_mode=args.fusion_mode,
        ablation=encoder_ablation,
        short_use_log_degree_penalty=args.short_use_log_degree_penalty,
        short_degree_penalty_bias=args.short_degree_penalty_bias,
        short_second_layer=args.short_second_layer,
        short_output_drop=args.short_output_drop,
        short_self_weight=args.short_self_weight,
        short_bottleneck_dim=args.short_bottleneck_dim,
        short_relation_drop=args.short_relation_drop,
        long_output_drop=args.long_output_drop,
        long_meta_path_drop=args.long_meta_path_drop,
        view_norm=args.view_norm,
        cross_view_residual=args.cross_view_residual,
        cross_view_residual_scale=args.cross_view_residual_scale,
        short_view_source=args.short_view_source,
        cl_align_mode=args.cl_align_mode,
        view_decorrelation_weight=args.view_decorrelation_weight,
        gate_init_long_weight=args.gate_init_long_weight,
    ).to(device)

    drug_mh_dim = fold_data['drug_mh'][0].shape[-1]
    right_mh = fold_data['fc_mh'] if target_side == 'fc' else fold_data['food_mh']
    right_mh_dim = right_mh[0].shape[-1]
    drug_proj = nn.Linear(drug_mh_dim, args.hidden_dim).to(device)
    fc_proj = nn.Linear(right_mh_dim, args.hidden_dim).to(device)
    nn.init.xavier_normal_(drug_proj.weight, gain=1.414)
    nn.init.xavier_normal_(fc_proj.weight, gain=1.414)

    if args.predictor_type == 'view_interaction':
        predictor = ComplementaryLinkPredictor(
            args.hidden_dim,
            args.lp_hidden,
            args.lp_dropout,
            ablation=args.ablation,
            long_weight=args.pair_long_weight,
            short_weight=args.pair_short_weight,
            synergy_weight=args.pair_synergy_weight,
            learn_weights=args.learn_pair_logit_weights,
        ).to(device)
    elif args.predictor_type == 'mlp':
        predictor = LinkPredictor(
            model.embed_dim,
            args.lp_hidden,
            args.lp_dropout,
            pair_feat_dim=0,
            use_edge_ops=args.predictor_edge_ops,
        ).to(device)
    else:
        raise ValueError(f'Unsupported predictor_type: {args.predictor_type}')

    all_params = (
        list(model.parameters())
        + list(drug_proj.parameters())
        + list(fc_proj.parameters())
        + list(predictor.parameters())
    )
    optimizer = torch.optim.Adam(all_params, lr=args.lr, weight_decay=args.l2)
    return model, drug_proj, fc_proj, predictor, optimizer
