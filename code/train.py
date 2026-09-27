import os
import csv
import random

import numpy as np
import torch


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True






import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
import torch


def evaluate_lp(predictor, z_drug, z_fc, pos_edges, neg_edges, device, threshold=0.5):
    pos = torch.LongTensor(pos_edges).to(device)
    neg = torch.LongTensor(neg_edges).to(device)
    with torch.no_grad():
        pos_logits = predictor(z_drug, z_fc, pos).view(-1)
        neg_logits = predictor(z_drug, z_fc, neg).view(-1)
        scores = torch.cat([torch.sigmoid(pos_logits), torch.sigmoid(neg_logits)]).cpu().numpy()
        labels = np.concatenate([np.ones(len(pos_edges)), np.zeros(len(neg_edges))])
        preds = (scores >= threshold).astype(int)
    return {
        'AUC': roc_auc_score(labels, scores),
        'AUPR': average_precision_score(labels, scores),
        'F1': f1_score(labels, preds, zero_division=0),
        'Recall': recall_score(labels, preds, zero_division=0),
        'Precision': precision_score(labels, preds, zero_division=0),
    }





import torch
import torch.nn as nn

from config import args


def train_joint_model(model, drug_proj, fc_proj, predictor, optimizer, fold_data, device, train_neg_edges, epochs_to_run):
    train_pos = torch.LongTensor(fold_data['train_pos']).to(device)
    train_neg = torch.LongTensor(train_neg_edges).to(device)

    pos_weight = torch.tensor(
        [float(len(train_neg_edges)) / max(float(len(fold_data['train_pos'])), 1.0)],
        device=device,
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    labels = torch.cat([
        torch.ones(len(train_pos), device=device),
        torch.zeros(len(train_neg), device=device),
    ])

    graph_feats = fold_data['graph_feats']
    smiles_feats = fold_data['smiles_feats']
    target_side = fold_data.get('target_type', 'fc')

    for epoch_idx in range(epochs_to_run):
        model.train()
        drug_proj.train()
        fc_proj.train()
        predictor.train()
        optimizer.zero_grad()

        drug_multi_hop = [drug_proj(m) for m in fold_data['drug_mh']]
        right_mh_key = 'fc_mh' if target_side == 'fc' else 'food_mh'
        right_multi_hop = [fc_proj(m) for m in fold_data[right_mh_key]]

        cl_loss, z_drug, z_right = model.forward_joint(
            graph_feats,
            smiles_feats,
            fold_data['pos_drug'],
            fold_data['pos_fc'],
            drug_multi_hop,
            right_multi_hop if target_side == 'fc' else None,
            fold_data['nei_drug'],
            fold_data['nei_fc'] if target_side == 'fc' else None,
            fold_data.get('nei_weight_drug'),
            fold_data.get('nei_weight_fc') if target_side == 'fc' else None,
            fold_data.get('nei_degree_drug'),
            fold_data.get('nei_degree_fc') if target_side == 'fc' else None,
            target_side=target_side,
            pos_food=fold_data.get('pos_food'),
            multi_hop_food=right_multi_hop if target_side == 'food' else None,
            nei_index_food=fold_data.get('nei_food') if target_side == 'food' else None,
            nei_weight_food=fold_data.get('nei_weight_food') if target_side == 'food' else None,
            nei_degree_food=fold_data.get('nei_degree_food') if target_side == 'food' else None,
        )

        scores = torch.cat([
            predictor(z_drug, z_right, train_pos),
            predictor(z_drug, z_right, train_neg),
        ])
        lp_loss = criterion(scores, labels)
        total_loss = lp_loss + cl_loss
        total_loss.backward()
        grad_clip = float(getattr(args, 'grad_clip', 0.0))
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters())
                + list(drug_proj.parameters())
                + list(fc_proj.parameters())
                + list(predictor.parameters()),
                max_norm=grad_clip,
            )
        optimizer.step()

        if args.log_interval > 0:
            curr_epoch = epoch_idx + 1
            if curr_epoch == 1 or curr_epoch % args.log_interval == 0 or curr_epoch == epochs_to_run:
                print(
                    f"      Epoch {curr_epoch}/{epochs_to_run} | "
                    f"lp_loss={lp_loss.item():.4f} cl_loss={cl_loss.item():.4f} "
                    f"total_loss={total_loss.item():.4f}"
                )





import torch

from config import args
from model import build_training_modules


def run_single(seed, fold_data, device):
    set_seed(seed)
    graph_feats = fold_data['graph_feats']
    smiles_feats = fold_data['smiles_feats']
    target_side = fold_data.get('target_type', 'fc')

    model, drug_proj, fc_proj, predictor, optimizer = build_training_modules(fold_data, device)
    train_joint_model(
        model,
        drug_proj,
        fc_proj,
        predictor,
        optimizer,
        fold_data,
        device,
        fold_data['train_neg'],
        args.epochs,
    )

    model.eval()
    drug_proj.eval()
    fc_proj.eval()
    predictor.eval()
    with torch.no_grad():
        drug_multi_hop = [drug_proj(m) for m in fold_data['drug_mh']]
        right_mh_key = 'fc_mh' if target_side == 'fc' else 'food_mh'
        right_multi_hop = [fc_proj(m) for m in fold_data[right_mh_key]]
        z_drug, z_right = model.get_embeds(
            graph_feats,
            smiles_feats,
            drug_multi_hop,
            right_multi_hop if target_side == 'fc' else None,
            fold_data['nei_drug'],
            fold_data['nei_fc'] if target_side == 'fc' else None,
            fold_data.get('nei_weight_drug'),
            fold_data.get('nei_weight_fc') if target_side == 'fc' else None,
            fold_data.get('nei_degree_drug'),
            fold_data.get('nei_degree_fc') if target_side == 'fc' else None,
            target_side=target_side,
            multi_hop_food=right_multi_hop if target_side == 'food' else None,
            nei_index_food=fold_data.get('nei_food') if target_side == 'food' else None,
            nei_weight_food=fold_data.get('nei_weight_food') if target_side == 'food' else None,
            nei_degree_food=fold_data.get('nei_degree_food') if target_side == 'food' else None,
        )

    return evaluate_lp(
        predictor,
        z_drug,
        z_right,
        fold_data['test_pos'],
        fold_data['test_neg'],
        device,
        args.threshold,
    )





import os
import numpy as np

from config import args


METRICS = ['AUC', 'AUPR', 'F1', 'Recall', 'Precision']


def _tag(value):
    return str(value).replace('.', 'p')


def _short_long_strength_name(name):
    mapping = {
        'semantic_lite': 'semlite',
        'semantic_mid_lite': 'semmid',
        'semantic_search_a': 'seasa',
        'semantic_search_b': 'seasb',
        'semantic_search_c': 'seasc',
        'semantic_search_d': 'seasd',
        'semantic_search_e': 'sease',
        'semantic_search_f': 'seasf',
        'semantic_search_g': 'seasg',
        'semantic_search_h': 'seash',
        'semantic_search_i': 'seasi',
        'semantic_search_j': 'seasj',
        'semantic_search_k': 'seask',
        'semantic_search_balanced': 'seasbal',
        'semantic_search_balanced_weak': 'seasbalw',
        'semantic_search_core': 'seascore',
        'semantic_restore_lite': 'semrestore',
        'semantic_restore_balanced': 'semrestorebal',
        'semantic_restore_synergy': 'semrestoresyn',
        'semantic_restore_strong': 'semrestorestrong',
        'semantic_low_lite': 'semlow',
        'semantic_micro_lite': 'semmicro',
        'semantic_ultra_lite': 'semultra',
        'minimal': 'min',
        'compact': 'compact',
        'original': 'orig',
        'ultra': 'ultra',
        'lite': 'lite',
    }
    return mapping.get(str(name), str(name))


def build_output_suffix(meta_mode='base'):
    task = getattr(args, 'task', 'dfc')
    out_suffix = f'_{task}'
    if args.ablation != 'none':
        out_suffix += f'_{args.ablation}'
        if getattr(args, 'logit_level_ablation', False) and args.ablation in ('wo_long', 'wo_short'):
            out_suffix += '_logitab'
    predictor_type = getattr(args, 'predictor_type', 'mlp')
    fusion_mode = getattr(args, 'fusion_mode', 'concat')
    if predictor_type != 'mlp':
        out_suffix += f'_pred{predictor_type}'
    if predictor_type == 'mlp' and getattr(args, 'predictor_edge_ops', False):
        out_suffix += '_edgeops'
    if fusion_mode != 'concat':
        out_suffix += f'_fusion{fusion_mode}'
    long_view_mode = getattr(args, 'long_view_mode', 'symmetric_long')
    if long_view_mode != 'symmetric_long':
        out_suffix += f'_long{long_view_mode}'
    long_hop_mode = getattr(args, 'long_hop_mode', 'soft_context')
    if long_hop_mode != 'soft_context':
        out_suffix += f'_hop{long_hop_mode}'
    default_long_strength = 'semantic_restore_balanced'
    long_strength = getattr(args, 'long_strength', default_long_strength)
    long_strength_controls_detail = long_strength != default_long_strength
    if long_strength != default_long_strength:


        out_suffix += f'_ls{_short_long_strength_name(long_strength)}'
    if long_hop_mode == 'soft_context':
        self_w = float(getattr(args, 'long_self_weight', 0.05))
        near_w = float(getattr(args, 'long_near_weight', 0.15))
        second_w = float(getattr(args, 'long_second_weight', 1.0))
        late_w = float(getattr(args, 'long_late_weight', 1.0))
        if not long_strength_controls_detail and (
            abs(self_w - 0.05) > 1e-12
            or abs(near_w - 0.15) > 1e-12
            or abs(second_w - 1.0) > 1e-12
            or abs(late_w - 1.0) > 1e-12
        ):
            out_suffix += f'_hopw{_tag(self_w)}_{_tag(near_w)}_{_tag(second_w)}_{_tag(late_w)}'
    pathsim_topk = int(getattr(args, 'pathsim_topk', 8))
    pathsim_topk_ratio = float(getattr(args, 'pathsim_topk_ratio', 0.06))
    pathsim_min_topk = int(getattr(args, 'pathsim_min_topk', 2))
    pathsim_min_sim = float(getattr(args, 'pathsim_min_sim', 0.0))
    if not long_strength_controls_detail and (
        pathsim_topk != 8
        or abs(pathsim_topk_ratio - 0.06) > 1e-12
        or pathsim_min_topk != 2
        or abs(pathsim_min_sim) > 1e-12
    ):
        out_suffix += (
            f'_psk{pathsim_topk}_psr{_tag(pathsim_topk_ratio)}'
            f'_psmin{pathsim_min_topk}_psm{_tag(pathsim_min_sim)}'
        )
    short_strength = getattr(args, 'short_strength', 'type_regularized')
    if short_strength != 'type_regularized':
        out_suffix += f'_shortstr{short_strength}'
    short_view_source = getattr(args, 'short_view_source', 'bridge')
    if short_view_source != 'bridge':
        out_suffix += f'_shortsrc{short_view_source}'
    short_second_layer = getattr(args, 'short_second_layer', 'type_attn')
    if short_second_layer != 'type_attn':
        out_suffix += f'_short{short_second_layer}'
    short_output_drop = float(getattr(args, 'short_output_drop', 0.08))
    if abs(short_output_drop - 0.08) > 1e-12:
        out_suffix += f'_sdrop{_tag(short_output_drop)}'
    short_self_weight = float(getattr(args, 'short_self_weight', 0.70))
    if abs(short_self_weight - 0.70) > 1e-12:
        out_suffix += f'_sself{_tag(short_self_weight)}'
    short_bottleneck_dim = int(getattr(args, 'short_bottleneck_dim', 0))
    if short_bottleneck_dim != 0:
        out_suffix += f'_sbn{short_bottleneck_dim}'
    sample_rate_drug = list(getattr(args, 'sample_rate_drug', [5]))
    sample_rate_fc = list(getattr(args, 'sample_rate_fc', [5]))
    if sample_rate_drug != [5] or sample_rate_fc != [5]:
        out_suffix += f'_sr{"x".join(str(x) for x in sample_rate_drug)}'
    long_encoder_mode = getattr(args, 'long_encoder_mode', 'semantic')
    if not long_strength_controls_detail and long_encoder_mode != 'semantic':
        out_suffix += f'_longenc{long_encoder_mode}'
    cl_align_mode = getattr(args, 'cl_align_mode', 'symmetric')
    if cl_align_mode != 'symmetric':
        out_suffix += f'_cl{cl_align_mode}'
    cl_pos_mode = getattr(args, 'cl_pos_mode', 'self')
    if cl_pos_mode != 'self':
        out_suffix += f'_clpos{cl_pos_mode}'
    long_hidden_dim = int(getattr(args, 'long_hidden_dim', 52))
    if not long_strength_controls_detail and long_hidden_dim != 52:
        out_suffix += f'_longhid{long_hidden_dim}'
    long_meta_path_drop = float(getattr(args, 'long_meta_path_drop', 0.0))
    if abs(long_meta_path_drop) > 1e-12:
        out_suffix += f'_mpdrop{_tag(long_meta_path_drop)}'
    if meta_mode != 'base':
        out_suffix += f'_meta{meta_mode}'
    neg_strategy = getattr(args, 'neg_strategy', 'global_random')
    out_suffix += f'_neg{neg_strategy}'
    run_tag = str(getattr(args, 'run_tag', '')).strip()
    if run_tag:
        safe_tag = ''.join(ch if ch.isalnum() or ch in ('-', '_') else '_' for ch in run_tag)
        out_suffix += f'_{safe_tag}'
    return out_suffix


def format_run_summary(run_dict):
    metric_parts = [f"{metric}={float(run_dict[metric]):.4f}" for metric in METRICS]
    return f"fold={run_dict['fold']} seed={run_dict['seed']} " + ' '.join(metric_parts)


def summarize_results(all_results):
    lines = []
    summary = {}
    for metric in METRICS:
        values = [result[metric] for result in all_results]
        summary[metric] = {'mean': float(np.mean(values)), 'std': float(np.std(values))}
        lines.append(f"  {metric:>12s}: {summary[metric]['mean']:.4f} +/- {summary[metric]['std']:.4f}")

    best_run_auc = max(all_results, key=lambda result: result['AUC'])
    best_run_aupr = max(all_results, key=lambda result: result['AUPR'])
    return {
        'lines': lines,
        'summary': summary,
        'best_run_auc': best_run_auc,
        'best_run_aupr': best_run_aupr,
        'best_auc_line': f"  Best run by AUC : {format_run_summary(best_run_auc)}",
        'best_aupr_line': f"  Best run by AUPR: {format_run_summary(best_run_aupr)}",
    }


def print_result_summary(dataset, neg_ratio, all_results, summarized):
    print(f"\n{'=' * 60}")
    print(f"DHMMDFI 5-Fold CV Results (dataset={dataset}, task={getattr(args, 'task', 'dfc')}, neg_ratio={neg_ratio}, {len(all_results)} runs)")
    print(f"{'=' * 60}")
    for line in summarized['lines']:
        print(line)
    print(summarized['best_auc_line'])
    print(summarized['best_aupr_line'])


def save_results(version_dir, dataset, neg_ratio, protocol_desc, raw, all_edges, all_results, summarized, runtime_seconds, out_suffix):
    out_dir = os.path.join(version_dir, 'results', '5fold')
    os.makedirs(out_dir, exist_ok=True)
    filename = f'result_dhmmdfi_{dataset}_neg{neg_ratio}{out_suffix}.txt'


    summary_candidate = os.path.join(out_dir, os.path.splitext(filename)[0] + '_summary.csv')
    if len(filename) > 180 or len(summary_candidate) > 240:
        run_tag = str(getattr(args, 'run_tag', '')).strip()
        safe_tag = ''.join(ch if ch.isalnum() or ch in ('-', '_') else '_' for ch in run_tag)
        ablation_tag = '' if args.ablation == 'none' else f'_{args.ablation}'
        tag = safe_tag[:48] if safe_tag else 'longname'
        filename = f'result_dhmmdfi_{dataset}_neg{neg_ratio}{ablation_tag}_{tag}.txt'
    output_file = os.path.join(out_dir, filename)
    output_stem = os.path.splitext(output_file)[0]

    with open(output_file, 'w', encoding='utf-8') as file:
        file.write('Version: DHMMDFI\n')
        file.write('Model: DHMMDFI main model\n')
        file.write(f"Task: {getattr(args, 'task', 'dfc')}\n")
        file.write(f'Ablation: {args.ablation}\n')
        file.write(f'Fusion mode: {getattr(args, "fusion_mode", "concat")}\n')
        file.write(f'Predictor type: {getattr(args, "predictor_type", "mlp")}\n')
        file.write(f'Predictor edge ops: {getattr(args, "predictor_edge_ops", False)}\n')
        file.write('Short-view mode: heterogeneous low-order relation view\n')
        file.write(f'Short-view strength: {getattr(args, "short_strength", "type_regularized")}\n')
        file.write(f'Short-view source: {getattr(args, "short_view_source", "bridge")}\n')
        file.write(f'Short-view second layer: {getattr(args, "short_second_layer", "type_attn")}\n')
        file.write(f'Long-view mode: {getattr(args, "long_view_mode", "symmetric_long")}\n')
        file.write(f'Long-view strength: {getattr(args, "long_strength", "semantic_restore_balanced")}\n')
        file.write(f'Long-view hop mode: {getattr(args, "long_hop_mode", "soft_context")}\n')
        file.write(f'Long-view soft hop weights: self={getattr(args, "long_self_weight", 0.05)}, near={getattr(args, "long_near_weight", 0.15)}, second={getattr(args, "long_second_weight", 1.0)}, late={getattr(args, "long_late_weight", 1.0)}\n')
        file.write(
            f'Long-view PathSim pruning: topk={getattr(args, "pathsim_topk", 8)}, '
            f'topk_ratio={getattr(args, "pathsim_topk_ratio", 0.06)}, '
            f'min_topk={getattr(args, "pathsim_min_topk", 2)}, '
            f'min_sim={getattr(args, "pathsim_min_sim", 0.0)}\n'
        )
        file.write(f'Long-view encoder: {getattr(args, "long_encoder_mode", "semantic")}\n')
        file.write(f'Long-view bottleneck hidden dim: {getattr(args, "long_hidden_dim", args.hidden_dim)}\n')
        file.write(f'Mamba dropout: {getattr(args, "mamba_dropout", 0.15)}\n')
        file.write(f'Long-view output dropout: {getattr(args, "long_output_drop", 0.0)}\n')
        file.write(f'Long-view meta-path dropout: {getattr(args, "long_meta_path_drop", 0.0)}\n')
        file.write(f'Per-view fusion LayerNorm: {getattr(args, "view_norm", True)}\n')
        file.write(f'Cross-view residual calibration: {getattr(args, "cross_view_residual", True)}\n')
        file.write(f'Short-view sample rate: drug={args.sample_rate_drug}, fc={args.sample_rate_fc}\n')
        file.write(f'Short-view degree penalty: {args.short_use_log_degree_penalty}, bias={args.short_degree_penalty_bias}\n')
        file.write(f'Short-view output dropout: {getattr(args, "short_output_drop", 0.08)}\n')
        file.write(f'Short-view self token weight: {getattr(args, "short_self_weight", 0.70)}\n')
        file.write(f'Short-view bottleneck dim: {getattr(args, "short_bottleneck_dim", 0)}\n')
        file.write(f"Contrastive alignment mode: {getattr(args, 'cl_align_mode', 'symmetric')}\n")
        file.write(f"Contrastive positive mode: {getattr(args, 'cl_pos_mode', 'self')}\n")
        file.write(f"View decorrelation weight: {getattr(args, 'view_decorrelation_weight', 0.0)}\n")
        file.write(f"Contrastive temperature tau: {getattr(args, 'tau', 0.8)}\n")
        file.write(f"Meta-path mode: {getattr(args, 'meta_mode', 'recorded_in_protocol')}\n")
        file.write(f'Predictor type detail: {getattr(args, "predictor_type", "mlp")}\n')
        file.write(f'Logit-level ablation: {getattr(args, "logit_level_ablation", False)}\n')
        file.write(f"Negative sampling: fixed global_random train/test negative folds, positive:negative=1:{neg_ratio}, neg_seed={getattr(args, 'neg_seed', 42)}\n")
        file.write("Food feature mode: random\n")
        if getattr(args, 'predictor_type', 'mlp') == 'view_interaction':
            file.write(
                'View-interaction predictor weights: '
                f'long={getattr(args, "pair_long_weight", 0.38)}, '
                f'short={getattr(args, "pair_short_weight", 0.65)}, '
                f'synergy={getattr(args, "pair_synergy_weight", 1.15)}\n'
            )
        else:
            file.write('View-interaction predictor weights: unused in concat MLP protocol\n')
        file.write('Use SMILES features: True; Drug/FC=SMILES Morgan FP, Food=fixed random feature\n')
        file.write(f'SMILES fp dim: {args.smiles_fp_dim}\n')
        file.write(f'SMILES radius: {args.smiles_radius}\n')
        file.write('Training: fixed epochs, no validation/early stopping\n')
        file.write(f'Protocol: {protocol_desc}\n')
        file.write(f'Run tag: {getattr(args, "run_tag", "")}\n')
        file.write(f'Dataset: {dataset}\n')
        file.write(f'Negative ratio: {neg_ratio}\n')
        file.write(f'Threshold policy: fixed threshold = {args.threshold}\n')
        file.write(f'Folds: {args.n_folds}\n')
        file.write(f'Seeds per fold: {args.seeds_per_fold}\n')
        file.write(f'Total runs: {len(all_results)}\n')
        file.write(f"Nodes: drug={raw['num_drug']}, food={raw['num_food']}, fc={raw['num_fc']}\n")
        target_name = 'D-FC' if getattr(args, 'task', 'dfc') == 'dfc' else 'D-F'
        file.write(f'{target_name} target edges: {len(all_edges)}\n')
        file.write('Key hyperparameters:\n')
        file.write(f'  hidden_dim={args.hidden_dim}, long_hidden_dim={getattr(args, "long_hidden_dim", args.hidden_dim)}, num_hops={args.num_hops}, long_strength={getattr(args, "long_strength", "semantic_restore_balanced")}, long_hop_mode={getattr(args, "long_hop_mode", "soft_context")}, long_self_weight={getattr(args, "long_self_weight", 0.05)}, long_near_weight={getattr(args, "long_near_weight", 0.15)}, long_second_weight={getattr(args, "long_second_weight", 1.0)}, long_late_weight={getattr(args, "long_late_weight", 1.0)}, long_encoder_mode={getattr(args, "long_encoder_mode", "semantic")}, mamba_dropout={getattr(args, "mamba_dropout", 0.15)}, long_meta_path_drop={getattr(args, "long_meta_path_drop", 0.0)}, cl_align_mode={getattr(args, "cl_align_mode", "symmetric")}, cl_pos_mode={getattr(args, "cl_pos_mode", "self")}, view_decorrelation_weight={getattr(args, "view_decorrelation_weight", 0.0)}\n')
        file.write(f'  lr={args.lr}, l2={args.l2}, epochs={args.epochs}\n')
        file.write(f'  grad_clip={getattr(args, "grad_clip", 0.0)}\n')
        file.write(f'  feat_drop={args.feat_drop}, attn_drop={args.attn_drop}, lp_hidden={args.lp_hidden}, lp_dropout={args.lp_dropout}\n')
        file.write('\nSummary:\n')
        for line in summarized['lines']:
            file.write(line + '\n')
        file.write('\nBest-run summary:\n')
        file.write(summarized['best_auc_line'] + '\n')
        file.write(summarized['best_aupr_line'] + '\n')
        file.write(f'\nTime: {runtime_seconds}s\n')
        file.write('\nPer-run details:\n')
        for idx, result in enumerate(all_results):
            parts = []
            for key, value in result.items():
                if isinstance(value, (int, float, np.floating)):
                    parts.append(f'{key}={float(value):.4f}')
                else:
                    parts.append(f'{key}={value}')
            file.write(f"  Run {idx + 1}: " + ' '.join(parts) + '\n')

    summary_csv = output_stem + '_summary.csv'
    per_run_csv = output_stem + '_runs.csv'
    common_fields = {
        'version': 'DHMMDFI',
        'dataset': dataset,
        'task': getattr(args, 'task', 'dfc'),
        'ablation': args.ablation,
        'neg_ratio': neg_ratio,
        'epochs': args.epochs,
        'n_folds': args.n_folds,
        'seeds': '|'.join(str(seed) for seed in args.seeds_per_fold),
        'tau': getattr(args, 'tau', 0.8),
        'cl_pos_mode': getattr(args, 'cl_pos_mode', 'self'),
        'fusion_mode': getattr(args, 'fusion_mode', 'concat'),
        'predictor_type': getattr(args, 'predictor_type', 'mlp'),
        'long_view_mode': getattr(args, 'long_view_mode', 'symmetric_long'),
        'long_strength': getattr(args, 'long_strength', ''),
        'long_encoder_mode': getattr(args, 'long_encoder_mode', ''),
        'short_second_layer': getattr(args, 'short_second_layer', ''),
        'short_view_source': getattr(args, 'short_view_source', ''),
        'run_tag': getattr(args, 'run_tag', ''),
    }
    with open(summary_csv, 'w', encoding='utf-8-sig', newline='') as file:
        fieldnames = list(common_fields.keys()) + [f'{metric}_mean' for metric in METRICS] + [f'{metric}_std' for metric in METRICS]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        row = dict(common_fields)
        for metric in METRICS:
            row[f'{metric}_mean'] = summarized['summary'][metric]['mean']
            row[f'{metric}_std'] = summarized['summary'][metric]['std']
        writer.writerow(row)

    with open(per_run_csv, 'w', encoding='utf-8-sig', newline='') as file:
        fieldnames = list(common_fields.keys()) + ['fold', 'seed'] + METRICS
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for result in all_results:
            row = dict(common_fields)
            row['fold'] = result.get('fold')
            row['seed'] = result.get('seed')
            for metric in METRICS:
                row[metric] = result.get(metric)
            writer.writerow(row)

    print(f'Results saved to {output_file}')
    print(f'CSV summary saved to {summary_csv}')
    print(f'CSV per-run details saved to {per_run_csv}')
    return output_file





import datetime
import os

import torch

from config import args
from data import load_raw_data, split_edges_kfold, prepare_fold_data, build_negative_folds, merge_negative_folds


def print_run_header(dataset, task, neg_ratio, meta_mode, neg_strategy, food_feature_mode):
    display_meta_mode = 'standard' if meta_mode == 'base' else meta_mode
    print('=' * 60)
    print('DHMMDFI - 5-Fold CV [PathSim Meta-path Protocol]')
    print(
        f'dataset={dataset}, task={task}, neg_ratio={neg_ratio}, split=classic_802, '
        f'meta_mode={display_meta_mode}, '
        f'neg_protocol={neg_strategy}, food_feature_mode={food_feature_mode}, '
        f'fusion={getattr(args, "fusion_mode", "pair_cross")}, '
        f'predictor={getattr(args, "predictor_type", "view_interaction")}, '
        f'predictor_edge_ops={getattr(args, "predictor_edge_ops", False)}, '
        f'short_view_source={getattr(args, "short_view_source", "bridge")}, '
        f'short_strength={getattr(args, "short_strength", "type_regularized")}, '
        f'short_second_layer={args.short_second_layer}, '
        f'long_view_mode={args.long_view_mode}, long_encoder={args.long_encoder_mode}, '
        f'long_strength={getattr(args, "long_strength", "semantic_restore_balanced")}, long_hop_mode={args.long_hop_mode}, '
        f'long_self_weight={args.long_self_weight}, long_near_weight={args.long_near_weight}, '
        f'long_second_weight={getattr(args, "long_second_weight", 1.0)}, '
        f'long_late_weight={getattr(args, "long_late_weight", 1.0)}, '
        f'pathsim_topk={args.pathsim_topk}, '
        f'pathsim_topk_ratio={getattr(args, "pathsim_topk_ratio", 0.06)}, '
        f'pathsim_min_topk={getattr(args, "pathsim_min_topk", 2)}, '
        f'pathsim_min_sim={args.pathsim_min_sim}, '
        f'long_hidden_dim={args.long_hidden_dim}, '
        f'mamba_dropout={getattr(args, "mamba_dropout", 0.15)}, '
        f'long_output_drop={getattr(args, "long_output_drop", 0.0)}, '
        f'long_meta_path_drop={getattr(args, "long_meta_path_drop", 0.0)}, '
        f'view_norm={getattr(args, "view_norm", True)}, '
        f'cross_view_residual={getattr(args, "cross_view_residual", True)}, '
        f'short_sample_rate={args.sample_rate_drug}, '
        f'short_degree_penalty={args.short_use_log_degree_penalty}, '
        f'short_output_drop={getattr(args, "short_output_drop", 0.08)}, '
        f'short_self_weight={getattr(args, "short_self_weight", 0.70)}, '
        f'short_bottleneck_dim={getattr(args, "short_bottleneck_dim", 0)}, '
        f'short_relation_drop={getattr(args, "short_relation_drop", 0.0)}, '
        f'short_view_source={getattr(args, "short_view_source", "bridge")}, '
        f'cl_align_mode={args.cl_align_mode}, '
        f'cl_pos_mode={getattr(args, "cl_pos_mode", "self")}, '
        f'view_decorrelation_weight={args.view_decorrelation_weight}, '
        f'smiles_fp={args.smiles_fp_dim}, smiles_radius={args.smiles_radius}'
    )
    print(f'Seeds/fold: {args.seeds_per_fold}, epochs={args.epochs}')
    print('=' * 60)


def run_experiment(run_options):
    dataset = run_options['dataset']
    task = run_options.get('task', 'dfc')
    neg_ratio = run_options['neg_ratio']
    protocol_desc = run_options['protocol_desc']
    meta_mode = run_options.get('meta_mode', 'base')
    neg_strategy = run_options.get('neg_strategy', 'global_random')
    neg_seed = run_options.get('neg_seed', 42)
    food_feature_mode = run_options.get('food_feature_mode', 'random')

    out_suffix = build_output_suffix(meta_mode)
    print_run_header(dataset, task, neg_ratio, meta_mode, neg_strategy, food_feature_mode)

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    raw = load_raw_data(dataset)
    print(f"Data root: {raw.get('data_root_used', 'unknown')}")
    print(f"Food-FC source: {raw.get('ffc_source', 'unknown')}")
    all_edges = raw['dfc_edges'] if task == 'dfc' else raw['df_edges']
    num_right = raw['num_fc'] if task == 'dfc' else raw['num_food']
    target_name = 'D-FC' if task == 'dfc' else 'D-F'
    print(f"Drug: {raw['num_drug']}, Food: {raw['num_food']}, FC: {raw['num_fc']}")
    print(f'{target_name} edges: {len(all_edges)}')

    folds = split_edges_kfold(all_edges, n_folds=args.n_folds, random_state=42)
    negative_folds = build_negative_folds(
        all_edges,
        num_right,
        folds,
        neg_ratio=neg_ratio,
        random_state=neg_seed,
        num_drug=raw['num_drug'],
    )
    quick_folds = int(run_options.get('quick_folds', 0) or 0)
    if quick_folds > 0:
        quick_folds = min(quick_folds, len(folds))
        print(f'Diagnostic quick_folds enabled: running first {quick_folds}/{len(folds)} folds')
        folds = folds[:quick_folds]

    all_results = []
    start = datetime.datetime.now()

    for fold_i, (train_idx, test_idx) in enumerate(folds):
        print(f"\n{'=' * 60}")
        print(f'Fold {fold_i + 1}/{args.n_folds}: {len(train_idx)} train, {len(test_idx)} test positive edges')
        print(f"{'=' * 60}")

        train_neg = merge_negative_folds(negative_folds, fold_i)
        test_neg = negative_folds[fold_i]
        fold_data = prepare_fold_data(raw, train_idx, test_idx, train_neg, test_neg, device, args.num_hops, neg_ratio, meta_mode)
        fold_data['neg_protocol'] = neg_strategy

        print(
            f"  Train: {len(fold_data['train_pos'])} pos / {len(fold_data['train_neg'])} neg, "
            f"Test: {len(fold_data['test_pos'])} pos / {len(fold_data['test_neg'])} neg"
        )
        cov = fold_data.get('smiles_coverage', {})
        if cov:
            print(
                '  Initial features: '
                f"Drug SMILES {cov['drug'][0]}/{cov['drug'][1]}, "
                f"Food {food_feature_mode} {cov['food'][0]}/{cov['food'][1]}, "
                f"FC SMILES {cov['fc'][0]}/{cov['fc'][1]}"
            )

        for seed in args.seeds_per_fold:
            print(f'\n  --- Fold {fold_i + 1}, Seed {seed} ---')
            result = run_single(seed, fold_data, device)
            result_with_meta = dict(result)
            result_with_meta['fold'] = fold_i + 1
            result_with_meta['seed'] = seed
            all_results.append(result_with_meta)
            print(
                f"    AUC:{result['AUC']:.4f} AUPR:{result['AUPR']:.4f} "
                f"F1:{result['F1']:.4f} Recall:{result['Recall']:.4f} "
                f"Precision:{result['Precision']:.4f}"
            )

    summarized = summarize_results(all_results)
    print_result_summary(dataset, neg_ratio, all_results, summarized)
    runtime_seconds = int((datetime.datetime.now() - start).seconds)
    print(f'\nTime: {runtime_seconds}s')

    save_results(
        version_dir=os.path.dirname(os.path.abspath(__file__)),
        dataset=dataset,
        neg_ratio=neg_ratio,
        protocol_desc=protocol_desc,
        raw=raw,
        all_edges=all_edges,
        all_results=all_results,
        summarized=summarized,
        runtime_seconds=runtime_seconds,
        out_suffix=out_suffix,
    )
