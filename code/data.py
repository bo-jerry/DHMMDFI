import os

import sys

import re

import random

import hashlib

import json

from collections import defaultdict



import numpy as np

import pandas as pd

import scipy.sparse as sp

import torch
import torch as th
from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import KFold



sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))





def task_data_dir_name(dataset, task=None):
    task = task or getattr(args, 'task', 'dfc')
    prefix = 'DrugBank' if str(dataset).lower() == 'drugbank' else 'PubMed'
    suffix = 'DFC' if str(task).lower() == 'dfc' else 'DF'
    return f'{prefix}-{suffix}'
from config import args



try:

    from rdkit import Chem, DataStructs, RDLogger

    from rdkit.Chem import AllChem, rdFingerprintGenerator

    RDLogger.DisableLog('rdApp.*')

    RDKIT_AVAILABLE = True

except Exception:

    Chem = None

    DataStructs = None

    AllChem = None

    rdFingerprintGenerator = None

    RDKIT_AVAILABLE = False





SMILES_LOOKUP_CACHE = {}

STRUCTURAL_FEATURE_MATRIX_CACHE = {}

FOOD_FEATURE_CACHE = {}

MORGAN_GENERATOR_CACHE = {}





def first_existing_path(*paths):

    for path in paths:

        if os.path.exists(path):

            return path

    return paths[0] if paths else None





def clean_name(name):

    if pd.isna(name):

        return None

    return re.sub(r'\s*\(https?://[^)]+\)', '', str(name).strip()).strip()





def build_hashed_smiles_fingerprint(smiles, fp_dim=512):



    smiles = str(smiles).strip()

    vec = np.zeros((fp_dim,), dtype=np.float32)

    if not smiles:

        return vec

    grams = set()

    max_n = min(4, len(smiles))

    for n in range(1, max_n + 1):

        for i in range(len(smiles) - n + 1):

            grams.add(smiles[i:i + n])

    for gram in grams:

        digest = hashlib.blake2b(gram.encode('utf-8'), digest_size=8).digest()

        idx = int.from_bytes(digest, byteorder='little', signed=False) % fp_dim

        vec[idx] = 1.0

    return vec





def get_morgan_generator(fp_dim=512, radius=2):

    cache_key = (int(fp_dim), int(radius))

    if cache_key not in MORGAN_GENERATOR_CACHE:

        MORGAN_GENERATOR_CACHE[cache_key] = rdFingerprintGenerator.GetMorganGenerator(

            radius=radius,

            fpSize=fp_dim,

        )

    return MORGAN_GENERATOR_CACHE[cache_key]





def load_smiles_feature_matrix(dataset, node_kind, num_nodes, fp_dim=512, radius=2):



    if node_kind == 'food':

        return np.zeros((num_nodes, fp_dim), dtype=np.float32)



    data_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data'))

    ds_dir = 'DrugBank-DFI' if dataset == 'drugbank' else 'PubMed-DFI'

    smiles_file = first_existing_path(

        os.path.join(data_root, ds_dir, 'smiles', f'{node_kind}_smiles.csv'),

        os.path.join(data_root, ds_dir, 'smiles', f'{node_kind}_smiles.xlsx'),

    )

    if not os.path.exists(smiles_file):

        return np.zeros((num_nodes, fp_dim), dtype=np.float32)



    df = pd.read_excel(smiles_file) if smiles_file.endswith('.xlsx') else pd.read_csv(smiles_file)

    fp_mat = np.zeros((num_nodes, fp_dim), dtype=np.float32)

    if 'match_status' in df.columns:

        df = df[df['match_status'].fillna('') == 'matched'].copy()

    if 'node_id' not in df.columns or 'smiles' not in df.columns:

        return fp_mat



    for _, row in df.iterrows():

        try:

            node_id = int(row['node_id'])

        except Exception:

            continue

        if node_id < 0 or node_id >= num_nodes:

            continue

        smiles = str(row['smiles']).strip() if pd.notna(row['smiles']) else ''

        if not smiles:

            continue

        if RDKIT_AVAILABLE:

            mol = Chem.MolFromSmiles(smiles)

            if mol is None:

                continue

            generator = get_morgan_generator(fp_dim=fp_dim, radius=radius)

            bitvect = generator.GetFingerprint(mol)

            arr = np.zeros((fp_dim,), dtype=np.float32)

            DataStructs.ConvertToNumpyArray(bitvect, arr)

            fp_mat[node_id] = arr

        else:

            fp_mat[node_id] = build_hashed_smiles_fingerprint(smiles, fp_dim=fp_dim)

    return fp_mat





def normalize_lookup_key(value):

    if pd.isna(value):

        return ''

    text = str(value).strip().lower().replace('\xa0', ' ')

    return re.sub(r'\s+', ' ', text)





def smiles_to_fingerprint(smiles, fp_dim=512, radius=2):

    smiles = str(smiles).strip() if smiles is not None else ''

    if not smiles:

        return np.zeros((fp_dim,), dtype=np.float32)

    if RDKIT_AVAILABLE:

        mol = Chem.MolFromSmiles(smiles)

        if mol is None:

            return np.zeros((fp_dim,), dtype=np.float32)

        generator = get_morgan_generator(fp_dim=fp_dim, radius=radius)

        bitvect = generator.GetFingerprint(mol)

        arr = np.zeros((fp_dim,), dtype=np.float32)

        DataStructs.ConvertToNumpyArray(bitvect, arr)

        return arr.astype(np.float32)

    return build_hashed_smiles_fingerprint(smiles, fp_dim=fp_dim)





def load_drugbank_smiles_lookup(data_root, fp_dim=512, radius=2):

    cache_key = ('drugbank', fp_dim, radius)

    if cache_key in SMILES_LOOKUP_CACHE:

        return SMILES_LOOKUP_CACHE[cache_key]



    smiles_dir = os.path.join(data_root, 'DrugBank-DFI', 'smiles')

    drug_path = first_existing_path(

        os.path.join(smiles_dir, 'drug_smiles.xlsx'),

        os.path.join(smiles_dir, 'Drug SMILES.xlsx'),

    )

    fc_path = first_existing_path(

        os.path.join(smiles_dir, 'fc_smiles.xlsx'),

        os.path.join(smiles_dir, 'Food-SMILES(DrugBank-dfc).xlsx'),

    )

    drug_lookup, fc_lookup = {}, {}



    if os.path.exists(drug_path):

        df = pd.read_excel(drug_path)

        for _, row in df.iterrows():

            smiles = row.get('SMILES')

            if pd.isna(smiles) or not str(smiles).strip():

                continue

            fp = smiles_to_fingerprint(smiles, fp_dim=fp_dim, radius=radius)

            drug_lookup[normalize_lookup_key(row.get('DrugBank ID'))] = fp

            drug_lookup[normalize_lookup_key(row.get('Drug Name'))] = fp



    if os.path.exists(fc_path):

        df = pd.read_excel(fc_path)

        for _, row in df.iterrows():

            smiles = row.get('SMILES')

            if pd.isna(smiles) or not str(smiles).strip():

                continue

            fc_lookup[normalize_lookup_key(row.get('Food Name'))] = smiles_to_fingerprint(

                smiles, fp_dim=fp_dim, radius=radius

            )



    SMILES_LOOKUP_CACHE[cache_key] = {'drug': drug_lookup, 'fc': fc_lookup}

    return SMILES_LOOKUP_CACHE[cache_key]





def load_pubmed_smiles_lookup(data_root, fp_dim=512, radius=2):

    cache_key = ('pubmed', fp_dim, radius)

    if cache_key in SMILES_LOOKUP_CACHE:

        return SMILES_LOOKUP_CACHE[cache_key]



    smiles_dir = os.path.join(data_root, 'PubMed-DFI', 'smiles')

    drug_path = first_existing_path(

        os.path.join(smiles_dir, 'drug_smiles.xlsx'),

        os.path.join(smiles_dir, 'Drug SMILES.xlsx'),

    )

    fc_table_path = first_existing_path(

        os.path.join(smiles_dir, 'fc_smiles.csv'),

        os.path.join(smiles_dir, 'fc_smiles.xlsx'),

    )

    fc_json_path = first_existing_path(

        os.path.join(smiles_dir, 'foodb_processed_compound_source.jsonl'),

        os.path.join(smiles_dir, 'Processed_Compound.json'),

    )

    drug_lookup, fc_lookup = {}, {}



    if os.path.exists(drug_path):

        df = pd.read_excel(drug_path)

        for _, row in df.iterrows():

            smiles = row.get('SMILES')

            if pd.isna(smiles) or not str(smiles).strip():

                continue

            fp = smiles_to_fingerprint(smiles, fp_dim=fp_dim, radius=radius)

            drug_lookup[normalize_lookup_key(row.get('DrugBank ID'))] = fp

            drug_lookup[normalize_lookup_key(row.get('Drug Name'))] = fp



    if os.path.exists(fc_table_path):

        df = pd.read_excel(fc_table_path) if fc_table_path.endswith('.xlsx') else pd.read_csv(fc_table_path)

        name_col = 'fc_name' if 'fc_name' in df.columns else 'name'

        smiles_col = 'smiles' if 'smiles' in df.columns else 'moldb_smiles'

        if name_col in df.columns and smiles_col in df.columns:

            for _, row in df.iterrows():

                name = normalize_lookup_key(row.get(name_col))

                smiles = row.get(smiles_col)

                if name and pd.notna(smiles) and str(smiles).strip():

                    fc_lookup[name] = smiles_to_fingerprint(smiles, fp_dim=fp_dim, radius=radius)

    elif os.path.exists(fc_json_path):

        with open(fc_json_path, 'r', encoding='utf-8') as file:

            for line in file:

                line = line.strip()

                if not line:

                    continue

                try:

                    record = json.loads(line)

                except json.JSONDecodeError:

                    continue

                name = normalize_lookup_key(record.get('name'))

                smiles = record.get('moldb_smiles')

                if name and smiles:

                    fc_lookup[name] = smiles_to_fingerprint(smiles, fp_dim=fp_dim, radius=radius)



    SMILES_LOOKUP_CACHE[cache_key] = {'drug': drug_lookup, 'fc': fc_lookup}

    return SMILES_LOOKUP_CACHE[cache_key]





def load_structural_feature_matrix(raw, node_kind, num_nodes, fp_dim=512, radius=2):

    pre_dir = raw.get('preprocessed_dir')
    precomputed_path = os.path.join(pre_dir, f'{node_kind}_features.npy') if pre_dir else None
    if precomputed_path and os.path.exists(precomputed_path):
        features = np.asarray(np.load(precomputed_path), dtype=np.float32)
        if features.shape == (int(num_nodes), int(fp_dim)):
            return features

    if node_kind == 'food':

        return np.zeros((num_nodes, fp_dim), dtype=np.float32)



    data_root = raw.get('data_root_used') or os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data'))

    id_to_name = raw.get(f'id2{node_kind}', {})

    cache_key = (

        raw['dataset'],

        node_kind,

        int(num_nodes),

        int(fp_dim),

        int(radius),

        os.path.abspath(data_root),

        tuple(id_to_name.get(i, '') for i in range(num_nodes)),

    )

    if cache_key in STRUCTURAL_FEATURE_MATRIX_CACHE:

        return STRUCTURAL_FEATURE_MATRIX_CACHE[cache_key]



    if raw['dataset'] == 'drugbank':

        lookup = load_drugbank_smiles_lookup(data_root, fp_dim=fp_dim, radius=radius)[node_kind]

    else:

        lookup = load_pubmed_smiles_lookup(data_root, fp_dim=fp_dim, radius=radius)[node_kind]



    fp_mat = np.zeros((num_nodes, fp_dim), dtype=np.float32)

    for node_id in range(num_nodes):

        key = normalize_lookup_key(id_to_name.get(node_id, ''))

        if not key:

            continue

        candidates = [key]

        if str(raw['dataset']).startswith('pubmed') and node_kind == 'fc':

            candidates.extend([f'{key} acid', f'{key} compound'])

        for candidate in candidates:

            if candidate in lookup:

                fp_mat[node_id] = lookup[candidate]

                break

    STRUCTURAL_FEATURE_MATRIX_CACHE[cache_key] = fp_mat

    return fp_mat





def load_precomputed_food_features(raw, num_food, feat_dim):
    pre_dir = raw.get('preprocessed_dir')
    path = os.path.join(pre_dir, 'food_features.npy') if pre_dir else None
    if path and os.path.exists(path):
        features = np.asarray(np.load(path), dtype=np.float32)
        if features.shape == (int(num_food), int(feat_dim)):
            return features
    return build_random_food_features(num_food, feat_dim, raw['dataset'])


def build_random_food_features(num_food, feat_dim, dataset, seed=2026):



    cache_key = (dataset, int(num_food), int(feat_dim), int(seed))

    if cache_key in FOOD_FEATURE_CACHE:

        return FOOD_FEATURE_CACHE[cache_key]

    dataset_offset = 17 if dataset == 'drugbank' else 31

    rng = np.random.RandomState(seed + dataset_offset + int(num_food))

    feat = rng.uniform(0.0, 1.0, size=(num_food, feat_dim)).astype(np.float32)

    FOOD_FEATURE_CACHE[cache_key] = feat

    return feat


def preprocess_dense_features(feat):

    feat = feat.astype(np.float32)

    rowsum = feat.sum(axis=1, keepdims=True)

    norm_feat = np.zeros_like(feat, dtype=np.float32)

    np.divide(feat, rowsum, out=norm_feat, where=rowsum > 0)

    return norm_feat





def count_nonzero_rows(feat):

    return int(np.count_nonzero(np.asarray(feat).sum(axis=1)))





def build_neighbor_index(adj_matrix, num_nodes):

    adj_csr = adj_matrix.tocsr()

    nei_list = []

    for i in range(num_nodes):

        nbs = adj_csr[i].indices.tolist()

        nei_list.append(np.array(nbs if nbs else [0]))

    return np.array(nei_list, dtype=object)





def normalize_adj_row(adj):

    adj = sp.csr_matrix(adj, dtype=np.float32)

    rowsum = np.array(adj.sum(1)).flatten()

    d_inv = np.zeros_like(rowsum, dtype=np.float32)

    np.divide(1.0, rowsum, out=d_inv, where=rowsum > 0)

    return sp.diags(d_inv, format='csr', dtype=np.float32) @ adj





def binarize_without_self_loops(adj):

    adj = sp.csr_matrix(adj, dtype=np.float32)

    diag = adj.diagonal()

    if np.any(diag):

        adj = adj - sp.diags(diag, offsets=0, shape=adj.shape, format='csr', dtype=np.float32)

        adj.eliminate_zeros()

    return (adj > 0).astype(np.float32).tocsr()


def pathsim_prune_meta_graph(path_count_mat, topk=40, min_sim=0.0, topk_ratio=0.0, min_topk=1):

    mat = sp.csr_matrix(path_count_mat, dtype=np.float32)
    n = mat.shape[0]
    diag = np.asarray(mat.diagonal(), dtype=np.float32)
    if n == 0:
        return mat
    mat = mat.tolil()
    mat.setdiag(0)
    mat = mat.tocsr()
    mat.eliminate_zeros()

    if int(topk) <= 0:
        return binarize_without_self_loops(mat)

    effective_topk = int(topk)
    if float(topk_ratio) > 0:
        ratio_k = int(np.ceil(float(topk_ratio) * float(n)))
        ratio_k = max(int(min_topk), ratio_k)
        effective_topk = max(1, min(effective_topk, ratio_k))

    rows, cols, vals = [], [], []
    for i in range(n):
        start, end = mat.indptr[i], mat.indptr[i + 1]
        neigh = mat.indices[start:end]
        counts = mat.data[start:end]
        if neigh.size == 0:
            continue
        denom = diag[i] + diag[neigh]
        valid = denom > 0
        if not np.any(valid):
            continue
        neigh = neigh[valid]
        scores = (2.0 * counts[valid]) / denom[valid]
        if float(min_sim) > 0:
            keep = scores >= float(min_sim)
            neigh = neigh[keep]
            scores = scores[keep]
        if neigh.size == 0:
            continue
        k = min(effective_topk, neigh.size)
        if neigh.size > k:
            order = np.argpartition(-scores, k - 1)[:k]
            order = order[np.argsort(-scores[order], kind='mergesort')]
        else:
            order = np.argsort(-scores, kind='mergesort')
        picked = neigh[order]
        picked_scores = scores[order]
        rows.extend([i] * len(picked))
        cols.extend(picked.tolist())
        vals.extend(picked_scores.tolist())

    pruned = sp.csr_matrix((vals, (rows, cols)), shape=mat.shape, dtype=np.float32)
    pruned = pruned.maximum(pruned.T)
    pruned.setdiag(0)
    pruned.eliminate_zeros()
    return pruned


def raw_count_prune_meta_graph(path_count_mat, topk=40, topk_ratio=0.0, min_topk=1,
                               binary=False):

    mat = sp.csr_matrix(path_count_mat, dtype=np.float32)
    n = mat.shape[0]
    if n == 0:
        return mat
    mat = mat.tolil()
    mat.setdiag(0)
    mat = mat.tocsr()
    mat.eliminate_zeros()

    if int(topk) <= 0:
        return binarize_without_self_loops(mat) if binary else mat

    effective_topk = int(topk)
    if float(topk_ratio) > 0:
        ratio_k = max(int(min_topk), int(np.ceil(float(topk_ratio) * float(n))))
        effective_topk = max(1, min(effective_topk, ratio_k))

    rows, cols, vals = [], [], []
    for i in range(n):
        start, end = mat.indptr[i], mat.indptr[i + 1]
        neigh = mat.indices[start:end]
        counts = mat.data[start:end]
        if neigh.size == 0:
            continue
        k = min(effective_topk, neigh.size)
        if neigh.size > k:
            order = np.argpartition(-counts, k - 1)[:k]
            order = order[np.argsort(-counts[order], kind='mergesort')]
        else:
            order = np.argsort(-counts, kind='mergesort')
        picked = neigh[order]
        picked_counts = counts[order]
        rows.extend([i] * len(picked))
        cols.extend(picked.tolist())
        vals.extend([1.0] * len(picked) if binary else picked_counts.tolist())

    pruned = sp.csr_matrix((vals, (rows, cols)), shape=mat.shape, dtype=np.float32)
    pruned = pruned.maximum(pruned.T)
    pruned.setdiag(0)
    pruned.eliminate_zeros()
    return pruned


def add_self_loops_binary(adj):

    adj = sp.csr_matrix(adj, dtype=np.float32)

    eye = sp.eye(adj.shape[0], format='csr', dtype=np.float32)

    return ((adj + eye) > 0).astype(np.float32).tocsr()


def identity_contrast_pos(num_nodes):

    return sp.eye(num_nodes, format='csr', dtype=np.float32)





def extract_multi_hop(adj_list, feat, num_hops):

    result = []

    for adj in adj_list:

        hops = [feat]

        adj_n = normalize_adj_row(adj)

        cur = feat

        for _ in range(1, num_hops):

            cur = adj_n @ cur

            hops.append(cur)

        result.append(np.stack(hops, axis=1).astype(np.float32))

    return result



def apply_long_hop_mode(
    multi_hop_features,
    self_weight=0.05,
    near_weight=0.0,
    second_weight=1.0,
    late_weight=1.0,
):

    filtered = []
    for features in multi_hop_features:
        masked = np.array(features, copy=True)
        if masked.shape[1] > 0:
            masked[:, 0, :] *= float(self_weight)
        if masked.shape[1] > 1:
            masked[:, 1, :] *= float(near_weight)
        if masked.shape[1] > 2:
            masked[:, 2, :] *= float(second_weight)
        if masked.shape[1] > 3:
            masked[:, 3:, :] *= float(late_weight)
        filtered.append(masked.astype(np.float32))
    return filtered





def attenuate_small_type_long_features(multi_hop_features, num_nodes, threshold=0, floor=1.0):

    threshold = int(threshold)
    if threshold <= 0 or int(num_nodes) >= threshold:
        return multi_hop_features, 1.0
    floor = float(np.clip(float(floor), 0.0, 1.0))
    scale = max(floor, float(num_nodes) / float(threshold))
    scaled = [np.asarray(features, dtype=np.float32) * scale for features in multi_hop_features]
    return scaled, float(scale)


def split_edges_kfold(edges, n_folds=5, random_state=42):

    edges = np.asarray(edges, dtype=np.int64)
    indices = np.arange(len(edges))
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    return [
        (train_idx.astype(np.int64).tolist(), test_idx.astype(np.int64).tolist())
        for train_idx, test_idx in kfold.split(indices)
    ]


def build_negative_folds(all_pos_edges, num_fc, positive_folds, neg_ratio=1, random_state=42,
                         num_drug=None):

    if int(neg_ratio) < 1:
        raise ValueError('neg_ratio must be >= 1')

    all_edges = np.asarray(all_pos_edges, dtype=np.int64)
    rng = np.random.RandomState(random_state)
    pos_set = set((int(d), int(fc)) for d, fc in all_edges)
    if num_drug is None:
        num_drug = int(all_edges[:, 0].max()) + 1

    all_candidates = [
        (drug_id, fc_id)
        for drug_id in range(int(num_drug))
        for fc_id in range(int(num_fc))
        if (drug_id, fc_id) not in pos_set
    ]
    total_needed = sum(len(test_idx) * int(neg_ratio) for _, test_idx in positive_folds)
    if len(all_candidates) < total_needed:
        raise ValueError('Global unobserved D-FC pool is too small for disjoint negative folds')

    return build_global_random_negative_folds(
        all_candidates=np.asarray(all_candidates, dtype=np.int64),
        positive_folds=positive_folds,
        neg_ratio=int(neg_ratio),
        rng=rng,
    )


def build_global_random_negative_folds(all_candidates, positive_folds, neg_ratio, rng, excluded=None):

    excluded = excluded or set()
    available = [idx for idx in range(len(all_candidates)) if idx not in excluded]
    total_needed = sum(len(test_idx) * int(neg_ratio) for _, test_idx in positive_folds)
    if len(available) < total_needed:
        raise ValueError('Not enough unused global-random negative candidates')
    chosen_order = rng.permutation(np.asarray(available, dtype=np.int64))[:total_needed]
    negative_folds = []
    offset = 0
    for _, test_idx in positive_folds:
        need = len(test_idx) * int(neg_ratio)
        chosen = all_candidates[chosen_order[offset:offset + need]]
        offset += need
        negative_folds.append(np.asarray(chosen, dtype=np.int64))
    return negative_folds


def merge_negative_folds(negative_folds, held_out_fold):



    blocks = [negative_folds[i] for i in range(len(negative_folds)) if i != held_out_fold]

    blocks = [block for block in blocks if len(block) > 0]

    return np.vstack(blocks) if blocks else np.zeros((0, 2), dtype=np.int64)



def load_raw_data(dataset='drugbank'):

    data_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))

    ds_dir = task_data_dir_name(dataset)



    if ds_dir is not None:

        if dataset == 'drugbank':

            pre_candidates = ['preprocessed_strict_dfc', 'preprocessed_strict_clean', 'preprocessed_strict_union', 'preprocessed']

        elif dataset == 'pubmed_hq':

            pre_candidates = ['preprocessed_high_quality']

        elif dataset == 'pubmed_complement_hq':

            pre_candidates = ['preprocessed_pubmed_complement_hq']

        elif dataset == 'pubmed_bridge_hard':

            pre_candidates = ['preprocessed_pubmed_bridge_hard']

        elif dataset == 'pubmed_bridge_hard_lowhub':

            pre_candidates = ['preprocessed_pubmed_bridge_hard_lowhub']

        else:

            pre_candidates = ['preprocessed_strict_clean', 'preprocessed_strict_dfc', 'preprocessed']

        pre_dir = os.path.join(data_root, ds_dir)

        required = ['num_nodes.npy', 'df_edges.npy', 'dfc_edges.npy', 'ffc_edges.npy']

        if all(os.path.exists(os.path.join(pre_dir, name)) for name in required):

            num_nodes = np.load(os.path.join(pre_dir, 'num_nodes.npy'))

            df_el = [tuple(x) for x in np.load(os.path.join(pre_dir, 'df_edges.npy')).tolist()]

            dfc_el = [tuple(x) for x in np.load(os.path.join(pre_dir, 'dfc_edges.npy')).tolist()]

            ffc_el = [tuple(x) for x in np.load(os.path.join(pre_dir, 'ffc_edges.npy')).tolist()]

            mappings = {}

            mappings_path = os.path.join(pre_dir, 'mappings.pkl')

            if os.path.exists(mappings_path):

                import pickle

                with open(mappings_path, 'rb') as file:

                    mappings = pickle.load(file)

            raw = {

                'num_drug': int(num_nodes[0]), 'num_food': int(num_nodes[1]), 'num_fc': int(num_nodes[2]),

                'df_edges': df_el, 'dfc_edges': dfc_el, 'ffc_edges': ffc_el, 'dataset': dataset,

                'data_root_used': pre_dir, 'preprocessed_dir': pre_dir,
                'ffc_source': os.path.join(pre_dir, 'ffc_edges.npy'),

            }

            raw.update({

                key: mappings.get(key, {})

                for key in ['drug2id', 'food2id', 'fc2id', 'id2drug', 'id2food', 'id2fc']

            })

            return raw



    if dataset == 'drugbank':

        data_dir = os.path.join(data_root, 'DrugBank-DFI')

        df_relation_path = os.path.join(data_dir, 'drugbank_drug_food.xlsx') if os.path.exists(os.path.join(data_dir, 'drugbank_drug_food.xlsx')) else os.path.join(data_dir, 'DFRelation.xlsx')

        df_constituent_path = os.path.join(data_dir, 'drugbank_drug_fc.xlsx') if os.path.exists(os.path.join(data_dir, 'drugbank_drug_fc.xlsx')) else os.path.join(data_dir, 'drug_food constituent.xlsx')

        food_component_csv = os.path.join(data_dir, 'drugbank_food_fc.csv')

        food_component_xlsx = os.path.join(data_dir, 'food_component.xlsx')

        df_relation = pd.read_excel(df_relation_path, sheet_name=0)

        df_constituent = pd.read_excel(df_constituent_path, sheet_name=0)

        use_food_component_csv = os.path.exists(food_component_csv)

        food_component = pd.read_csv(food_component_csv) if use_food_component_csv else pd.read_excel(food_component_xlsx, sheet_name=0)

        for df in [df_relation, df_constituent, food_component]:

            df.columns = [str(c).strip() for c in df.columns]



        df_edges = df_relation[['DrugBank ID', 'Food Name']].copy()

        df_edges.columns = ['drug', 'food']

        df_edges['drug'] = df_edges['drug'].astype(str).str.strip()

        df_edges['food'] = df_edges['food'].apply(clean_name)



        dfc_edges = df_constituent[['DrugBank ID', 'Food Constituent']].copy()

        dfc_edges.columns = ['drug', 'fc']

        dfc_edges['drug'] = dfc_edges['drug'].astype(str).str.strip()

        dfc_edges['fc'] = dfc_edges['fc'].astype(str).str.strip()



        if use_food_component_csv:

            ffc_edges = food_component[['food', 'food_compound']].copy()

        else:

            fc_col = [c for c in food_component.columns if 'food' in c.lower() or 'Food' in c][0]

            cc_col = [c for c in food_component.columns if 'component' in c.lower() or 'Component' in c][0]

            ffc_edges = food_component[[fc_col, cc_col]].copy()

        ffc_edges.columns = ['food', 'fc']

        ffc_edges['food'] = ffc_edges['food'].astype(str).str.strip()

        ffc_edges['fc'] = ffc_edges['fc'].astype(str).str.strip()

    elif str(dataset).startswith('pubmed'):

        data_dir = os.path.join(data_root, 'PubMed-DFI')

        df_raw = pd.read_csv(os.path.join(data_dir, 'pubmed_drug_food.csv'))

        dfc_raw = pd.read_csv(os.path.join(data_dir, 'pubmed_drug_fc.csv'))

        ffc_csv = os.path.join(data_dir, 'pubmed_food_fc.csv')

        ffc_xls = os.path.join(data_dir, 'pubmed_food_fc.xls')

        if os.path.exists(ffc_csv):

            ffc_raw = pd.read_csv(ffc_csv)

            ffc_source = ffc_csv

        else:

            ffc_raw = pd.read_excel(ffc_xls, sheet_name=0)

            ffc_source = ffc_xls



        df_edges = df_raw[['drug', 'food']].copy()

        df_edges['drug'] = df_edges['drug'].astype(str).str.strip()

        df_edges['food'] = df_edges['food'].astype(str).str.strip()



        dfc_edges = dfc_raw[['drug', 'food_compound']].copy()

        dfc_edges.columns = ['drug', 'fc']

        dfc_edges['drug'] = dfc_edges['drug'].astype(str).str.strip()

        dfc_edges['fc'] = dfc_edges['fc'].astype(str).str.strip()



        ffc_edges = ffc_raw[['food', 'food_compound']].copy()

        ffc_edges.columns = ['food', 'fc']

        ffc_edges['food'] = ffc_edges['food'].astype(str).str.strip()

        ffc_edges['fc'] = ffc_edges['fc'].astype(str).str.strip()

    else:

        raise ValueError(f'Unknown dataset: {dataset}')



    df_edges = df_edges.dropna().drop_duplicates()

    dfc_edges = dfc_edges.dropna().drop_duplicates()

    ffc_edges = ffc_edges.dropna().drop_duplicates()



    all_drugs = sorted(set(df_edges['drug'].unique()) | set(dfc_edges['drug'].unique()))

    all_foods = sorted(set(df_edges['food'].unique()) | set(ffc_edges['food'].unique()))

    all_fcs = sorted(set(dfc_edges['fc'].unique()) | set(ffc_edges['fc'].unique()))

    drug2id = {d: i for i, d in enumerate(all_drugs)}

    food2id = {f: i for i, f in enumerate(all_foods)}

    fc2id = {c: i for i, c in enumerate(all_fcs)}



    df_el = [(drug2id[r['drug']], food2id[r['food']]) for _, r in df_edges.iterrows() if r['drug'] in drug2id and r['food'] in food2id]

    dfc_el = [(drug2id[r['drug']], fc2id[r['fc']]) for _, r in dfc_edges.iterrows() if r['drug'] in drug2id and r['fc'] in fc2id]

    ffc_el = [(food2id[r['food']], fc2id[r['fc']]) for _, r in ffc_edges.iterrows() if r['food'] in food2id and r['fc'] in fc2id]



    return {

        'num_drug': len(all_drugs), 'num_food': len(all_foods), 'num_fc': len(all_fcs),

        'df_edges': list(set(df_el)), 'dfc_edges': list(set(dfc_el)), 'ffc_edges': list(set(ffc_el)),

        'dataset': dataset, 'data_root_used': data_root,

        'ffc_source': ffc_source if str(dataset).startswith('pubmed') else (food_component_csv if os.path.exists(food_component_csv) else food_component_xlsx),

        'drug2id': drug2id, 'food2id': food2id, 'fc2id': fc2id,

        'id2drug': {i: name for name, i in drug2id.items()},

        'id2food': {i: name for name, i in food2id.items()},

        'id2fc': {i: name for name, i in fc2id.items()},

    }





def prepare_fold_data(raw, train_val_idx, test_idx, train_neg_edges, test_neg_edges, device, num_hops=3, neg_ratio=1, meta_mode='base'):



    task = getattr(args, 'task', 'dfc')
    if task not in ('dfc', 'df'):
        raise ValueError(f"Unsupported task: {task}")

    nd, nf, nfc = raw['num_drug'], raw['num_food'], raw['num_fc']

    all_dfc = np.array(raw['dfc_edges'], dtype=np.int64)
    all_df = np.array(raw['df_edges'], dtype=np.int64)
    all_target = all_dfc if task == 'dfc' else all_df
    target_type = 'fc' if task == 'dfc' else 'food'
    num_right = nfc if task == 'dfc' else nf

    df_edges = raw['df_edges']

    ffc_edges = raw['ffc_edges']



    train_pos = all_target[train_val_idx].copy()

    test_pos = all_target[test_idx].copy()

    train_neg = np.asarray(train_neg_edges, dtype=np.int64).copy()

    test_neg = np.asarray(test_neg_edges, dtype=np.int64).copy()



    rng = np.random.RandomState(42 + int(np.sum(test_idx) % 100000))

    if len(train_pos) > 0:

        train_pos = train_pos[rng.permutation(len(train_pos))]

    if len(train_neg) > 0:

        train_neg = train_neg[rng.permutation(len(train_neg))]



    val_pos = np.zeros((0, 2), dtype=np.int64)

    val_neg = np.zeros((0, 2), dtype=np.int64)



    dr, dc = zip(*df_edges)

    df_adj_full = sp.csr_matrix((np.ones(len(df_edges)), (dr, dc)), shape=(nd, nf))

    if task == 'df':
        df_adj_tr = sp.csr_matrix((np.ones(len(train_pos)), (train_pos[:, 0], train_pos[:, 1])), shape=(nd, nf))
        df_adj = df_adj_tr
        dfc_adj_tr = sp.csr_matrix((np.ones(len(all_dfc)), (all_dfc[:, 0], all_dfc[:, 1])), shape=(nd, nfc))
    else:
        df_adj = df_adj_full
        dfc_adj_tr = sp.csr_matrix((np.ones(len(train_pos)), (train_pos[:, 0], train_pos[:, 1])), shape=(nd, nfc))

    fr, fcc = zip(*ffc_edges)

    ffc_adj = sp.csr_matrix((np.ones(len(ffc_edges)), (fr, fcc)), shape=(nf, nfc))



    pathsim_topk = int(getattr(args, 'pathsim_topk', 4))
    pathsim_topk_ratio = float(getattr(args, 'pathsim_topk_ratio', 0.0))
    pathsim_min_topk = int(getattr(args, 'pathsim_min_topk', 1))
    pathsim_min_sim = float(getattr(args, 'pathsim_min_sim', 0.0))
    pathsim_mode = str(getattr(args, 'pathsim_mode', 'pathsim'))

    def make_long_meta(raw_meta):
        if pathsim_mode == 'raw_binary':
            return raw_count_prune_meta_graph(
                raw_meta,
                topk=pathsim_topk,
                topk_ratio=pathsim_topk_ratio,
                min_topk=pathsim_min_topk,
                binary=True,
            )
        return pathsim_prune_meta_graph(
            raw_meta,
            topk=pathsim_topk,
            min_sim=pathsim_min_sim,
            topk_ratio=pathsim_topk_ratio,
            min_topk=pathsim_min_topk,
        )

    dfd_raw = df_adj @ df_adj.T
    dfcd_raw = dfc_adj_tr @ dfc_adj_tr.T
    fcdfc_raw = dfc_adj_tr.T @ dfc_adj_tr
    fcffc_raw = ffc_adj.T @ ffc_adj
    fdf_raw = df_adj.T @ df_adj
    ffcf_raw = ffc_adj @ ffc_adj.T

    dfd = make_long_meta(dfd_raw)
    dfcd = make_long_meta(dfcd_raw)
    fcdfc = make_long_meta(fcdfc_raw)
    fcffc = make_long_meta(fcffc_raw)
    fdf = make_long_meta(fdf_raw)
    ffcf = make_long_meta(ffcf_raw)



    long_view_mode = getattr(args, 'long_view_mode', 'symmetric_long')
    if long_view_mode not in ('base_pair', 'symmetric_long', 'semantic_only', 'full'):
        raise ValueError(f"Unsupported long_view_mode: {long_view_mode}")

    if long_view_mode in ('semantic_only', 'symmetric_long'):
        if task == 'dfc':




            drug_meta = [dfd]
            fc_meta = [fcffc]
            food_meta = [fdf, ffcf]
            if long_view_mode == 'symmetric_long':





                drug_meta.append(make_long_meta(df_adj @ ffc_adj @ ffc_adj.T @ df_adj.T))
                fc_meta.append(make_long_meta(ffc_adj.T @ df_adj.T @ df_adj @ ffc_adj))
                food_meta.append(make_long_meta(df_adj.T @ dfc_adj_tr @ dfc_adj_tr.T @ df_adj))
        else:


            drug_meta = [dfcd]
            fc_meta = [fcdfc, fcffc]
            food_meta = [ffcf]
            if long_view_mode == 'symmetric_long':





                drug_meta.append(make_long_meta(dfc_adj_tr @ ffc_adj.T @ ffc_adj @ dfc_adj_tr.T))
                food_meta.append(make_long_meta(ffc_adj @ dfc_adj_tr.T @ dfc_adj_tr @ ffc_adj.T))
    elif long_view_mode == 'base_pair':
        drug_meta = [dfd, dfcd]
        fc_meta = [fcdfc, fcffc]
        food_meta = [fdf, ffcf]
    else:
        drug_meta = [dfd, dfcd]
        fc_meta = [fcdfc, fcffc]
        food_meta = [fdf, ffcf]



        drug_meta.append(make_long_meta(df_adj @ ffc_adj @ ffc_adj.T @ df_adj.T))

        fc_meta.append(make_long_meta(dfc_adj_tr.T @ df_adj @ df_adj.T @ dfc_adj_tr))

        food_meta.append(make_long_meta(df_adj.T @ dfc_adj_tr @ dfc_adj_tr.T @ df_adj))

        food_meta.append(make_long_meta(ffc_adj @ dfc_adj_tr.T @ dfc_adj_tr @ ffc_adj.T))





    smiles_fp_dim = 512

    smiles_radius = 2

    drug_feat = load_structural_feature_matrix(raw, 'drug', nd, fp_dim=smiles_fp_dim, radius=smiles_radius)

    fc_feat = load_structural_feature_matrix(raw, 'fc', nfc, fp_dim=smiles_fp_dim, radius=smiles_radius)

    food_feat = load_precomputed_food_features(raw, nf, smiles_fp_dim)

    drug_mh = extract_multi_hop(drug_meta, drug_feat, num_hops)

    fc_mh = extract_multi_hop(fc_meta, fc_feat, num_hops)

    food_mh = extract_multi_hop(food_meta, food_feat, num_hops)

    long_self_weight = getattr(args, 'long_self_weight', 0.05)
    long_near_weight = getattr(args, 'long_near_weight', 0.0)
    long_second_weight = getattr(args, 'long_second_weight', 1.0)
    long_late_weight = getattr(args, 'long_late_weight', 1.0)
    drug_mh = apply_long_hop_mode(drug_mh, long_self_weight, long_near_weight, long_second_weight, long_late_weight)
    fc_mh = apply_long_hop_mode(fc_mh, long_self_weight, long_near_weight, long_second_weight, long_late_weight)
    food_mh = apply_long_hop_mode(food_mh, long_self_weight, long_near_weight, long_second_weight, long_late_weight)

    small_type_threshold = int(getattr(args, 'long_small_type_threshold', 0))
    small_type_floor = float(getattr(args, 'long_small_type_floor', 1.0))
    drug_mh, long_type_scale_drug = attenuate_small_type_long_features(
        drug_mh, nd, small_type_threshold, small_type_floor
    )
    fc_mh, long_type_scale_fc = attenuate_small_type_long_features(
        fc_mh, nfc, small_type_threshold, small_type_floor
    )
    food_mh, long_type_scale_food = attenuate_small_type_long_features(
        food_mh, nf, small_type_threshold, small_type_floor
    )



    nei_f = build_neighbor_index(df_adj, nd)

    nei_fc = build_neighbor_index(dfc_adj_tr, nd)

    nei_d = build_neighbor_index(dfc_adj_tr.T, nfc)

    nei_food = build_neighbor_index(ffc_adj.T, nfc)

    nei_drug_for_food = build_neighbor_index(df_adj.T, nf)

    nei_fc_for_food = build_neighbor_index(ffc_adj, nf)



    deg_food = torch.FloatTensor(np.asarray(df_adj.sum(axis=0)).reshape(-1).astype(np.float32)).to(device)

    deg_fc = torch.FloatTensor(np.asarray(dfc_adj_tr.sum(axis=0)).reshape(-1).astype(np.float32)).to(device)

    deg_drug = torch.FloatTensor(np.asarray(dfc_adj_tr.sum(axis=1)).reshape(-1).astype(np.float32)).to(device)

    deg_food_for_fc = torch.FloatTensor(np.asarray(ffc_adj.sum(axis=1)).reshape(-1).astype(np.float32)).to(device)

    nei_degree_drug = [deg_food, deg_fc]

    nei_degree_fc = [deg_drug, deg_food_for_fc]

    deg_drug_for_food = torch.FloatTensor(np.asarray(df_adj.sum(axis=1)).reshape(-1).astype(np.float32)).to(device)

    deg_fc_for_food = torch.FloatTensor(np.asarray(ffc_adj.sum(axis=0)).reshape(-1).astype(np.float32)).to(device)

    nei_degree_food = [deg_drug_for_food, deg_fc_for_food]



    cl_pos_mode = getattr(args, 'cl_pos_mode', 'self')
    if cl_pos_mode == 'self':
        pos_drug = identity_contrast_pos(nd)
        pos_fc = identity_contrast_pos(nfc)
        pos_food = identity_contrast_pos(nf)
    elif cl_pos_mode == 'semantic_self':
        pos_drug = add_self_loops_binary((dfd + dfcd) > 0)
        pos_fc = add_self_loops_binary((fcdfc + fcffc) > 0)
        pos_food = add_self_loops_binary((fdf + ffcf) > 0)
    else:
        raise ValueError(f"Unsupported cl_pos_mode: {cl_pos_mode}")



    train_side_pos_by_drug = defaultdict(set)

    for d, fc in train_pos:

        train_side_pos_by_drug[int(d)].add(int(fc))



    train_fc_degree = np.asarray(dfc_adj_tr.sum(axis=0)).reshape(-1).astype(np.float32)

    train_fc_degree = np.maximum(train_fc_degree, 1.0)

    graph_feats = [

        torch.FloatTensor(preprocess_dense_features(drug_feat)).to(device),

        torch.FloatTensor(preprocess_dense_features(food_feat)).to(device),

        torch.FloatTensor(preprocess_dense_features(fc_feat)).to(device),

    ]

    smiles_feats = [None, None, None]

    drug_mh_t = [torch.FloatTensor(m).to(device) for m in drug_mh]

    fc_mh_t = [torch.FloatTensor(m).to(device) for m in fc_mh]

    food_mh_t = [torch.FloatTensor(m).to(device) for m in food_mh]

    pos_drug_t = sparse_mx_to_torch_sparse_tensor(pos_drug).to(device)

    pos_fc_t = sparse_mx_to_torch_sparse_tensor(pos_fc).to(device)

    pos_food_t = sparse_mx_to_torch_sparse_tensor(pos_food).to(device)







    short_view_source = getattr(args, 'short_view_source', 'all')
    if task == 'dfc':
        if short_view_source == 'target':
            nei_idx_drug = [[torch.LongTensor(i) for i in nei_fc]]
            nei_idx_fc = [[torch.LongTensor(i) for i in nei_d]]
            nei_degree_drug = [deg_fc]
            nei_degree_fc = [deg_drug]
        elif short_view_source == 'bridge':
            nei_idx_drug = [[torch.LongTensor(i) for i in nei_f]]
            nei_idx_fc = [[torch.LongTensor(i) for i in nei_food]]
            nei_degree_drug = [deg_food]
            nei_degree_fc = [deg_food_for_fc]
        elif short_view_source == 'all':
            nei_idx_drug = [[torch.LongTensor(i) for i in nei_fc], [torch.LongTensor(i) for i in nei_f]]
            nei_idx_fc = [[torch.LongTensor(i) for i in nei_d], [torch.LongTensor(i) for i in nei_food]]
            nei_degree_drug = [deg_fc, deg_food]
            nei_degree_fc = [deg_drug, deg_food_for_fc]
        else:
            raise ValueError("short_view_source must be 'target', 'bridge', or 'all'")
        nei_idx_food = [[torch.LongTensor(i) for i in nei_drug_for_food], [torch.LongTensor(i) for i in nei_fc_for_food]]
    else:
        if short_view_source == 'target':
            nei_idx_drug = [[torch.LongTensor(i) for i in nei_f]]
            nei_idx_food = [[torch.LongTensor(i) for i in nei_drug_for_food]]
            nei_degree_drug = [deg_food]
            nei_degree_food = [deg_drug_for_food]
        elif short_view_source == 'bridge':
            nei_idx_drug = [[torch.LongTensor(i) for i in nei_fc]]
            nei_idx_food = [[torch.LongTensor(i) for i in nei_fc_for_food]]
            nei_degree_drug = [deg_fc]
            nei_degree_food = [deg_fc_for_food]
        elif short_view_source == 'all':
            nei_idx_drug = [[torch.LongTensor(i) for i in nei_f], [torch.LongTensor(i) for i in nei_fc]]
            nei_idx_food = [[torch.LongTensor(i) for i in nei_drug_for_food], [torch.LongTensor(i) for i in nei_fc_for_food]]
            nei_degree_drug = [deg_food, deg_fc]
            nei_degree_food = [deg_drug_for_food, deg_fc_for_food]
        else:
            raise ValueError("short_view_source must be 'target', 'bridge', or 'all'")
        nei_idx_fc = [[torch.LongTensor(i) for i in nei_d], [torch.LongTensor(i) for i in nei_food]]



    return {

        'graph_feats': graph_feats, 'smiles_feats': smiles_feats,

        'drug_mh': drug_mh_t, 'fc_mh': fc_mh_t, 'food_mh': food_mh_t,

        'pos_drug': pos_drug_t, 'pos_fc': pos_fc_t, 'pos_food': pos_food_t,

        'nei_drug': nei_idx_drug, 'nei_fc': nei_idx_fc, 'nei_food': nei_idx_food,

        'train_pos': train_pos, 'train_neg': train_neg,

        'val_pos': val_pos, 'val_neg': val_neg,

        'test_pos': test_pos, 'test_neg': test_neg,

        'P_drug': len(drug_meta), 'P_fc': len(fc_meta), 'P_food': len(food_meta),

        'nei_weight_drug': None, 'nei_weight_fc': None,

        'nei_degree_drug': nei_degree_drug, 'nei_degree_fc': nei_degree_fc, 'nei_degree_food': nei_degree_food,

        'train_side_pos_by_drug': train_side_pos_by_drug,

        'train_fc_degree': train_fc_degree,

        'neg_ratio': int(neg_ratio),

        'neg_protocol': 'controlled_negative_folds',

        'meta_mode': meta_mode,
        'long_view_mode': long_view_mode,
        'long_hop_mode': 'soft_context',
        'long_self_weight': float(long_self_weight),
        'long_near_weight': float(long_near_weight),
        'long_second_weight': float(long_second_weight),
        'long_late_weight': float(long_late_weight),
        'long_small_type_threshold': int(small_type_threshold),
        'long_small_type_floor': float(small_type_floor),
        'long_type_scale_drug': float(long_type_scale_drug),
        'long_type_scale_fc': float(long_type_scale_fc),
        'long_type_scale_food': float(long_type_scale_food),

        'task': task,

        'target_type': target_type,

        'num_right': int(num_right),

        'food_feature_mode': 'random',

        'use_smiles_features': True,

        'smiles_fp_dim': int(smiles_fp_dim),

        'smiles_radius': int(smiles_radius),

        'smiles_coverage': {

            'drug': (count_nonzero_rows(drug_feat), nd),

            'food': (count_nonzero_rows(food_feat), nf),

            'fc': (count_nonzero_rows(fc_feat), nfc),

        },

    }




def encode_onehot(labels):
    labels = labels.reshape(-1, 1)
    enc = OneHotEncoder()
    enc.fit(labels)
    labels_onehot = enc.transform(labels).toarray()
    return labels_onehot


def preprocess_features(features):

    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense()


def normalize_adj(adj):

    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def sparse_mx_to_torch_sparse_tensor(sparse_mx):

    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = th.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = th.from_numpy(sparse_mx.data)
    shape = th.Size(sparse_mx.shape)
    return th.sparse.FloatTensor(indices, values, shape)


def load_acm(ratio, type_num):

    path = "../data/acm/"
    label = np.load(path + "labels.npy").astype('int32')
    label = encode_onehot(label)
    nei_a = np.load(path + "nei_a.npy", allow_pickle=True)
    nei_s = np.load(path + "nei_s.npy", allow_pickle=True)
    feat_p = sp.load_npz(path + "p_feat.npz")
    feat_a = sp.eye(type_num[1])
    feat_s = sp.eye(type_num[2])
    pap = sp.load_npz(path + "pap.npz")
    psp = sp.load_npz(path + "psp.npz")
    pos = sp.load_npz(path + "pos.npz")
    train = [np.load(path + "train_" + str(i) + ".npy") for i in ratio]
    test = [np.load(path + "test_" + str(i) + ".npy") for i in ratio]
    val = [np.load(path + "val_" + str(i) + ".npy") for i in ratio]

    label = th.FloatTensor(label)
    nei_a = [th.LongTensor(i) for i in nei_a]
    nei_s = [th.LongTensor(i) for i in nei_s]
    feat_p = th.FloatTensor(preprocess_features(feat_p))
    feat_a = th.FloatTensor(preprocess_features(feat_a))
    feat_s = th.FloatTensor(preprocess_features(feat_s))
    pap = sparse_mx_to_torch_sparse_tensor(normalize_adj(pap))
    psp = sparse_mx_to_torch_sparse_tensor(normalize_adj(psp))
    pos = sparse_mx_to_torch_sparse_tensor(pos)
    train = [th.LongTensor(i) for i in train]
    val = [th.LongTensor(i) for i in val]
    test = [th.LongTensor(i) for i in test]
    return [nei_a, nei_s], [feat_p, feat_a, feat_s], [pap, psp], pos, label, train, val, test


def load_dblp(ratio, type_num):

    path = "../data/dblp/"
    label = np.load(path + "labels.npy").astype('int32')
    label = encode_onehot(label)
    nei_p = np.load(path + "nei_p.npy", allow_pickle=True)
    feat_a = sp.load_npz(path + "a_feat.npz").astype("float32")
    feat_p = sp.eye(type_num[1])
    apa = sp.load_npz(path + "apa.npz")
    apcpa = sp.load_npz(path + "apcpa.npz")
    aptpa = sp.load_npz(path + "aptpa.npz")
    pos = sp.load_npz(path + "pos.npz")
    train = [np.load(path + "train_" + str(i) + ".npy") for i in ratio]
    test = [np.load(path + "test_" + str(i) + ".npy") for i in ratio]
    val = [np.load(path + "val_" + str(i) + ".npy") for i in ratio]

    label = th.FloatTensor(label)
    nei_p = [th.LongTensor(i) for i in nei_p]
    feat_p = th.FloatTensor(preprocess_features(feat_p))
    feat_a = th.FloatTensor(preprocess_features(feat_a))
    apa = sparse_mx_to_torch_sparse_tensor(normalize_adj(apa))
    apcpa = sparse_mx_to_torch_sparse_tensor(normalize_adj(apcpa))
    aptpa = sparse_mx_to_torch_sparse_tensor(normalize_adj(aptpa))
    pos = sparse_mx_to_torch_sparse_tensor(pos)
    train = [th.LongTensor(i) for i in train]
    val = [th.LongTensor(i) for i in val]
    test = [th.LongTensor(i) for i in test]
    return [nei_p], [feat_a, feat_p], [apa, apcpa, aptpa], pos, label, train, val, test


def load_aminer(ratio, type_num):

    path = "../data/aminer/"
    label = np.load(path + "labels.npy").astype('int32')
    label = encode_onehot(label)
    nei_a = np.load(path + "nei_a.npy", allow_pickle=True)
    nei_r = np.load(path + "nei_r.npy", allow_pickle=True)

    feat_p = sp.eye(type_num[0])
    feat_a = sp.eye(type_num[1])
    feat_r = sp.eye(type_num[2])
    pap = sp.load_npz(path + "pap.npz")
    prp = sp.load_npz(path + "prp.npz")
    pos = sp.load_npz(path + "pos.npz")
    train = [np.load(path + "train_" + str(i) + ".npy") for i in ratio]
    test = [np.load(path + "test_" + str(i) + ".npy") for i in ratio]
    val = [np.load(path + "val_" + str(i) + ".npy") for i in ratio]

    label = th.FloatTensor(label)
    nei_a = [th.LongTensor(i) for i in nei_a]
    nei_r = [th.LongTensor(i) for i in nei_r]
    feat_p = th.FloatTensor(preprocess_features(feat_p))
    feat_a = th.FloatTensor(preprocess_features(feat_a))
    feat_r = th.FloatTensor(preprocess_features(feat_r))
    pap = sparse_mx_to_torch_sparse_tensor(normalize_adj(pap))
    prp = sparse_mx_to_torch_sparse_tensor(normalize_adj(prp))
    pos = sparse_mx_to_torch_sparse_tensor(pos)
    train = [th.LongTensor(i) for i in train]
    val = [th.LongTensor(i) for i in val]
    test = [th.LongTensor(i) for i in test]
    return [nei_a, nei_r], [feat_p, feat_a, feat_r], [pap, prp], pos, label, train, val, test


def load_freebase(ratio, type_num):

    path = "../data/freebase/"
    label = np.load(path + "labels.npy").astype('int32')
    label = encode_onehot(label)
    nei_d = np.load(path + "nei_d.npy", allow_pickle=True)
    nei_a = np.load(path + "nei_a.npy", allow_pickle=True)
    nei_w = np.load(path + "nei_w.npy", allow_pickle=True)
    feat_m = sp.eye(type_num[0])
    feat_d = sp.eye(type_num[1])
    feat_a = sp.eye(type_num[2])
    feat_w = sp.eye(type_num[3])

    mam = sp.load_npz(path + "mam.npz")
    mdm = sp.load_npz(path + "mdm.npz")
    mwm = sp.load_npz(path + "mwm.npz")
    pos = sp.load_npz(path + "pos.npz")
    train = [np.load(path + "train_" + str(i) + ".npy") for i in ratio]
    test = [np.load(path + "test_" + str(i) + ".npy") for i in ratio]
    val = [np.load(path + "val_" + str(i) + ".npy") for i in ratio]

    label = th.FloatTensor(label)
    nei_d = [th.LongTensor(i) for i in nei_d]
    nei_a = [th.LongTensor(i) for i in nei_a]
    nei_w = [th.LongTensor(i) for i in nei_w]
    feat_m = th.FloatTensor(preprocess_features(feat_m))
    feat_d = th.FloatTensor(preprocess_features(feat_d))
    feat_a = th.FloatTensor(preprocess_features(feat_a))
    feat_w = th.FloatTensor(preprocess_features(feat_w))
    mam = sparse_mx_to_torch_sparse_tensor(normalize_adj(mam))
    mdm = sparse_mx_to_torch_sparse_tensor(normalize_adj(mdm))
    mwm = sparse_mx_to_torch_sparse_tensor(normalize_adj(mwm))
    pos = sparse_mx_to_torch_sparse_tensor(pos)
    train = [th.LongTensor(i) for i in train]
    val = [th.LongTensor(i) for i in val]
    test = [th.LongTensor(i) for i in test]
    return [nei_d, nei_a, nei_w], [feat_m, feat_d, feat_a, feat_w], [mdm, mam, mwm], pos, label, train, val, test


def load_data(dataset, ratio, type_num):
    if dataset == "acm":
        data = load_acm(ratio, type_num)
    elif dataset == "dblp":
        data = load_dblp(ratio, type_num)
    elif dataset == "aminer":
        data = load_aminer(ratio, type_num)
    elif dataset == "freebase":
        data = load_freebase(ratio, type_num)
    return data
