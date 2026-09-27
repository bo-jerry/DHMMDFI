import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))


class Args:

    n_folds = 5


    seeds_per_fold = [42, 123, 456, 789, 1024]
    gpu = 0


    hidden_dim = 64




    long_hidden_dim = 24




    num_hops = 5
    d_state = 16
    d_conv = 4
    expand = 2
    n_mamba_layers = 1



    long_encoder_mode = 'semantic'





    sample_rate_drug = [5]
    sample_rate_fc = [5]
    short_use_log_degree_penalty = False
    short_degree_penalty_bias = 2.718281828
    short_output_drop = 0.15




    short_self_weight = 0.40



    short_bottleneck_dim = 5



    short_relation_drop = 0.10
    short_second_layer = 'relation_gat'






    short_view_source = 'target'



    long_view_mode = 'symmetric_long'


    long_hop_mode = 'soft_context'
    long_self_weight = 0.0
    long_near_weight = 0.02
    long_second_weight = 1.0
    long_late_weight = 1.0



    pathsim_topk = 1










    pathsim_topk_ratio = 0.03
    pathsim_min_topk = 1
    pathsim_min_sim = 0.0


    pathsim_mode = 'pathsim'





    long_small_type_threshold = 200
    long_small_type_floor = 0.20


    long_output_drop = 0.15





    long_meta_path_drop = 0.0



    view_norm = True





    cross_view_residual = True
    cross_view_residual_scale = 1.0


    epochs = 100
    log_interval = 20
    lr = 0.001
    l2 = 1e-4
    grad_clip = 2.0







    cl_align_mode = 'symmetric'







    cl_pos_mode = 'self'
    view_decorrelation_weight = 0.001
    tau = 2.5
    lam = 0.5
    feat_drop = 0.15
    attn_drop = 0.15
    mamba_dropout = 0.05


    lp_hidden = 48
    lp_dropout = 0.20
    threshold = 0.5







    fusion_mode = 'concat_calibrated'
    predictor_type = 'mlp'




    predictor_edge_ops = False

    pair_main_weight = 1.0
    pair_long_weight = 0.38
    pair_short_weight = 0.65
    pair_synergy_weight = 1.15
    learn_pair_logit_weights = False



    logit_level_ablation = False



    long_strength = 'semantic_restore_lite'
    short_strength = 'relation_strong'

    gate_init_long_weight = 0.5



    neg_strategy = 'global_random'










    ablation = 'none'
    meta_mode = 'base'


    smiles_fp_dim = 512
    smiles_radius = 2


args = Args()







import argparse



def parse_cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='drugbank', choices=['drugbank', 'pubmed'])
    parser.add_argument('--task', type=str, default='dfc', choices=['dfc', 'df'])
    parsed = parser.parse_args()
    for name in dir(args):
        if not name.startswith('_') and not hasattr(parsed, name):
            value = getattr(args, name)
            if not callable(value):
                setattr(parsed, name, value)
    parsed.neg_ratio = 1
    parsed.meta_mode = 'base'
    parsed.predictor_type = args.predictor_type
    parsed.predictor_edge_ops = args.predictor_edge_ops
    parsed.fusion_mode = args.fusion_mode
    parsed.cl_align_mode = args.cl_align_mode
    parsed.cl_pos_mode = args.cl_pos_mode
    parsed.view_decorrelation_weight = None
    parsed.epochs = None
    parsed.lr = None
    parsed.lp_dropout = None
    parsed.long_encoder_mode = None
    parsed.short_sample_rate = None
    parsed.short_output_drop = None
    parsed.short_self_weight = None
    parsed.short_bottleneck_dim = None
    parsed.short_relation_drop = None
    parsed.no_short_degree_penalty = False
    parsed.no_view_norm = False
    parsed.cross_view_residual_scale = None
    parsed.custom_long_params = False
    parsed.seeds = ''
    parsed.neg_seed = 42
    parsed.quick_folds = 0
    parsed.run_tag = ''
    return parsed


def apply_cli_overrides(cli_args):
    args.dataset = cli_args.dataset
    args.ablation = 'none'
    args.predictor_type = cli_args.predictor_type
    args.predictor_edge_ops = cli_args.predictor_edge_ops



    args.fusion_mode = 'pair_cross' if args.predictor_type == 'view_interaction' else cli_args.fusion_mode
    args.meta_mode = cli_args.meta_mode
    args.task = cli_args.task
    args.food_feature_mode = 'random'
    args.cl_align_mode = cli_args.cl_align_mode
    args.cl_pos_mode = cli_args.cl_pos_mode
    args.tau = cli_args.tau
    if cli_args.view_decorrelation_weight is not None:
        args.view_decorrelation_weight = cli_args.view_decorrelation_weight
    elif cli_args.dataset == 'drugbank':
        args.view_decorrelation_weight = 0.0
    else:
        args.view_decorrelation_weight = Args.view_decorrelation_weight
    if cli_args.epochs is not None:
        args.epochs = cli_args.epochs
        args.dataset_profile = 'manual_epochs'
    elif cli_args.dataset == 'drugbank':


        args.epochs = 140
        args.dataset_profile = 'drugbank_l229'
    else:
        args.epochs = Args.epochs
        args.dataset_profile = 'pubmed_l181'
    args.log_interval = cli_args.log_interval
    if cli_args.lr is not None:
        args.lr = cli_args.lr
    elif cli_args.dataset == 'drugbank':
        args.lr = 0.001
    else:
        args.lr = Args.lr
    args.grad_clip = cli_args.grad_clip
    args.feat_drop = cli_args.feat_drop
    args.attn_drop = cli_args.attn_drop
    args.lp_hidden = cli_args.lp_hidden
    if cli_args.lp_dropout is not None:
        args.lp_dropout = cli_args.lp_dropout
    elif cli_args.dataset == 'drugbank':
        args.lp_dropout = 0.16
    else:
        args.lp_dropout = Args.lp_dropout
    args.num_hops = cli_args.num_hops
    args.long_hidden_dim = cli_args.long_hidden_dim
    args.n_mamba_layers = cli_args.n_mamba_layers
    args.mamba_dropout = cli_args.mamba_dropout
    args.long_encoder_mode = cli_args.long_encoder_mode if cli_args.long_encoder_mode is not None else args.long_encoder_mode



    args.long_strength = cli_args.long_strength
    if cli_args.short_sample_rate is not None:
        args.sample_rate_drug = [cli_args.short_sample_rate]
        args.sample_rate_fc = [cli_args.short_sample_rate]
    args.short_strength = cli_args.short_strength
    args.short_second_layer = cli_args.short_second_layer
    args.short_view_source = cli_args.short_view_source
    if cli_args.short_output_drop is not None:
        args.short_output_drop = cli_args.short_output_drop
    if cli_args.short_self_weight is not None:
        args.short_self_weight = cli_args.short_self_weight
    if cli_args.short_bottleneck_dim is not None:
        args.short_bottleneck_dim = cli_args.short_bottleneck_dim
    if cli_args.short_relation_drop is not None:
        args.short_relation_drop = cli_args.short_relation_drop
    if cli_args.no_short_degree_penalty:
        args.short_use_log_degree_penalty = False
    args.long_view_mode = cli_args.long_view_mode
    args.long_hop_mode = 'soft_context'
    args.long_self_weight = cli_args.long_self_weight
    args.long_near_weight = cli_args.long_near_weight
    args.long_second_weight = cli_args.long_second_weight
    args.long_late_weight = cli_args.long_late_weight
    args.pathsim_topk = cli_args.pathsim_topk
    args.pathsim_topk_ratio = cli_args.pathsim_topk_ratio
    args.pathsim_min_topk = cli_args.pathsim_min_topk
    args.pathsim_min_sim = cli_args.pathsim_min_sim
    args.long_small_type_threshold = cli_args.long_small_type_threshold
    args.long_small_type_floor = cli_args.long_small_type_floor
    args.long_output_drop = cli_args.long_output_drop
    args.long_meta_path_drop = cli_args.long_meta_path_drop
    args.view_norm = not cli_args.no_view_norm
    args.cross_view_residual = bool(cli_args.cross_view_residual)
    if cli_args.cross_view_residual_scale is not None:
        args.cross_view_residual_scale = cli_args.cross_view_residual_scale
    elif cli_args.dataset == 'drugbank':
        args.cross_view_residual_scale = 0.8
    else:
        args.cross_view_residual_scale = Args.cross_view_residual_scale
    args.neg_strategy = 'global_random'
    args.logit_level_ablation = False
    args.neg_seed = cli_args.neg_seed
    if cli_args.seeds.strip():
        args.seeds_per_fold = [int(x.strip()) for x in cli_args.seeds.split(',') if x.strip()]
    args.run_tag = cli_args.run_tag




    args.long_second_weight = 1.0
    args.long_late_weight = 1.0

    if args.long_strength == 'ultra':
        args.long_self_weight = 0.0
        args.long_near_weight = 0.10
        args.pathsim_topk = 5
        args.pathsim_topk_ratio = 0.06
        args.pathsim_min_topk = 2
        args.pair_long_weight = 0.30
        if cli_args.long_encoder_mode is None:
            args.long_encoder_mode = 'simple'
        args.n_mamba_layers = 1
        args.long_hidden_dim = 32
    elif args.long_strength == 'lite':
        args.long_self_weight = 0.02
        args.long_near_weight = 0.18
        args.pathsim_topk = 8
        args.pathsim_topk_ratio = 0.06
        args.pathsim_min_topk = 2
        args.pair_long_weight = 0.38
        if cli_args.long_encoder_mode is None:
            args.long_encoder_mode = 'simple'
        args.n_mamba_layers = 1
        args.long_hidden_dim = 48
    elif args.long_strength == 'semantic_lite':
        args.long_self_weight = 0.01
        args.long_near_weight = 0.12
        args.pathsim_topk = 6
        args.pathsim_topk_ratio = 0.06
        args.pathsim_min_topk = 2
        args.pair_long_weight = 0.34
        if cli_args.long_encoder_mode is None:
            args.long_encoder_mode = 'semantic'
        args.n_mamba_layers = 1
        args.long_hidden_dim = 48
    elif args.long_strength == 'semantic_mid_lite':
        args.long_self_weight = 0.005
        args.long_near_weight = 0.10
        args.pathsim_topk = 5
        args.pathsim_topk_ratio = 0.06
        args.pathsim_min_topk = 2
        args.pair_long_weight = 0.30
        if cli_args.long_encoder_mode is None:
            args.long_encoder_mode = 'semantic'
        args.n_mamba_layers = 1
        args.long_hidden_dim = 40
    elif args.long_strength == 'semantic_search_a':
        args.long_self_weight = 0.004
        args.long_near_weight = 0.095
        args.pathsim_topk = 5
        args.pathsim_topk_ratio = 0.06
        args.pathsim_min_topk = 2
        args.pair_long_weight = 0.29
        if cli_args.long_encoder_mode is None:
            args.long_encoder_mode = 'semantic'
        args.n_mamba_layers = 1
        args.long_hidden_dim = 38
    elif args.long_strength == 'semantic_search_b':
        args.long_self_weight = 0.003
        args.long_near_weight = 0.09
        args.pathsim_topk = 4
        args.pathsim_topk_ratio = 0.06
        args.pathsim_min_topk = 2
        args.pair_long_weight = 0.29
        if cli_args.long_encoder_mode is None:
            args.long_encoder_mode = 'semantic'
        args.n_mamba_layers = 1
        args.long_hidden_dim = 38
    elif args.long_strength == 'semantic_search_c':
        args.long_self_weight = 0.002
        args.long_near_weight = 0.085
        args.pathsim_topk = 4
        args.pathsim_topk_ratio = 0.06
        args.pathsim_min_topk = 2
        args.pair_long_weight = 0.28
        if cli_args.long_encoder_mode is None:
            args.long_encoder_mode = 'semantic'
        args.n_mamba_layers = 1
        args.long_hidden_dim = 36
    elif args.long_strength == 'semantic_search_d':
        args.long_self_weight = 0.05
        args.long_near_weight = 0.0
        args.long_second_weight = 1.0
        args.pathsim_topk = 4
        args.pathsim_topk_ratio = 0.06
        args.pathsim_min_topk = 2
        args.pair_long_weight = 0.255
        if cli_args.long_encoder_mode is None:
            args.long_encoder_mode = 'semantic'
        args.n_mamba_layers = 1
        args.long_hidden_dim = 36
    elif args.long_strength == 'semantic_search_e':
        args.long_self_weight = 0.001
        args.long_near_weight = 0.075
        args.pathsim_topk = 3
        args.pathsim_topk_ratio = 0.06
        args.pathsim_min_topk = 2
        args.pair_long_weight = 0.25
        if cli_args.long_encoder_mode is None:
            args.long_encoder_mode = 'semantic'
        args.n_mamba_layers = 1
        args.long_hidden_dim = 34
    elif args.long_strength == 'semantic_search_f':
        args.long_self_weight = 0.0
        args.long_near_weight = 0.07
        args.pathsim_topk = 3
        args.pathsim_topk_ratio = 0.06
        args.pathsim_min_topk = 2
        args.pair_long_weight = 0.24
        if cli_args.long_encoder_mode is None:
            args.long_encoder_mode = 'semantic'
        args.n_mamba_layers = 1
        args.long_hidden_dim = 32
    elif args.long_strength == 'semantic_search_g':
        args.long_self_weight = 0.03
        args.long_near_weight = 0.0
        args.long_second_weight = 0.65
        args.long_late_weight = 0.45
        args.pathsim_topk = 3
        args.pathsim_topk_ratio = 0.06
        args.pathsim_min_topk = 2
        if cli_args.long_encoder_mode is None:
            args.long_encoder_mode = 'semantic'
        args.n_mamba_layers = 1
        args.long_hidden_dim = 34
    elif args.long_strength == 'semantic_search_h':
        args.long_self_weight = 0.02
        args.long_near_weight = 0.0
        args.long_second_weight = 0.45
        args.long_late_weight = 0.25
        args.pathsim_topk = 3
        args.pathsim_topk_ratio = 0.06
        args.pathsim_min_topk = 2
        if cli_args.long_encoder_mode is None:
            args.long_encoder_mode = 'semantic'
        args.n_mamba_layers = 1
        args.long_hidden_dim = 32
    elif args.long_strength == 'semantic_search_i':
        args.long_self_weight = 0.01
        args.long_near_weight = 0.0
        args.long_second_weight = 0.30
        args.long_late_weight = 0.15
        args.pathsim_topk = 2
        args.pathsim_topk_ratio = 0.06
        args.pathsim_min_topk = 1
        if cli_args.long_encoder_mode is None:
            args.long_encoder_mode = 'semantic'
        args.n_mamba_layers = 1
        args.long_hidden_dim = 32
    elif args.long_strength == 'semantic_search_j':
        args.long_self_weight = 0.005
        args.long_near_weight = 0.0
        args.long_second_weight = 0.20
        args.long_late_weight = 0.08
        args.pathsim_topk = 2
        args.pathsim_topk_ratio = 0.06
        args.pathsim_min_topk = 1
        if cli_args.long_encoder_mode is None:
            args.long_encoder_mode = 'semantic'
        args.n_mamba_layers = 1
        args.long_hidden_dim = 28
    elif args.long_strength == 'semantic_search_k':
        args.long_self_weight = 0.0
        args.long_near_weight = 0.0
        args.long_second_weight = 0.12
        args.long_late_weight = 0.04
        args.pathsim_topk = 1
        args.pathsim_topk_ratio = 0.06
        args.pathsim_min_topk = 1
        if cli_args.long_encoder_mode is None:
            args.long_encoder_mode = 'semantic'
        args.n_mamba_layers = 1
        args.long_hidden_dim = 24
    elif args.long_strength == 'semantic_search_balanced':
        args.long_self_weight = 0.02
        args.long_near_weight = 0.0
        args.long_second_weight = 0.60
        args.long_late_weight = 0.40
        args.pathsim_topk = 6
        args.pathsim_topk_ratio = 0.06
        args.pathsim_min_topk = 2
        if cli_args.long_encoder_mode is None:
            args.long_encoder_mode = 'semantic'
        args.n_mamba_layers = 1
        args.long_hidden_dim = 36
    elif args.long_strength == 'semantic_search_balanced_weak':
        args.long_self_weight = 0.015
        args.long_near_weight = 0.0
        args.long_second_weight = 0.45
        args.long_late_weight = 0.25
        args.pathsim_topk = 5
        args.pathsim_topk_ratio = 0.06
        args.pathsim_min_topk = 2
        if cli_args.long_encoder_mode is None:
            args.long_encoder_mode = 'semantic'
        args.n_mamba_layers = 1
        args.long_hidden_dim = 34
    elif args.long_strength == 'semantic_search_core':
        args.long_self_weight = 0.01
        args.long_near_weight = 0.05
        args.long_second_weight = 1.0
        args.long_late_weight = 1.0
        args.pathsim_topk = 4
        args.pathsim_topk_ratio = 0.06
        args.pathsim_min_topk = 2
        if cli_args.long_encoder_mode is None:
            args.long_encoder_mode = 'semantic'
        args.n_mamba_layers = 1
        args.long_hidden_dim = 36
    elif args.long_strength == 'm1':


        args.num_hops = 8
        args.long_hidden_dim = 48
        args.n_mamba_layers = 3


        args.long_self_weight = 0.012 if cli_args.dataset == 'pubmed' else 0.0
        args.long_near_weight = 0.01
        args.long_second_weight = 1.0
        args.long_late_weight = 1.0
        args.pathsim_topk = 4
        args.pathsim_topk_ratio = 0.03
        args.pathsim_min_topk = 1
        args.pathsim_min_sim = 0.0
        args.long_output_drop = 0.10
        if cli_args.long_encoder_mode is None:
            args.long_encoder_mode = 'semantic'
    elif args.long_strength == 'm2':


        args.num_hops = 9
        args.long_hidden_dim = 48
        args.n_mamba_layers = 3
        args.long_self_weight = 0.0
        args.long_near_weight = 0.0
        args.long_second_weight = 0.80
        args.long_late_weight = 1.0
        args.pathsim_topk = 4
        args.pathsim_topk_ratio = 0.03
        args.pathsim_min_topk = 1
        args.pathsim_min_sim = 0.0
        args.long_output_drop = 0.10
        if cli_args.long_encoder_mode is None:
            args.long_encoder_mode = 'semantic'
    elif args.long_strength == 'm3':


        args.num_hops = 8
        args.long_hidden_dim = 64
        args.n_mamba_layers = 3
        args.long_self_weight = 0.0
        args.long_near_weight = 0.01
        args.long_second_weight = 1.0
        args.long_late_weight = 1.0
        args.pathsim_topk = 6
        args.pathsim_topk_ratio = 0.03
        args.pathsim_min_topk = 1
        args.pathsim_min_sim = 0.0
        args.long_output_drop = 0.10
        if cli_args.long_encoder_mode is None:
            args.long_encoder_mode = 'semantic'
    elif args.long_strength == 'm4':


        args.num_hops = 10
        args.long_hidden_dim = 48
        args.n_mamba_layers = 2
        args.long_self_weight = 0.0
        args.long_near_weight = 0.0
        args.long_second_weight = 0.80
        args.long_late_weight = 1.0
        args.pathsim_topk = 4
        args.pathsim_topk_ratio = 0.03
        args.pathsim_min_topk = 1
        args.pathsim_min_sim = 0.0
        args.long_output_drop = 0.08
        if cli_args.long_encoder_mode is None:
            args.long_encoder_mode = 'semantic'
    elif args.long_strength in ('m5', 'm6', 'm7', 'db1', 'db2', 'db3'):
        if args.long_strength in ('db1', 'db2', 'db3') and cli_args.dataset != 'drugbank':
            raise ValueError('db1/db2/db3 presets are restricted to dataset=drugbank')





        args.num_hops = 8
        args.long_hidden_dim = 48
        args.n_mamba_layers = 3
        args.long_self_weight = 0.0
        args.long_near_weight = 0.01
        args.long_second_weight = 1.0
        args.long_late_weight = 1.0
        args.pathsim_topk = 2
        args.pathsim_topk_ratio = 0.03
        args.pathsim_min_topk = 1
        args.pathsim_min_sim = 0.0
        args.long_output_drop = 0.10
        if cli_args.long_encoder_mode is None:
            args.long_encoder_mode = 'semantic'
    elif args.long_strength == 'semantic_restore_lite':





        args.long_self_weight = 0.0
        args.long_near_weight = 0.02
        args.long_second_weight = 1.0
        args.long_late_weight = 1.0
        args.pathsim_topk = 1
        args.pathsim_topk_ratio = 0.03
        args.pathsim_min_topk = 1
        if cli_args.long_encoder_mode is None:
            args.long_encoder_mode = 'semantic'
        args.n_mamba_layers = 1
        args.long_hidden_dim = 24
    elif args.long_strength == 'semantic_restore_balanced':





        args.long_self_weight = 0.05
        args.long_near_weight = 0.15
        args.long_second_weight = 1.0
        args.long_late_weight = 1.0
        args.pathsim_topk = 8


        args.pathsim_topk_ratio = 0.026
        args.pathsim_min_topk = 1
        if cli_args.long_encoder_mode is None:
            args.long_encoder_mode = 'semantic'
        args.n_mamba_layers = 1
        args.long_hidden_dim = 52
    elif args.long_strength == 'semantic_restore_synergy':





        args.long_self_weight = 0.02
        args.long_near_weight = 0.08
        args.long_second_weight = 1.0
        args.long_late_weight = 1.0
        args.pathsim_topk = 16
        args.pathsim_topk_ratio = 0.06
        args.pathsim_min_topk = 2
        if cli_args.long_encoder_mode is None:
            args.long_encoder_mode = 'semantic'
        args.n_mamba_layers = 1
        args.long_hidden_dim = 64
    elif args.long_strength == 'semantic_restore_strong':



        args.long_self_weight = 0.06
        args.long_near_weight = 0.18
        args.long_second_weight = 1.0
        args.long_late_weight = 1.0
        args.pathsim_topk = 8
        args.pathsim_topk_ratio = 0.06
        args.pathsim_min_topk = 2
        if cli_args.long_encoder_mode is None:
            args.long_encoder_mode = 'semantic'
        args.n_mamba_layers = 1
        args.long_hidden_dim = 56
    elif args.long_strength == 'semantic_low_lite':
        args.long_self_weight = 0.002
        args.long_near_weight = 0.09
        args.pathsim_topk = 4
        args.pathsim_topk_ratio = 0.06
        args.pathsim_min_topk = 2
        args.pair_long_weight = 0.27
        if cli_args.long_encoder_mode is None:
            args.long_encoder_mode = 'semantic'
        args.n_mamba_layers = 1
        args.long_hidden_dim = 36
    elif args.long_strength == 'semantic_micro_lite':
        args.long_self_weight = 0.001
        args.long_near_weight = 0.085
        args.pathsim_topk = 4
        args.pathsim_topk_ratio = 0.06
        args.pathsim_min_topk = 2
        args.pair_long_weight = 0.26
        if cli_args.long_encoder_mode is None:
            args.long_encoder_mode = 'semantic'
        args.n_mamba_layers = 1
        args.long_hidden_dim = 34
    elif args.long_strength == 'semantic_ultra_lite':
        args.long_self_weight = 0.0
        args.long_near_weight = 0.08
        args.pathsim_topk = 4
        args.pathsim_topk_ratio = 0.06
        args.pathsim_min_topk = 2
        args.pair_long_weight = 0.25
        if cli_args.long_encoder_mode is None:
            args.long_encoder_mode = 'semantic'
        args.n_mamba_layers = 1
        args.long_hidden_dim = 32
    elif args.long_strength == 'minimal':
        args.long_self_weight = 0.05
        args.long_near_weight = 0.25
        args.pathsim_topk = 10
        args.pathsim_topk_ratio = 0.06
        args.pathsim_min_topk = 2
        args.pair_long_weight = 0.45
        if cli_args.long_encoder_mode is None:
            args.long_encoder_mode = 'semantic'
        args.n_mamba_layers = 2
        args.long_hidden_dim = 64
    elif args.long_strength == 'compact':
        args.long_self_weight = 0.15
        args.long_near_weight = 0.45
        args.pathsim_topk = 20
        args.pathsim_topk_ratio = 0.06
        args.pathsim_min_topk = 2
        args.pair_long_weight = 0.6
        if cli_args.long_encoder_mode is None:
            args.long_encoder_mode = 'semantic'
        args.n_mamba_layers = 2
        args.long_hidden_dim = 64
    elif args.long_strength == 'original':
        args.long_self_weight = 0.35
        args.long_near_weight = 0.75
        args.pathsim_topk = 40
        args.pathsim_topk_ratio = 0.06
        args.pathsim_min_topk = 2
        args.pair_long_weight = 0.8
        if cli_args.long_encoder_mode is None:
            args.long_encoder_mode = 'semantic'
        args.n_mamba_layers = 2
        args.long_hidden_dim = 64

    if args.short_strength == 'type_regularized':



        args.short_second_layer = 'type_attn'
        args.short_use_log_degree_penalty = False
        if cli_args.short_sample_rate is None:
            args.sample_rate_drug = [5]
            args.sample_rate_fc = [5]
        if cli_args.short_self_weight is None:
            args.short_self_weight = 0.70
        if cli_args.short_bottleneck_dim is None:
            args.short_bottleneck_dim = 0
        if cli_args.short_output_drop is None:
            args.short_output_drop = 0.08
        args.pair_short_weight = 0.60
        args.pair_synergy_weight = 1.15
    elif args.short_strength == 'type_complement':
        args.short_second_layer = 'type_attn'
        args.short_use_log_degree_penalty = True
        if cli_args.short_sample_rate is None:
            args.sample_rate_drug = [4]
            args.sample_rate_fc = [4]
        if cli_args.short_output_drop is None:
            args.short_output_drop = 0.0
        if cli_args.short_self_weight is None:
            args.short_self_weight = 0.35
        if cli_args.short_bottleneck_dim is None:
            args.short_bottleneck_dim = 24
        args.pair_short_weight = 0.55
        args.pair_synergy_weight = 1.20
    elif args.short_strength == 'type_balanced':
        args.short_second_layer = 'type_attn'
        args.short_use_log_degree_penalty = False
        if cli_args.short_sample_rate is None:
            args.sample_rate_drug = [6]
            args.sample_rate_fc = [6]
        if cli_args.short_output_drop is None:
            args.short_output_drop = 0.06
        if cli_args.short_self_weight is None:
            args.short_self_weight = 0.80
        if cli_args.short_bottleneck_dim is None:
            args.short_bottleneck_dim = 24
        args.pair_short_weight = 0.65
        args.pair_synergy_weight = 1.15
    elif args.short_strength == 'relation_strong':
        args.short_use_log_degree_penalty = False
        if cli_args.short_sample_rate is None:
            args.sample_rate_drug = list(args.sample_rate_drug)
            args.sample_rate_fc = list(args.sample_rate_fc)
        if cli_args.short_output_drop is None:
            args.short_output_drop = 0.15
        if cli_args.short_self_weight is None:
            args.short_self_weight = 0.40
        if cli_args.short_bottleneck_dim is None:
            args.short_bottleneck_dim = int(args.short_bottleneck_dim)
        args.pair_short_weight = 0.8
        args.pair_synergy_weight = 1.0
    elif args.short_strength == 'type_light':
        args.short_second_layer = 'type_attn'
        args.short_use_log_degree_penalty = False
        if cli_args.short_sample_rate is None:
            args.sample_rate_drug = [4]
            args.sample_rate_fc = [4]
        if cli_args.short_output_drop is None:
            args.short_output_drop = 0.0
        if cli_args.short_self_weight is None:
            args.short_self_weight = 0.60
        if cli_args.short_bottleneck_dim is None:
            args.short_bottleneck_dim = 16
        args.pair_short_weight = 0.55
        args.pair_synergy_weight = 1.25
    if cli_args.no_short_degree_penalty:
        args.short_use_log_degree_penalty = False

    if cli_args.dataset == 'drugbank' and args.short_strength == 'relation_strong':
        if cli_args.short_sample_rate is None:
            args.sample_rate_drug = [4]
            args.sample_rate_fc = [4]
        if cli_args.short_output_drop is None:
            args.short_output_drop = 0.12
        if cli_args.short_self_weight is None:
            args.short_self_weight = 0.35
        if cli_args.short_bottleneck_dim is None:
            args.short_bottleneck_dim = 5
        if cli_args.short_relation_drop is None:
            args.short_relation_drop = 0.08

    if args.long_strength in ('m6', 'm7'):
        if cli_args.short_sample_rate is None:
            rate = 4 if args.long_strength == 'm6' else 3
            args.sample_rate_drug = [rate]
            args.sample_rate_fc = [rate]
        if cli_args.short_self_weight is None:
            args.short_self_weight = 0.30 if args.long_strength == 'm6' else 0.20
        if cli_args.short_bottleneck_dim is None:
            args.short_bottleneck_dim = 5
        if cli_args.short_output_drop is None:
            args.short_output_drop = 0.18 if args.long_strength == 'm6' else 0.20
        if cli_args.short_relation_drop is None:
            args.short_relation_drop = 0.12 if args.long_strength == 'm6' else 0.15

    if args.long_strength in ('db1', 'db2', 'db3'):



        args.epochs = 140
        args.lr = 0.0011
        args.tau = 3.0
        if args.long_strength == 'db1':
            rate, self_w, sdrop, rdrop = 4, 0.25, 0.16, 0.12
        elif args.long_strength == 'db2':
            rate, self_w, sdrop, rdrop = 3, 0.18, 0.20, 0.15
        else:
            rate, self_w, sdrop, rdrop = 4, 0.30, 0.16, 0.12
            args.short_view_source = 'bridge'
        if cli_args.short_sample_rate is None:
            args.sample_rate_drug = [rate]
            args.sample_rate_fc = [rate]
        if cli_args.short_self_weight is None:
            args.short_self_weight = self_w
        if cli_args.short_bottleneck_dim is None:
            args.short_bottleneck_dim = 5
        if cli_args.short_output_drop is None:
            args.short_output_drop = sdrop
        if cli_args.short_relation_drop is None:
            args.short_relation_drop = rdrop

    if cli_args.dataset == 'pubmed' and not cli_args.custom_long_params and args.long_strength not in ('m1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'db1', 'db2', 'db3'):
        args.pathsim_topk = 2
        args.long_self_weight = 0.012

    if cli_args.custom_long_params:
        args.long_self_weight = cli_args.long_self_weight
        args.long_near_weight = cli_args.long_near_weight
        args.long_second_weight = cli_args.long_second_weight
        args.long_late_weight = cli_args.long_late_weight
        args.pathsim_topk = cli_args.pathsim_topk
        args.pathsim_topk_ratio = cli_args.pathsim_topk_ratio
        args.pathsim_min_topk = cli_args.pathsim_min_topk
        args.pathsim_min_sim = cli_args.pathsim_min_sim
        args.long_small_type_threshold = cli_args.long_small_type_threshold
        args.long_small_type_floor = cli_args.long_small_type_floor
        args.long_output_drop = cli_args.long_output_drop
        args.long_meta_path_drop = cli_args.long_meta_path_drop
        args.long_hidden_dim = cli_args.long_hidden_dim
        args.n_mamba_layers = cli_args.n_mamba_layers
        if cli_args.long_encoder_mode is not None:
            args.long_encoder_mode = cli_args.long_encoder_mode













    if (cli_args.task == 'dfc' and args.long_strength == 'semantic_restore_lite'
            and not cli_args.custom_long_params):
        args.num_hops = 5
        args.long_hidden_dim = 24
        args.n_mamba_layers = 1
        args.long_encoder_mode = 'semantic'


        args.long_self_weight = 0.012 if cli_args.dataset == 'pubmed' else 0.0
        args.long_near_weight = 0.02
        args.long_second_weight = 1.0
        args.long_late_weight = 1.0
        args.pathsim_topk = 2 if cli_args.dataset == 'pubmed' else 1
        args.pathsim_topk_ratio = 0.03
        args.pathsim_min_topk = 1
        args.pathsim_min_sim = 0.0
        args.long_output_drop = 0.15
        args.short_view_source = 'target'
        args.short_second_layer = 'relation_gat'
        args.short_strength = 'relation_strong'
        args.sample_rate_drug = [5] if cli_args.dataset == 'pubmed' else [4]
        args.sample_rate_fc = [5] if cli_args.dataset == 'pubmed' else [4]
        args.short_bottleneck_dim = 5
        args.short_relation_drop = 0.10 if cli_args.dataset == 'pubmed' else 0.08
        args.short_output_drop = 0.15 if cli_args.dataset == 'pubmed' else 0.12
        args.short_self_weight = 0.40 if cli_args.dataset == 'pubmed' else 0.35
        args.view_decorrelation_weight = 0.001 if cli_args.dataset == 'pubmed' else 0.0
        args.cross_view_residual_scale = 1.0 if cli_args.dataset == 'pubmed' else 0.8



        if cli_args.epochs is None:
            args.epochs = 140 if cli_args.dataset == 'drugbank' else 100
        if cli_args.lp_dropout is None:
            args.lp_dropout = 0.20 if cli_args.dataset == 'pubmed' else 0.16
        if cli_args.lr is None and cli_args.dataset == 'pubmed':
            args.lr = 0.0009
        args.dataset_profile = (
            'dhmmdfi_pubmed_dfc_paper'
            if cli_args.dataset == 'pubmed'
            else 'dhmmdfi_drugbank_dfc_paper'
        )


    args.pathsim_mode = 'pathsim'
    if args.ablation == 'mamba_mlp':


        args.long_encoder_mode = 'mlp_ultra'
    elif args.ablation == 'mamba_mlp_strict':
        if cli_args.dataset != 'drugbank':
            raise ValueError('mamba_mlp_strict is restricted to dataset=drugbank')
        args.long_encoder_mode = 'mlp_strict'
    elif args.ablation == 'mamba_mlp_ultra':

        args.long_encoder_mode = 'mlp_strict'
    elif args.ablation in ('mamba_mean', 'wo_mamba'):
        args.long_encoder_mode = 'mean'
    elif args.ablation == 'wo_pathsim':
        args.pathsim_mode = 'raw_binary'


def build_run_options(cli_args):
    apply_cli_overrides(cli_args)
    return {
        'dataset': cli_args.dataset,
        'task': cli_args.task,
        'food_feature_mode': 'random',
        'neg_ratio': cli_args.neg_ratio,
        'meta_mode': cli_args.meta_mode,
        'neg_strategy': 'global_random',
        'neg_seed': cli_args.neg_seed,
        'quick_folds': cli_args.quick_folds,
        'protocol_desc': (
            f'classic 5-fold edge split (8:2, no validation); task={cli_args.task}; '
            f'dataset_profile={getattr(args, "dataset_profile", "default")}; '
            f'food_feature_mode=random; fusion_mode={args.fusion_mode}; predictor_type={args.predictor_type}; '
            f'predictor_edge_ops={args.predictor_edge_ops}; '
            f'long_view_mode={cli_args.long_view_mode}; short_strength={args.short_strength}; '
            f'short_second_layer={args.short_second_layer}; short_sample_rate={args.sample_rate_drug}; '
            f'short_view_source={args.short_view_source}; '
            f'short_output_drop={getattr(args, "short_output_drop", 0.0)}; '
            f'short_self_weight={getattr(args, "short_self_weight", 1.0)}; '
            f'short_bottleneck_dim={getattr(args, "short_bottleneck_dim", 0)}; '
            f'short_relation_drop={getattr(args, "short_relation_drop", 0.0)}; '
            f'num_hops={args.num_hops}; long_hidden_dim={args.long_hidden_dim}; '
            f'n_mamba_layers={args.n_mamba_layers}; long_encoder_mode={args.long_encoder_mode}; '
            f'mamba_dropout={args.mamba_dropout}; '
            f'long_strength={args.long_strength}; long_hop_mode=soft_context; '
            f'long_self_weight={args.long_self_weight}; long_near_weight={args.long_near_weight}; '
            f'long_second_weight={args.long_second_weight}; '
            f'long_late_weight={args.long_late_weight}; '
            f'pathsim_mode={args.pathsim_mode}; pathsim_topk={args.pathsim_topk}; pathsim_min_sim={args.pathsim_min_sim}; '
            f'pathsim_topk_ratio={args.pathsim_topk_ratio}; pathsim_min_topk={args.pathsim_min_topk}; '
            f'long_small_type_threshold={args.long_small_type_threshold}; '
            f'long_small_type_floor={args.long_small_type_floor}; '
            f'long_output_drop={args.long_output_drop}; view_norm={args.view_norm}; '
            f'long_meta_path_drop={args.long_meta_path_drop}; '
            f'cross_view_residual={args.cross_view_residual}; '
            f'cross_view_residual_scale={args.cross_view_residual_scale}; '
            f'cl_align_mode={cli_args.cl_align_mode}; '
            f'cl_pos_mode={args.cl_pos_mode}; '
            f'tau={args.tau}; '
            f'view_decorrelation_weight={args.view_decorrelation_weight}; '
            f'lr={args.lr}; '
            f'grad_clip={args.grad_clip}; '
            f'feat_drop={args.feat_drop}; attn_drop={args.attn_drop}; '
            f'lp_hidden={args.lp_hidden}; lp_dropout={args.lp_dropout}; '
            f'ablation={args.ablation}; '
            f'logit_level_ablation={args.logit_level_ablation}; '
            f'fixed global_random train/test negative folds over target pairs; '
            f'neg_seed={cli_args.neg_seed}; positive:negative ratio=1:{cli_args.neg_ratio}'
        ),
    }
