import os

print(">>> USING PARAM FILE FROM:", os.path.abspath(__file__))

def get_params():
    print("USING CONFIGURATION: MIC + GCC + multi ACCDOA")

    params = dict(
        quick_test=False,
        finetune_mode=False,

        # Input/Output paths
        dataset_dir= os.path.normpath('/content/drive/MyDrive/dcase_2024'),
        feat_label_dir=os.path.normpath('/content/drive/MyDrive/dcase_2024/seld_feat_label/'),
        model_dir=os.path.normpath('models'),
        dcase_output_dir=os.path.normpath('results'),

        # Dataset
        mode='dev',
        dataset='mic',

        # Feature extraction
        fs=24000,
        hop_len_s=0.02,
        label_hop_len_s=0.1,
        max_audio_len_s=60,
        nb_mel_bins=64,
        use_salsalite=False,
        raw_chunks=False,
        saved_chunks=False,

        # Model
        model='seldnet',
        modality='audio',
        multi_accdoa=True,
        thresh_unify=15,

        # DNN
        label_sequence_length=50,
        batch_size=64,
        eval_batch_size=64,
        dropout_rate=0.05,
        nb_cnn2d_filt=64,
        f_pool_size=[4, 4, 2],

        nb_heads=8,
        nb_self_attn_layers=2,
        nb_transformer_layers=2,
        nb_rnn_layers=2,
        rnn_size=128,
        nb_fnn_layers=1,
        fnn_size=128,
        optimizer='adam',

        nb_epochs=300,
        eval_freq=25,
        lr=1e-3,
        final_lr=1e-5,
        weight_decay=0.05,

        predict_tdoa=False,
        warmup=5,
        relative_dist=True,
        no_dist=False,

        # Metrics
        average='macro',
        segment_based_metrics=False,
        evaluate_distance=True,
        lad_doa_thresh=20,
        lad_dist_thresh=float('inf'),
        lad_reldist_thresh=1.0,

        # Mic setup
        n_mics=4,
    )

    # Derived parameters
    params['feature_label_resolution'] = int(params['label_hop_len_s'] // params['hop_len_s'])
    params['feature_sequence_length'] = params['label_sequence_length'] * params['feature_label_resolution']
    params['t_pool_size'] = [params['feature_label_resolution'], 1, 1]
    params['patience'] = int(params['nb_epochs'])
    params['model_dir'] += '_' + params['modality']
    params['dcase_output_dir'] += '_' + params['modality']


    # Class count (based on dataset folder name)
    if '2024' in params['dataset_dir']:
        params['unique_classes'] = 13
    else:
        raise ValueError("Unknown dataset version in path.")

    # Print params for verification
    for key, value in params.items():
        print(f"\t{key}: {value}")

    return params

