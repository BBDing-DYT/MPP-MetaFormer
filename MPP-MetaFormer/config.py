def hyper_parameters_config_set(dataset_name='BACE', token_mixer_name='Self-Attention', use_pre_ln=False, use_double_route_residue=False):
    config = {}

    config['batch_size'] = 256
    config['single_label_data_set_path'] = "./datasets/" + dataset_name + "/" + dataset_name
    config['multi_label_data_set_path'] = "./datasets/" + dataset_name + "/" + dataset_name
    config['learning_rate'] = 1e-4
    config['train_epoch'] = 100
    config['workers'] = 0

    config['drug_vocab_size'] = 80
    config['max_drug_seq'] = 100
    config['emb_size'] = 384
    config['dropout_rate'] = 0.2
    config['token_mixer_name'] = token_mixer_name

    config['drop_last_train'] = False
    config['drop_last_val_and_test'] = False
    config['use_pre_ln'] = use_pre_ln
    config['use_double_route_residue'] = use_double_route_residue

    return config
