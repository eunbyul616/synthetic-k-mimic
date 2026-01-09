import os
from pathlib import Path
import torch

import config_manager

from Trainer.TabularEhrGen.train_static_vae import StaticVAETrainer
from Trainer.TabularEhrGen.train_temporal_vae import TemporalVAETrainer
from Trainer.TabularEhrGen.train_conditional_gan import load_embedders
from Models.TabularEhrGen.ConditionalGAN import build_model

from Datasets.dataset_k_mimic import KMIMICDataset as CustomDataset
from Utils.namespace import _load_yaml
from Utils.reproducibility import lock_seed
from Evaluation.DistributionSimilarity.distribution_comparison import *


def get_batch(batch, device):
    sc, tc, sn, tn, sl, tl, sm, tm, condition = batch

    sc = sc.to(device)
    tc = tc.to(device)
    sn = sn.to(device)
    tn = tn.to(device)
    sl = sl.to(device)
    tl = tl.to(device)
    sm = sm.to(device)
    tm = tm.to(device)
    condition = condition.to(device)

    return sc, tc, sn, tn, sl, tl, sm, tm, condition


def get_static_target(sn_data, sc_data, sl_data, static_mask, static_categorical_card):
    sc_target = []
    sb_target = []
    s_idx = 0
    for i, dim in enumerate(static_categorical_card[:-2]):
        if dim > 1:
            sc_target.append(sc_data[:, s_idx:s_idx + dim])
        else:
            sb_target.append(sc_data[:, s_idx:s_idx + dim])
        s_idx += dim
    sc_target = torch.cat(sc_target, dim=-1)
    if len(sb_target) > 0:
        sb_target = torch.cat(sb_target, dim=-1)
    else:
        sb_target = torch.empty((sc_target.shape[0], 0), device=sc_target.device)

    data = torch.cat([sn_data, sc_target, sb_target, sl_data, static_mask], dim=-1)

    return data


def get_temporal_target(tn_data, tc_data, tl_data, temporal_mask, temporal_categorical_card):
    tc_target = []
    tb_target = []
    s_idx = 0
    for i, dim in enumerate(temporal_categorical_card[:-1]):
        if dim > 1:
            tc_target.append(tc_data[:, :, s_idx:s_idx + dim])
        else:
            tb_target.append(tc_data[:, :, s_idx:s_idx + dim])
        s_idx += dim

    tc_target = torch.cat(tc_target, dim=-1)
    if len(tb_target) > 0:
        tb_target = torch.cat(tb_target, dim=-1)
    else:
        tb_target = torch.empty(tc_target.shape[0], tc_target.shape[1], 0, device=tc_target.device)

    data = torch.cat([tn_data, tc_target, tb_target, tl_data, temporal_mask], dim=-1)

    return data

def activate_static_hat(x_num_hat, x_cat_logits, x_bin_logits, x_listwise_logits, x_mask_logits,
                        use_gumbel=True, logit_threshold=0.5):
    act_x_hat = []
    act_x_hat.append(x_num_hat)
    for sc_hat in x_cat_logits:
        _act_sc_hat = torch.softmax(sc_hat, dim=-1)
        _act_sc_hat = (_act_sc_hat >= logit_threshold).float()
        act_x_hat.append(_act_sc_hat)

    for sb_hat in x_bin_logits:
        if use_gumbel:
            _act_sb_hat = sb_hat
        else:
            _act_sb_hat = torch.sigmoid(sb_hat)
            _act_sb_hat = (_act_sb_hat >= logit_threshold).float()
        act_x_hat.append(_act_sb_hat)

    for sl_hat in x_listwise_logits:
        if use_gumbel:
            _act_sl_hat = sl_hat
        else:
            _act_sl_hat = torch.sigmoid(sl_hat)
            _act_sl_hat = (_act_sl_hat >= logit_threshold).float()
        act_x_hat.append(_act_sl_hat)

    for sm_hat in x_mask_logits:
        if use_gumbel:
            _act_sm_hat = sm_hat
        else:
            _act_sm_hat = torch.sigmoid(sm_hat)
            _act_sm_hat = (_act_sm_hat >= logit_threshold).float()
        act_x_hat.append(_act_sm_hat)
    act_x_hat = torch.cat(act_x_hat, dim=-1)

    return act_x_hat

def activate_temporal_hat(x_num_hat, x_cat_logits, x_bin_logits, x_listwise_logits, x_mask_logits,
                        use_gumbel=True, logit_threshold=0.5):
    act_x_hat = []
    act_x_hat.append(x_num_hat)
    for tc_hat in x_cat_logits:
        _act_tc_hat = torch.softmax(tc_hat, dim=-1)
        _act_tc_hat = (_act_tc_hat >= logit_threshold).float()
        act_x_hat.append(_act_tc_hat)

    for tb_hat in x_bin_logits:
        _act_tb_hat = torch.sigmoid(tb_hat)
        _act_tb_hat = (_act_tb_hat >= logit_threshold).float()
        act_x_hat.append(_act_tb_hat)

    for tl_hat in x_listwise_logits:
        if use_gumbel:
            _act_tl_hat = tl_hat
        else:
            _act_tl_hat = torch.sigmoid(tl_hat)
            _act_tl_hat = (_act_tl_hat >= logit_threshold).float()
        act_x_hat.append(_act_tl_hat)

    for tm_hat in x_mask_logits:
        if use_gumbel:
            _act_tm_hat = tm_hat
        else:
            _act_tm_hat = torch.sigmoid(tm_hat)
            _act_tm_hat = (_act_tm_hat >= logit_threshold).float()
        act_x_hat.append(_act_tm_hat)
    act_x_hat = torch.cat(act_x_hat, dim=-1)

    return act_x_hat

def get_static_categorical_card(dataset, static_transformer):
    sc_cols = dataset.sc_cols
    sn_cols = dataset.sn_cols
    sl_cols = dataset.sl_cols

    # static_categorical_card
    categorical_feature_info = []
    for c in (sc_cols + sl_cols):
        info = [info for info in static_transformer._data_manipulation_info_list if info.column_name == c][0]
        categorical_feature_info.append(info)
    static_categorical_card = [info.output_dimensions for info in categorical_feature_info]

    return static_categorical_card

def get_temporal_categorical_card(dataset, temporal_transformer):
    tn_cols = dataset.tn_cols
    tc_cols = dataset.tc_cols
    tl_cols = dataset.tl_cols

    # temporal_categorical_card
    categorical_feature_info = []
    for c in (tc_cols + tl_cols):
        info = [info for info in temporal_transformer._data_manipulation_info_list if info.column_name == c][0]
        categorical_feature_info.append(info)
    temporal_categorical_card = [info.output_dimensions for info in categorical_feature_info]

    return temporal_categorical_card


def get_static_info(dataset):
    sc_cols = dataset.sc_cols
    sn_cols = dataset.sn_cols
    sl_cols = dataset.sl_cols

    sc_cols = []
    sb_cols = []
    for c in dataset.sc_cols:
        info = [info for info in static_transformer._data_manipulation_info_list if info.column_name == c][0]
        dim = info.output_dimensions
        if dim == 1:
            sb_cols.append(c)
        else:
            sc_cols.append(c)

    categorical_feature_info = []
    for c in (sn_cols + sc_cols + sb_cols + sl_cols):
        info = [info for info in static_transformer._data_manipulation_info_list if info.column_name == c][0]
        categorical_feature_info.append(info)
    for c in (dataset.sc_cols + sl_cols + sn_cols):
        info = [info for info in static_transformer._data_manipulation_info_list if
                info.column_name == f'{c}_mask'][0]
        categorical_feature_info.append(info)
    static_feature_info = categorical_feature_info

    return static_feature_info


def get_temporal_info(dataset):
    tn_cols = dataset.tn_cols
    tc_cols = dataset.tc_cols
    tl_cols = dataset.tl_cols

    tc_cols = []
    tb_cols = []
    for c in dataset.tc_cols:
        info = [info for info in temporal_transformer._data_manipulation_info_list if info.column_name == c][0]
        dim = info.output_dimensions
        if dim == 1:
            tb_cols.append(c)
        else:
            tc_cols.append(c)

    categorical_feature_info = []
    for c in (tn_cols + tc_cols + tb_cols + tl_cols):
        info = [info for info in temporal_transformer._data_manipulation_info_list if info.column_name == c][0]
        categorical_feature_info.append(info)
    for c in (dataset.tc_cols + tl_cols + tn_cols):
        info = [info for info in temporal_transformer._data_manipulation_info_list
                if info.column_name == f'{c}_mask'][0]
        categorical_feature_info.append(info)
    temporal_feature_info = categorical_feature_info

    return temporal_feature_info


def generate_samples(checkpoint_cfg,  dataset, static_ae, temporal_ae, gan, n_samples, use_gumbel=True, logit_threshold=0.5):
    n = n_samples // len(dataset)

    seq_len = checkpoint_cfg.dataloader.seq_len
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=checkpoint_cfg.dataloader.batch_size,
                                             shuffle=False)
    static_transformer = dataset.static_transformer
    temporal_transformer = dataset.temporal_transformer

    sc_cols = dataset.sc_cols
    tc_cols = dataset.tc_cols
    sn_cols = dataset.sn_cols
    tn_cols = dataset.tn_cols
    sl_cols = dataset.sl_cols
    tl_cols = dataset.tl_cols

    static_categorical_card = get_static_categorical_card(dataset, static_transformer)
    temporal_categorical_card = get_temporal_categorical_card(dataset, temporal_transformer)

    static_feature_info = get_static_info(dataset)
    temporal_feature_info = get_temporal_info(dataset)

    conditions = []
    with torch.inference_mode():
        for batch in dataloader:
            sc, tc, sn, tn, sl, tl, sm, tm, condition = get_batch(batch, device=torch.device(f'cuda:{checkpoint_cfg.device_num}'))
            conditions.append(condition.detach().cpu())
    conditions = torch.cat(conditions, dim=0)
    conditions = conditions.repeat(n, 1)

    diagnoses_prefix = checkpoint_cfg.preprocess.icd_code.diagnoses_prefix
    procedure_prefix = checkpoint_cfg.preprocess.icd_code.procedure_prefix
    proc_prefix = checkpoint_cfg.preprocess.proc_prefix

    batch_size = checkpoint_cfg.dataloader.batch_size
    data_hat = {'static_data': [], 'temporal_data': []}
    device = torch.device(f'cuda:{checkpoint_cfg.device_num}')
    with torch.inference_mode():
        for s_idx in range(0, n_samples, batch_size):
            bsz = min(batch_size, n_samples - s_idx)
            fake = gan.generator.sample(n_samples=bsz,
                                        condition=conditions[s_idx:s_idx + bsz].to(device),
                                        device=device)
            idx = static_ae.decoder.fc1.in_features
            s_num_hat, s_cat_hat, s_bin_hat, s_listwise_hat, s_mask_hat = static_ae.decoder(fake[:, :idx], hard=True)
            t_num_hat, t_cat_hat, t_bin_hat, t_listwise_hat, t_mask_hat = temporal_ae.decoder(fake[:, idx:],
                                                                                              seq_len=seq_len,
                                                                                              hard=True,
                                                                                              z_s=fake[:, :idx])

            s_act_x_hat = activate_static_hat(s_num_hat, s_cat_hat, s_bin_hat, s_listwise_hat, s_mask_hat,
                                              use_gumbel=use_gumbel, logit_threshold=logit_threshold)
            data_hat['static_data'].append(s_act_x_hat.detach().cpu())
            t_act_x_hat = activate_temporal_hat(t_num_hat, t_cat_hat, t_bin_hat, t_listwise_hat, t_mask_hat,
                                                use_gumbel=use_gumbel, logit_threshold=logit_threshold)
            data_hat['temporal_data'].append(t_act_x_hat.detach().cpu())

        data_hat['static_data'] = torch.concatenate(data_hat['static_data'], dim=0)
        data_hat['temporal_data'] = torch.concatenate(data_hat['temporal_data'], dim=0)

    temporal_data = dataset.temporal_data
    mask = ~pd.isna(temporal_data)
    mask = mask.values
    feature_num = mask.shape[-1]
    reshape_mask = mask.reshape(-1, seq_len, feature_num)
    indices = torch.randint(0, reshape_mask.shape[0], (n_samples,))
    sampled_mask = reshape_mask[indices]
    sampled_mask = sampled_mask.reshape(-1, feature_num)

    static_data_hat, temporal_data_hat = inverse_transform_samples(data_hat, static_transformer,
                                                                   temporal_transformer,
                                                                   static_feature_info,
                                                                   temporal_feature_info,
                                                                   mask=sampled_mask)

    # expire_flag
    static_data_hat['icu_expire_flag'] = conditions[:, 1].detach().cpu().numpy()
    static_data_hat['hospital_expire_flag'] = conditions[:, 1:].detach().cpu().numpy().max(axis=-1)

    return static_data_hat, temporal_data_hat


def inverse_transform_samples(data_hat, static_transformer, temporal_transformer,
                              static_feature_info, temporal_feature_info, mask=None):
    static_data_hat = static_transformer.inverse_transform(data_hat['static_data'].detach().cpu().numpy(),
                                                           static_feature_info)
    feature_dim = data_hat['temporal_data'].shape[-1]
    temporal_data_hat = temporal_transformer.inverse_transform(
        data_hat['temporal_data'].view(-1, feature_dim).detach().cpu().numpy(),
        temporal_feature_info
    )

    if mask is not None:
        temporal_data_hat[~mask.all(axis=1)] = np.nan

    return static_data_hat, temporal_data_hat

if __name__ == '__main__':
    config_manager.load_config()
    eval_cfg = config_manager.config
    lock_seed(seed=eval_cfg.seed, multi_gpu=False, activate_cudnn=False)
    cols = None

    dataset_name = eval_cfg.dataset.dataset_name
    fname = eval_cfg.preprocess.preprocess_fname_suffix
    train_ratio, test_ratio = eval_cfg.preprocess.train_valid_test_split_ratio
    dataset_fname = f'{dataset_name}_{fname}_{int(train_ratio * 10)}.h5'

    dataset = CustomDataset(cfg=eval_cfg,
                            dataset_name=dataset_name,
                            dataset_fname=dataset_fname,
                            mode='test',
                            condition_col=eval_cfg.data.condition_col,
                            static_cols=None)

    model_name = eval_cfg.evaluation.model_name
    checkpoint = eval_cfg.evaluation.checkpoint
    print(checkpoint)

    csv_save_path = os.path.join(eval_cfg.path.eval_file_path, model_name, checkpoint)
    print(csv_save_path)
    os.makedirs(csv_save_path, exist_ok=True)

    model_checkpoint_path = os.path.join('/'.join(Path(eval_cfg.path.ckpt_path).parts[:-1]), model_name, checkpoint)
    cfg = _load_yaml(os.path.join(model_checkpoint_path, 'config.yaml'))
    static_ae, temporal_ae = load_embedders(cfg)
    gan = build_model(cfg.model.gan, device=torch.device(f'cuda:{cfg.device_num}'))
    gan_checkpoint = torch.load(os.path.join(model_checkpoint_path, 'checkpoint.pth.tar'),
                                map_location=f'cuda:{cfg.device_num}')
    gan.load_state_dict(gan_checkpoint['state_dict'])

    static_ae.eval()
    temporal_ae.eval()
    gan.eval()

    use_gumbel = cfg.model.static_autoencoder.use_gumbel
    logit_threshold = cfg.model.static_autoencoder.logit_threshold
    seq_len = cfg.dataloader.seq_len
    dataloader = torch.utils.data.DataLoader(dataset, 
                                             batch_size=cfg.dataloader.batch_size, 
                                             shuffle=False)
    static_transformer = dataset.static_transformer
    temporal_transformer = dataset.temporal_transformer

    sc_cols = dataset.sc_cols
    tc_cols = dataset.tc_cols
    sn_cols = dataset.sn_cols
    tn_cols = dataset.tn_cols
    sl_cols = dataset.sl_cols
    tl_cols = dataset.tl_cols

    static_categorical_card = get_static_categorical_card(dataset, static_transformer)
    temporal_categorical_card = get_temporal_categorical_card(dataset, temporal_transformer)

    static_feature_info = get_static_info(dataset)
    temporal_feature_info = get_temporal_info(dataset)

    diagnoses_prefix = cfg.preprocess.icd_code.diagnoses_prefix
    procedure_prefix = cfg.preprocess.icd_code.procedure_prefix
    proc_prefix = cfg.preprocess.proc_prefix

    data = {'static_data': [], 'temporal_data': []}
    data_hat = {'static_data': [], 'temporal_data': []}
    conditions = []

    enc_z = []
    enc_z_hat = []
    with torch.no_grad():
        for batch in dataloader:
            sc, tc, sn, tn, sl, tl, sm, tm, condition = get_batch(batch, device=torch.device(f'cuda:{cfg.device_num}'))
            (sc_data_target, sc_data_converted,
             sb_data_target, sb_data_converted) = StaticVAETrainer.onehot_to_index_with_mask(sc,
                                                                                            static_categorical_card)

            (s_num_hat, s_cat_logits, s_bin_logits, s_listwise_logits, s_mask_logits,
             s_mu, s_logvar, s_z, s_condition_hat) = static_ae(x_num=sn,
                                                               x_cat=sc_data_converted,
                                                               x_bin=sb_data_converted,
                                                               x_listwise=sl,
                                                               x_mask=sm,
                                                               use_gumbel=use_gumbel)
            (tc_data_target, tc_data_converted,
             tb_data_target, tb_data_converted) = TemporalVAETrainer.temporal_onehot_to_index_with_mask(tc,
                                                                                                       temporal_categorical_card)
            (t_num_hat, t_cat_logits, t_bin_logits, t_listwise_logits, t_mask_logits,
             t_mu, t_logvar, t_z, t_condition_hat) = temporal_ae(x_num=tn,
                                                                 x_cat=tc_data_converted,
                                                                 x_bin=tb_data_converted,
                                                                 x_listwise=tl,
                                                                 x_mask=tm,
                                                                 use_gumbel=use_gumbel,
                                                                 z_s=s_z)
            z = torch.cat([s_z, t_z], dim=-1)
            out = gan(z, condition=condition)
            fake = out['fake']
            static_x = get_static_target(sn, sc, sl, sm, static_categorical_card)
            data['static_data'].append(static_x)

            temporal_x = get_temporal_target(tn, tc, tl, tm, temporal_categorical_card)
            data['temporal_data'].append(temporal_x)

            enc_z.append(z.detach().cpu())
            enc_z_hat.append(fake.detach().cpu())

            # decoding
            idx = static_ae.decoder.fc1.in_features
            s_num_hat, s_cat_hat, s_bin_hat, s_listwise_hat, s_mask_hat = static_ae.decoder(fake[:, :idx], hard=True)
            t_num_hat, t_cat_hat, t_bin_hat, t_listwise_hat, t_mask_hat = temporal_ae.decoder(fake[:, idx:],
                                                                                              seq_len=seq_len,
                                                                                              hard=True,
                                                                                              z_s=fake[:, :idx])

            s_act_x_hat = activate_static_hat(s_num_hat, s_cat_hat, s_bin_hat, s_listwise_hat, s_mask_hat,
                                              use_gumbel=use_gumbel, logit_threshold=logit_threshold)
            data_hat['static_data'].append(s_act_x_hat)
            t_act_x_hat = activate_temporal_hat(t_num_hat, t_cat_hat, t_bin_hat, t_listwise_hat, t_mask_hat,
                                                use_gumbel=use_gumbel, logit_threshold=logit_threshold)
            data_hat['temporal_data'].append(t_act_x_hat)
            conditions.append(condition)

        data['static_data'] = torch.concatenate(data['static_data'], dim=0)
        data['temporal_data'] = torch.concatenate(data['temporal_data'], dim=0)
        data_hat['static_data'] = torch.concatenate(data_hat['static_data'], dim=0)
        data_hat['temporal_data'] = torch.concatenate(data_hat['temporal_data'], dim=0)
        conditions = torch.concatenate(conditions, dim=0)
        enc_z = torch.concatenate(enc_z, dim=0)
        enc_z_hat = torch.concatenate(enc_z_hat, dim=0)

        static_data = static_transformer.inverse_transform(data['static_data'].detach().cpu().numpy(),
                                                                static_feature_info)
        static_data_hat = static_transformer.inverse_transform(data_hat['static_data'].detach().cpu().numpy(),
                                                                    static_feature_info)

        # expire_flag
        static_data['icu_expire_flag'] = conditions[:, 1].detach().cpu().numpy()
        static_data['hospital_expire_flag'] = conditions[:, 1:].detach().cpu().numpy().max(axis=-1)

        static_data_hat['icu_expire_flag'] = conditions[:, 1].detach().cpu().numpy()
        static_data_hat['hospital_expire_flag'] = conditions[:, 1:].detach().cpu().numpy().max(axis=-1)

        mask = data['temporal_data'] != cfg.dataloader.pad_value
        mask = mask.view(-1, data['temporal_data'].shape[-1]).detach().cpu().numpy()
        feature_dim = data['temporal_data'].shape[-1]
        temporal_data = temporal_transformer.inverse_transform(
            data['temporal_data'].view(-1, feature_dim).detach().cpu().numpy(), temporal_feature_info)
        temporal_data_hat = temporal_transformer.inverse_transform(
            data_hat['temporal_data'].view(-1, feature_dim).detach().cpu().numpy(), temporal_feature_info)

        # temporal_data = temporal_data[mask.all(axis=1)]
        # temporal_data_hat = temporal_data_hat[mask.all(axis=1)]
        temporal_data[~mask.all(axis=1)] = np.nan
        temporal_data_hat[~mask.all(axis=1)] = np.nan


    static_data.to_csv(os.path.join(csv_save_path, 'static_data.csv'), index=False)
    temporal_data.to_csv(os.path.join(csv_save_path, 'temporal_data.csv'), index=False)

    static_data_hat.to_csv(os.path.join(csv_save_path, 'static_reconstructed.csv'), index=False)
    temporal_data_hat.to_csv(os.path.join(csv_save_path, 'temporal_reconstructed.csv'), index=False)

    enc_z = enc_z.numpy()
    enc_z_hat = enc_z_hat.numpy()
    np.save(os.path.join(csv_save_path, 'enc_z.npy'), enc_z)
    np.save(os.path.join(csv_save_path, 'enc_z_hat.npy'), enc_z_hat)

    n = 20
    n_samples = len(dataset) * n
    static_data_hat, temporal_data_hat = generate_samples(cfg, dataset,
                                                          static_ae, temporal_ae, gan,
                                                          n_samples,
                                                          use_gumbel=use_gumbel,
                                                          logit_threshold=logit_threshold)

    static_data_hat.to_csv(os.path.join(csv_save_path, f'static_reconstructed_{n_samples}.csv'), index=False)
    temporal_data_hat.to_csv(os.path.join(csv_save_path, f'temporal_reconstructed_{n_samples}.csv'), index=False)
