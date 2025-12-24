
import os
from pathlib import Path
import numpy as np
import torch

from Trainer.utils import *
from Utils.reproducibility import lock_seed
from Datasets.dataset_k_mimic import KMIMICDataset as CustomDataset
from Trainer.EHRSafe.utils import (unpack_static_batch, unpack_temporal_batch, get_feature_info,
                                    apply_static_activation, apply_temporal_activation,
                                    get_temporal_categorical_card, get_static_categorical_card,
                                    split_data_by_type)


import config_manager


def load_model(cfg, model_name, checkpoint):
    from Utils.namespace import _load_yaml
    from Models.EHRSafe import StaticCategoricalAutoEncoder, TemporalCategoricalAutoEncoder, EHRSafeAutoEncoder, GAN

    model_checkpoint_path = os.path.join('/'.join(Path(cfg.path.ckpt_path).parts[:-1]), model_name, checkpoint)
    cfg = _load_yaml(os.path.join(model_checkpoint_path, 'config.yaml'))

    # load embedders
    checkpoint_saved_root = '/'.join(Path(cfg.path.ckpt_path).parts[:-1])

    # static categorical autoencoder
    static_cate_embedder_checkpoint_path = os.path.join(checkpoint_saved_root,
                                                        cfg.train.static_categorical_ae.name,
                                                        cfg.train.static_categorical_ae.checkpoint)
    static_cate_config = _load_yaml(os.path.join(static_cate_embedder_checkpoint_path, 'config.yaml'))
    static_cate_ae = StaticCategoricalAutoEncoder.build_model(static_cate_config.model.static_categorical_autoencoder,
                                                              device=torch.device(f'cuda:{cfg.device_num}'))
    static_cate_ae_checkpoint = torch.load(
        os.path.join(static_cate_embedder_checkpoint_path, 'checkpoint_best.pth.tar'),
        map_location=f'cuda:{cfg.device_num}')
    static_cate_ae.load_state_dict(static_cate_ae_checkpoint['state_dict'])

    # temporal categorical autoencoder
    temporal_cate_embedder_checkpoint_path = os.path.join(checkpoint_saved_root,
                                                          cfg.train.temporal_categorical_ae.name,
                                                          cfg.train.temporal_categorical_ae.checkpoint)
    temporal_cate_config = _load_yaml(os.path.join(temporal_cate_embedder_checkpoint_path, 'config.yaml'))
    temporal_cate_ae = TemporalCategoricalAutoEncoder.build_model(
        temporal_cate_config.model.temporal_categorical_autoencoder,
        device=torch.device(f'cuda:{cfg.device_num}'))
    temporal_cate_ae_checkpoint = torch.load(
        os.path.join(temporal_cate_embedder_checkpoint_path, 'checkpoint_best.pth.tar'),
        map_location=f'cuda:{cfg.device_num}')
    temporal_cate_ae.load_state_dict(temporal_cate_ae_checkpoint['state_dict'])

    # ehr safe autoencoder
    ehr_safe_embedder_checkpoint_path = os.path.join(checkpoint_saved_root,
                                                     cfg.train.ehr_safe_ae.name,
                                                     cfg.train.ehr_safe_ae.checkpoint)
    ehr_safe_config = _load_yaml(os.path.join(ehr_safe_embedder_checkpoint_path, 'config.yaml'))
    ehr_safe_ae = EHRSafeAutoEncoder.build_model(
        ehr_safe_config.model.ehr_safe_autoencoder,
        device=torch.device(f'cuda:{cfg.device_num}'))
    ehr_safe_ae_checkpoint = torch.load(
        os.path.join(ehr_safe_embedder_checkpoint_path, 'checkpoint_best.pth.tar'),
        map_location=f'cuda:{cfg.device_num}')
    ehr_safe_ae.load_state_dict(ehr_safe_ae_checkpoint['state_dict'])

    gan = GAN.build_model(cfg.model.gan, device=torch.device(f'cuda:{cfg.device_num}'))
    gan_checkpoint = torch.load(os.path.join(model_checkpoint_path, 'checkpoint.pth.tar'),
                                map_location=f'cuda:{cfg.device_num}')
    gan.load_state_dict(gan_checkpoint['state_dict'])

    static_cate_ae.eval()
    temporal_cate_ae.eval()
    ehr_safe_ae.eval()
    gan.eval()

    return static_cate_ae, temporal_cate_ae, ehr_safe_ae, gan, cfg


def generate_samples(dataset,
                     checkpoint_cfg,
                     static_cate_ae,
                     temporal_cate_ae,
                     ehr_safe_ae,
                     gan,
                     n_samples,
                     logit_threshold=0.5):

    device = torch.device(f'cuda:{checkpoint_cfg.device_num}' if torch.cuda.is_available() else 'cpu')
    seq_len = checkpoint_cfg.dataloader.seq_len

    static_transformer = dataset.static_transformer
    temporal_transformer = dataset.temporal_transformer

    (
        sc_feature_info,
        sn_feature_info,
        sm_feature_info,
        tc_feature_info,
        tn_feature_info,
        tm_feature_info,
    ) = get_feature_info(static_transformer, temporal_transformer)

    diagnoses_prefix = checkpoint_cfg.preprocess.icd_code.diagnoses_prefix
    procedure_prefix = checkpoint_cfg.preprocess.icd_code.procedure_prefix
    proc_prefix = checkpoint_cfg.preprocess.proc_prefix

    fake = gan.generator.sample(n_samples=n_samples, device=torch.device(f'cuda:{checkpoint_cfg.device_num}'))
    dec_fake = ehr_safe_ae.decoder(fake)

    sc_rep_dim = static_cate_ae.decoder.embedding_dim
    tc_rep_dim = temporal_cate_ae.decoder.embedding_dim
    sn_dim = sum([info.output_dimensions for info in sn_feature_info])
    tn_dim = sum([info.output_dimensions for info in tn_feature_info])
    sm_dim = sum([info.output_dimensions for info in sm_feature_info])
    tm_dim = sum([info.output_dimensions for info in tm_feature_info])
    (
        sc_rep_hat,
        tc_rep_hat,
        sn_hat,
        tn_hat,
        static_mask_hat,
        temporal_mask_hat
    ) = split_data_by_type(dec_fake, sc_rep_dim, tc_rep_dim, sn_dim, tn_dim, sm_dim, tm_dim, seq_len)

    # static reconstruction
    sc_hat = static_cate_ae.decoder(sc_rep_hat)
    sc_cols = dataset.sc_cols
    sl_cols = dataset.sl_cols
    sn_cols = dataset.sn_cols
    static_categorical_card, static_categorical_feature_info = get_static_categorical_card(dataset,
                                                                                           static_transformer)
    sc_binary_cols = [col.column_name for col in static_categorical_feature_info if
                      col.column_type == 'Binary']

    act_sc_hat, act_sn_hat, act_sm_hat = apply_static_activation(
        sc_cols=sc_cols,
        sc_binary_cols=sc_binary_cols,
        sl_cols=sl_cols,
        sc_hat=sc_hat,
        sn_hat=sn_hat,
        sm_hat=static_mask_hat,
        batch_size=sn_hat.shape[0],
        logit_threshold=logit_threshold)
    static_data_hat = torch.cat([act_sc_hat, act_sn_hat, act_sm_hat], dim=-1)

    # temporal reconstruction
    tc_hat = temporal_cate_ae.decoder(tc_rep_hat)
    tc_cols = dataset.tc_cols
    tl_cols = dataset.tl_cols
    tn_cols = dataset.tn_cols
    temporal_categorical_card, temporal_categorical_feature_info = get_temporal_categorical_card(
        dataset,
        temporal_transformer)
    temporal_categorical_feature_info = temporal_categorical_feature_info
    tc_binary_cols = [col.column_name for col in temporal_categorical_feature_info if
                      col.column_type == 'Binary']

    batch_size = tn_hat.shape[0]
    act_tc_hat, act_tn_hat, act_tm_hat = apply_temporal_activation(
        tc_cols=tc_cols,
        tc_binary_cols=tc_binary_cols,
        tl_cols=tl_cols,
        tc_hat=tc_hat,
        tn_hat=tn_hat.reshape(batch_size, seq_len, -1),
        tm_hat=temporal_mask_hat.reshape(batch_size, seq_len, -1),
        batch_size=tn_hat.shape[0],
        seq_len=seq_len,
        logit_threshold=logit_threshold)
    temporal_data_hat = torch.cat([act_tc_hat, act_tn_hat, act_tm_hat], dim=-1)

    static_feature_info = sc_feature_info + sn_feature_info + sm_feature_info
    temporal_feature_info = tc_feature_info + tn_feature_info + tm_feature_info

    data_hat = {
        'static_data': static_data_hat,
        'temporal_data': temporal_data_hat
    }

    static_data_hat, temporal_data_hat = inverse_transform_samples(data_hat,
                                                                   static_transformer, temporal_transformer,
                                                                   static_feature_info, temporal_feature_info)

    return static_data_hat, temporal_data_hat


def inverse_transform_samples(data_hat,
                              static_transformer, temporal_transformer,
                              static_feature_info, temporal_feature_info,
                              mask=None):

    static_data_hat = static_transformer.inverse_transform(
        data_hat['static_data'].detach().cpu().numpy(),
        static_feature_info
    )
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

    dataset_name = eval_cfg.dataset.dataset_name
    fname = eval_cfg.preprocess.preprocess_fname_suffix
    train_ratio, test_ratio = eval_cfg.preprocess.train_valid_test_split_ratio
    dataset_fname = f'{dataset_name}_{fname}_{int(train_ratio * 10)}.h5'
    cols = None

    dataset = CustomDataset(cfg=eval_cfg,
                            dataset_name=dataset_name,
                            dataset_fname=dataset_fname,
                            mode='test',
                            condition_col=getattr(eval_cfg.data, 'condition_col', None),
                            static_cols=cols)

    model_name = eval_cfg.evaluation.model_name
    checkpoint = eval_cfg.evaluation.checkpoint
    print(checkpoint)

    csv_save_path = os.path.join(eval_cfg.path.eval_file_path, model_name, checkpoint)
    print(csv_save_path)
    os.makedirs(csv_save_path, exist_ok=True)

    static_cate_ae, temporal_cate_ae, ehr_safe_ae, gan, checkpoint_cfg = load_model(eval_cfg, model_name, checkpoint)

    logit_threshold = 0.5
    seq_len = checkpoint_cfg.dataloader.seq_len
    static_transformer = dataset.static_transformer
    temporal_transformer = dataset.temporal_transformer

    seq_len = checkpoint_cfg.dataloader.seq_len
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=checkpoint_cfg.dataloader.batch_size,
                                             shuffle=False)

    sc_cols = dataset.sc_cols
    tc_cols = dataset.tc_cols
    sn_cols = dataset.sn_cols
    tn_cols = dataset.tn_cols
    sl_cols = dataset.sl_cols
    tl_cols = dataset.tl_cols

    static_categorical_card, static_categorical_feature_info = get_static_categorical_card(dataset,
                                                                                           static_transformer)
    sc_binary_cols = [col.column_name for col in static_categorical_feature_info if col.column_type == 'Binary']

    temporal_categorical_card, temporal_categorical_feature_info = get_temporal_categorical_card(dataset,
                                                                                                 temporal_transformer)
    tc_binary_cols = [col.column_name for col in temporal_categorical_feature_info if col.column_type == 'Binary']

    diagnoses_prefix = checkpoint_cfg.preprocess.icd_code.diagnoses_prefix
    procedure_prefix = checkpoint_cfg.preprocess.icd_code.procedure_prefix
    proc_prefix = checkpoint_cfg.preprocess.proc_prefix

    (sc_feature_info,
     sn_feature_info,
     sm_feature_info,
     tc_feature_info,
     tn_feature_info,
     tm_feature_info) = get_feature_info(static_transformer, temporal_transformer)

    target = {'static_data': [], 'temporal_data': [], 'time_data': []}
    data_hat = {'sc_rep_hat': [], 'tc_rep_hat': [], 'sn_hat': [], 'tn_hat': [],
                'static_mask_hat': [], 'temporal_mask_hat': [], 'time_hat': []}

    with torch.no_grad():
        for batch in dataloader:
            device = torch.device(f'cuda:{checkpoint_cfg.device_num}' if torch.cuda.is_available() else 'cpu')
            sc, sn, sl, sm = unpack_static_batch(batch, device=device)
            tc, tn, tl, tm, _ = unpack_temporal_batch(batch, device=device)

            # concat list-wise categorical
            sc_in = torch.cat([sc, sl], dim=-1)
            tc_in = torch.cat([tc, tl], dim=-1)

            batch_size, seq_len, _ = tc_in.size()

            # flatten temporal numerical & masks across time
            tn_flat = tn.reshape(batch_size, -1)
            tm_flat = tm.reshape(batch_size, -1)

            # encoding
            sc_rep, _ = static_cate_ae(sc_in)
            tc_rep, _ = temporal_cate_ae(tc_in)

            x = torch.cat([sc_rep, sn, tc_rep, tn_flat, sm, tm_flat], dim=-1)
            rep, _ = ehr_safe_ae(x)

            out = {
                "rep": rep,
                "batch_size": batch_size,
                "seq_len": seq_len,
                "sc": sc,
                "sn": sn,
                "sl": sl,
                "sm": sm,
                "tc": tc,
                "tn": tn,
                "tl": tl,
                "tm": tm,
                "sc_rep": sc_rep,
                "tc_rep": tc_rep,
            }

            rep = out['rep']
            batch_size = out['batch_size']
            seq_len = out['seq_len']

            # ----- target data (real) -----
            static_x = torch.cat([out['sc'], out['sl'], out['sn'], out['sm']], dim=-1)
            target['static_data'].append(static_x)

            temporal_x = torch.cat([out['tc'], out['tl'], out['tn'], out['tm']], dim=-1)
            target['temporal_data'].append(temporal_x)

            gan_out = gan(rep)
            fake = gan_out['fake']
            dec_fake = ehr_safe_ae.decoder(fake)

            sc_rep_dim = checkpoint_cfg.model.static_categorical_autoencoder.decoder.embedding_dim
            tc_rep_dim = checkpoint_cfg.model.temporal_categorical_autoencoder.decoder.embedding_dim
            sn_dim = sum([info.output_dimensions for info in sn_feature_info])
            tn_dim = sum([info.output_dimensions for info in tn_feature_info])
            sm_dim = sum([info.output_dimensions for info in sm_feature_info])
            tm_dim = sum([info.output_dimensions for info in tm_feature_info])
            (
                sc_rep_hat,
                tc_rep_hat,
                sn_hat,
                tn_hat,
                static_mask_hat,
                temporal_mask_hat
            ) = split_data_by_type(dec_fake, sc_rep_dim, tc_rep_dim, sn_dim, tn_dim, sm_dim, tm_dim, seq_len)

            data_hat['sc_rep_hat'].append(sc_rep_hat)
            data_hat['tc_rep_hat'].append(tc_rep_hat)
            data_hat['sn_hat'].append(sn_hat)
            data_hat['static_mask_hat'].append(static_mask_hat)
            data_hat['tn_hat'].append(tn_hat.view(batch_size, seq_len, -1))
            data_hat['temporal_mask_hat'].append(
                temporal_mask_hat.view(batch_size, seq_len, -1)
            )

            # concat over all batches
        target['static_data'] = torch.cat(target['static_data'], dim=0)
        target['temporal_data'] = torch.cat(target['temporal_data'], dim=0)

        data_hat['sc_rep_hat'] = torch.cat(data_hat['sc_rep_hat'], dim=0)
        data_hat['tc_rep_hat'] = torch.cat(data_hat['tc_rep_hat'], dim=0)
        data_hat['sn_hat'] = torch.cat(data_hat['sn_hat'], dim=0)
        data_hat['tn_hat'] = torch.cat(data_hat['tn_hat'], dim=0)
        data_hat['static_mask_hat'] = torch.cat(data_hat['static_mask_hat'], dim=0)
        data_hat['temporal_mask_hat'] = torch.cat(
            data_hat['temporal_mask_hat'], dim=0
        )

        # ----- static reconstruction -----
        sc_hat = static_cate_ae.decoder(data_hat['sc_rep_hat'])
        act_sc_hat, act_sn_hat, act_sm_hat = apply_static_activation(
            sc_cols=sc_cols,
            sc_binary_cols=sc_binary_cols,
            sl_cols=sl_cols,
            sc_hat=sc_hat,
            sn_hat=data_hat['sn_hat'],
            sm_hat=data_hat['static_mask_hat'],
            batch_size=data_hat['sn_hat'].shape[0],
            logit_threshold=logit_threshold)
        static_data_hat = torch.cat([act_sc_hat, act_sn_hat, act_sm_hat], dim=-1)

        # ----- temporal reconstruction -----
        tc_hat = temporal_cate_ae.decoder(data_hat['tc_rep_hat'])
        act_tc_hat, act_tn_hat, act_tm_hat = apply_temporal_activation(
            tc_cols=tc_cols,
            tc_binary_cols=tc_binary_cols,
            tl_cols=tl_cols,
            tc_hat=tc_hat,
            tn_hat=data_hat['tn_hat'],
            tm_hat=data_hat['temporal_mask_hat'],
            batch_size=data_hat['tn_hat'].shape[0],
            seq_len=seq_len,
            logit_threshold=logit_threshold)
        temporal_data_hat = torch.cat(
            [act_tc_hat, act_tn_hat, act_tm_hat], dim=-1
        )

        # ----- inverse transform -----
        static_feature_info = sc_feature_info + sn_feature_info + sm_feature_info
        static_data = target['static_data'].detach().cpu().numpy()
        static_data_hat = static_data_hat.detach().cpu().numpy()
        static_data, static_data_hat = inverse_transform(
            real=static_data,
            synthetic=static_data_hat,
            transformer=static_transformer,
            feature_info=static_feature_info,
        )

        mask = target['temporal_data'] != checkpoint_cfg.dataloader.pad_value
        mask = mask.reshape(-1, target['temporal_data'].shape[-1])
        mask = mask[:, 0].reshape(-1, 1).repeat(
            1, target['temporal_data'].shape[-1]
        )
        mask = mask.detach().cpu().numpy()

        temporal_feature_info = tc_feature_info + tn_feature_info + tm_feature_info
        temporal_data = (
            target['temporal_data']
            .reshape(-1, target['temporal_data'].shape[-1])
            .detach()
            .cpu()
            .numpy()
        )
        temporal_data_hat = (
            temporal_data_hat.reshape(-1, temporal_data_hat.shape[-1])
            .detach()
            .cpu()
            .numpy()
        )
        temporal_data, temporal_data_hat = inverse_transform(
            real=temporal_data,
            synthetic=temporal_data_hat,
            transformer=temporal_transformer,
            feature_info=temporal_feature_info,
            mask=mask,
        )

    static_data.to_csv(os.path.join(csv_save_path, 'static_data.csv'), index=False)
    temporal_data.to_csv(os.path.join(csv_save_path, 'temporal_data.csv'), index=False)

    static_data_hat.to_csv(os.path.join(csv_save_path, 'static_reconstructed.csv'), index=False)
    temporal_data_hat.to_csv(os.path.join(csv_save_path, 'temporal_reconstructed.csv'), index=False)

    n = 20
    n_samples = len(dataset) * n
    static_data_hat, temporal_data_hat = generate_samples(dataset,
                                                          checkpoint_cfg,
                                                          static_cate_ae,
                                                          temporal_cate_ae,
                                                          ehr_safe_ae,
                                                          gan,
                                                          n_samples,
                                                          logit_threshold)

    static_data_hat.to_csv(os.path.join(csv_save_path, f'static_reconstructed_{n_samples}.csv'), index=False)
    temporal_data_hat.to_csv(os.path.join(csv_save_path, f'temporal_reconstructed_{n_samples}.csv'), index=False)