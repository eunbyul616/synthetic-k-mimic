import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from Trainer.train import Trainer
from Trainer.utils import *
from DataLoaders.loaderbase import get_dataloaders
from Utils.namespace import set_cfg
from Utils.train import save_ckpt, load_ckpt

from Trainer.tabular_ehr_gen.train_static_vae import StaticVAETrainer
from Trainer.tabular_ehr_gen.train_temporal_vae import TemporalVAETrainer

from Models.TabularEhrGen.ConditionalGAN import build_model

from Evaluation.DistributionSimilarity.correlation_comparison import *
from Visualization.timeseries import *


# Single discriminator
class GANTrainer(Trainer):
    def __init__(self, config, rank, **kwargs):
        super(GANTrainer, self).__init__(config=config, rank=rank, **kwargs)

        self.static_ae = kwargs['static_ae']
        self.temporal_ae = kwargs['temporal_ae']

        self.disc_loss_fn = kwargs['loss_fn']
        self.gen_loss_fn = kwargs['gen_loss_fn']
        self.disc_optimizer = kwargs['optimizer']
        self.gen_optimizer = kwargs['gen_optimizer']
        self.disc_scheduler = kwargs['scheduler']
        self.gen_scheduler = kwargs['gen_scheduler']

        self.start_epoch = load_ckpt(
            self.cfg,
            self.model,
            self.rank,
            gen_optimizer=self.gen_optimizer,
            disc_optimizer=self.disc_optimizer,
            gen_scheduler=self.gen_scheduler,
            disc_scheduler=self.disc_scheduler)

        self.dataloaders = get_dataloaders(cfg=self.cfg,
                                           train_dataset=kwargs['train_dataset'],
                                           valid_dataset=kwargs['validation_dataset'],
                                           test_dataset=kwargs['test_dataset'],
                                           rank=self.rank,
                                           world_size=self.world_size,
                                           collate_fn=kwargs['collate_fn'])
        self.test_loss = {key: [] for key in self.loss_keys}
        self.logit_threshold = 0.5

        self.sc_cols = self.dataloaders.train.dataset.sc_cols
        self.tc_cols = self.dataloaders.train.dataset.tc_cols
        self.sn_cols = self.dataloaders.train.dataset.sn_cols
        self.tn_cols = self.dataloaders.train.dataset.tn_cols
        self.sl_cols = self.dataloaders.train.dataset.sl_cols
        self.tl_cols = self.dataloaders.train.dataset.tl_cols

        self.sc_mask_dim = self.dataloaders.train.dataset.sc_mask_data.shape[-1]
        self.sl_mask_dim = self.dataloaders.train.dataset.sl_mask_data.shape[-1]
        self.sn_mask_dim = self.dataloaders.train.dataset.sn_mask_data.shape[-1]

        self.tc_mask_dim = self.dataloaders.train.dataset.tc_mask_data.shape[-1]
        self.tl_mask_dim = self.dataloaders.train.dataset.tl_mask_data.shape[-1]
        self.tn_mask_dim = self.dataloaders.train.dataset.tn_mask_data.shape[-1]

        self.static_transformer = self.dataloaders.train.dataset.static_transformer
        self.temporal_transformer = self.dataloaders.train.dataset.temporal_transformer

        # static_categorical_card
        categorical_feature_info = []
        for c in (self.sc_cols + self.sl_cols):
            info = [info for info in self.static_transformer._data_manipulation_info_list if info.column_name == c][0]
            categorical_feature_info.append(info)
        self.static_categorical_card = [info.output_dimensions for info in categorical_feature_info]

        # temporal_categorical_card
        categorical_feature_info = []
        for c in (self.tc_cols + self.tl_cols):
            info = [info for info in self.temporal_transformer._data_manipulation_info_list if info.column_name == c][0]
            categorical_feature_info.append(info)
        self.temporal_categorical_card = [info.output_dimensions for info in categorical_feature_info]

        # split categorical features and binary features
        sc_cols = []
        sb_cols = []
        for c in self.sc_cols:
            info = [info for info in self.static_transformer._data_manipulation_info_list if info.column_name == c][0]
            dim = info.output_dimensions
            if dim == 1:
                sb_cols.append(c)
            else:
                sc_cols.append(c)
        self.sb_cols = sb_cols
        self.sc_cols = sc_cols

        categorical_feature_info = []
        for c in (self.sn_cols + self.sc_cols + self.sb_cols + self.sl_cols):
            info = [info for info in self.static_transformer._data_manipulation_info_list if info.column_name == c][0]
            categorical_feature_info.append(info)
        for c in (self.dataloaders.train.dataset.sc_cols + self.sl_cols + self.sn_cols):
            info = [info for info in self.static_transformer._data_manipulation_info_list if
                    info.column_name == f'{c}_mask'][0]
            categorical_feature_info.append(info)
        self.static_feature_info = categorical_feature_info

        # split categorical features and binary features
        tc_cols = []
        tb_cols = []
        for c in self.tc_cols:
            info = [info for info in self.temporal_transformer._data_manipulation_info_list if info.column_name == c][0]
            dim = info.output_dimensions
            if dim == 1:
                tb_cols.append(c)
            else:
                tc_cols.append(c)
        self.tb_cols = tb_cols
        self.tc_cols = tc_cols

        categorical_feature_info = []
        for c in (self.tn_cols + self.tc_cols + self.tb_cols + self.tl_cols):
            info = [info for info in self.temporal_transformer._data_manipulation_info_list if info.column_name == c][0]
            categorical_feature_info.append(info)
        for c in (self.dataloaders.train.dataset.tc_cols + self.tl_cols + self.tn_cols):
            info = [info for info in self.temporal_transformer._data_manipulation_info_list if info.column_name == f'{c}_mask'][0]
            categorical_feature_info.append(info)
        self.temporal_feature_info = categorical_feature_info

        self.diagnoses_prefix = self.cfg.preprocess.icd_code.diagnoses_prefix
        self.procedure_prefix = self.cfg.preprocess.icd_code.procedure_prefix
        self.proc_prefix = self.cfg.preprocess.proc_prefix

        self.use_gumbel = True

    def run_epochs(self):
        # freeze the weights
        for param in self.static_ae.parameters():
            param.requires_grad = False
        for param in self.temporal_ae.parameters():
            param.requires_grad = False

        self.static_ae.eval()
        self.temporal_ae.eval()

        for epoch in range(self.start_epoch, self.total_epochs+1):
            self.model.train()
            self.train_one_epoch(epoch)
            self.model.eval()
            self.validate_one_epoch(epoch)

            if epoch % self.cfg.train.general.eval_freq == 0:
                self.eval_one_epoch(epoch)

            if self.scheduler is not None:
                stop_flag, self.scheduler_flag = self.train_watcher.check(epoch,
                                                                          self.validation_loss[self.target_key][-1])
                if stop_flag:
                    break

                # if self.scheduler_flag:
                #     self.scheduler.step()
                if (epoch + 1) % 100 == 0:
                    self.scheduler.step()
                    self.gen_scheduler.step()

            save_ckpt(
                cfg=self.cfg,
                epoch=epoch,
                validation_loss=self.validation_loss['Gen_Loss'],
                states={'epoch': epoch,
                        'arch': self.cfg.log.time,
                        'model': self.model,
                        'optimizer': self.disc_optimizer.state_dict(),
                        'scheduler': self.disc_scheduler.state_dict() if self.disc_scheduler is not None else None,
                        'gen_optimizer': self.gen_optimizer.state_dict(),
                        'gen_scheduler': self.gen_scheduler.state_dict() if self.gen_scheduler is not None else None
                        },
                rank=self.rank,
                save_condition='last'
            )
            self.wandb.log(self.train_loss, self.validation_loss, epoch)

        self.wandb.cleanup()

    def _set_iterator_postfix(self, iterator, loss, disc_loss, gen_loss):
        iterator.set_postfix(loss=loss / (iterator.n + 1),
                             disc_loss=disc_loss / (iterator.n + 1),
                             gen_loss=gen_loss / (iterator.n + 1),
                             disc_lr=self.disc_optimizer.param_groups[0]['lr'],
                             gen_lr=self.gen_optimizer.param_groups[0]['lr'])

    def _get_batch(self, batch):
        sc, tc, sn, tn, sl, tl, sm, tm, condition = batch
        sc = sc.to(self.device)
        tc = tc.to(self.device)
        sn = sn.to(self.device)
        tn = tn.to(self.device)
        sl = sl.to(self.device)
        tl = tl.to(self.device)
        sm = sm.to(self.device)
        tm = tm.to(self.device)
        condition = condition.to(self.device)

        return sc, tc, sn, tn, sl, tl, sm, tm, condition

    def moment_match(self, real, fake):
        real = real.view(real.size(0), -1)
        fake = fake.view(fake.size(0), -1)

        mu_r, mu_f = real.mean(0), fake.mean(0)
        c_r = (real - mu_r).T @ (real - mu_r) / (real.size(0) - 1)
        c_f = (fake - mu_f).T @ (fake - mu_f) / (fake.size(0) - 1)
        return (mu_r - mu_f).pow(2).mean() + (c_r - c_f).pow(2).mean()

    def train_one_epoch(self, epoch):
        total_losses = {key: 0.0 for key in self.loss_keys}

        train_iterator = self._set_iterator(self.dataloaders.train, epoch, mode='Train')

        for batch in train_iterator:
            sc, tc, sn, tn, sl, tl, sm, tm, condition = self._get_batch(batch)
            (sc_data_target, sc_data_converted,
             sb_data_target, sb_data_converted) = StaticVAETrainer.onehot_to_index_with_mask(sc, self.static_categorical_card)
            (s_num_hat, s_cat_logits, s_bin_logits, s_listwise_logits, s_mask_logits,
             s_mu, s_logvar, s_z, s_condition_hat) = self.static_ae(x_num=sn,
                                                                    x_cat=sc_data_converted,
                                                                    x_bin=sb_data_converted,
                                                                    x_listwise=sl,
                                                                    x_mask=sm,
                                                                    use_gumbel=self.use_gumbel)

            (tc_data_target, tc_data_converted,
             tb_data_target, tb_data_converted) = TemporalVAETrainer.temporal_onehot_to_index_with_mask(tc, self.temporal_categorical_card)
            (t_num_hat, t_cat_logits, t_bin_logits, t_listwise_logits, t_mask_logits,
             t_mu, t_logvar, t_z, t_condition_hat) = self.temporal_ae(x_num=tn,
                                                                      x_cat=tc_data_converted,
                                                                      x_bin=tb_data_converted,
                                                                      x_listwise=tl,
                                                                      x_mask=tm,
                                                                      use_gumbel=self.use_gumbel)

            z = torch.cat([s_z, t_z], dim=-1)
            batch_size, seq_len, _ = tc.size()
            self.batch_size = batch_size
            self.seq_len = seq_len

            self.model.discriminator.train()
            self.model.generator.eval()
            for _ in range(self.cfg.train.discriminator_steps):
                self.disc_optimizer.zero_grad()

                disc_loss = 0
                out = self.model(z, condition=condition)
                fake = out['fake']
                disc_fake = out['disc_fake']
                disc_real = out['disc_real']

                gp = self.model.calculate_gradient_penalty(
                    real=z,
                    fake=fake,
                    device=self.device,
                    lambda_gp=10,
                    condition=condition
                )

                disc_loss = self.disc_loss_fn(disc_fake, disc_real)
                total_disc_loss = disc_loss + gp
                total_disc_loss.backward(retain_graph=True)
                self.disc_optimizer.step()

            self.model.discriminator.eval()
            self.model.generator.train()
            self.gen_optimizer.zero_grad()
            gen_loss = 0
            out = self.model(z, condition=condition)
            disc_fake = out['disc_fake']
            disc_real = out['disc_real']
            gen_loss = self.gen_loss_fn(disc_fake, disc_real)

            idx = self.static_ae.decoder.fc1.in_features
            gen_loss.backward()
            self.gen_optimizer.step()

            loss = total_disc_loss + gen_loss
            total_losses['Total_Loss'] += loss.item()
            total_losses['Disc_Loss'] += total_disc_loss.item()
            total_losses['Gen_Loss'] += gen_loss.item()
            total_losses['GP'] += gp.item()

            self._set_iterator_postfix(train_iterator, total_losses['Total_Loss'], total_losses['Disc_Loss'], total_losses['Gen_Loss'])

        for key in self.loss_keys:
            self.train_loss[key].append(total_losses[key] / len(self.dataloaders.train))

    def validate_one_epoch(self, epoch):
        total_losses = {key: 0.0 for key in self.loss_keys}

        validation_iterator = self._set_iterator(self.dataloaders.valid, epoch, mode='val')

        with torch.no_grad():
            for batch in validation_iterator:
                sc, tc, sn, tn, sl, tl, sm, tm, condition = self._get_batch(batch)
                batch_size, seq_len, _ = tc.size()

                (sc_data_target, sc_data_converted,
                 sb_data_target, sb_data_converted) = StaticVAETrainer.onehot_to_index_with_mask(sc,
                                                                                                self.static_categorical_card)
                (s_num_hat, s_cat_logits, s_bin_logits, s_listwise_logits, s_mask_logits,
                 s_mu, s_logvar, s_z, s_condition_hat) = self.static_ae(x_num=sn,
                                                                        x_cat=sc_data_converted,
                                                                        x_bin=sb_data_converted,
                                                                        x_listwise=sl,
                                                                        x_mask=sm,
                                                                        use_gumbel=self.use_gumbel)

                (tc_data_target, tc_data_converted,
                 tb_data_target, tb_data_converted) = TemporalVAETrainer.temporal_onehot_to_index_with_mask(tc,
                                                                                                           self.temporal_categorical_card)
                (t_num_hat, t_cat_logits, t_bin_logits, t_listwise_logits, t_mask_logits,
                 t_mu, t_logvar, t_z, t_condition_hat) = self.temporal_ae(x_num=tn,
                                                                          x_cat=tc_data_converted,
                                                                          x_bin=tb_data_converted,
                                                                          x_listwise=tl,
                                                                          x_mask=tm,
                                                                          use_gumbel=self.use_gumbel)

                z = torch.cat([s_z, t_z], dim=-1)

                out = self.model(z, condition=condition)
                fake = out['fake']
                disc_fake = out['disc_fake']
                disc_real = out['disc_real']

                disc_loss = self.disc_loss_fn(disc_fake, disc_real)
                gen_loss = self.gen_loss_fn(disc_fake, disc_real)

                idx = self.static_ae.decoder.fc1.in_features
                loss = disc_loss + gen_loss

                total_losses['Total_Loss'] += loss.item()
                total_losses['Disc_Loss'] += disc_loss.item()
                total_losses['Gen_Loss'] += gen_loss.item()
                self._set_iterator_postfix(validation_iterator, total_losses['Total_Loss'], total_losses['Disc_Loss'], total_losses['Gen_Loss'])

            for key in self.loss_keys:
                self.validation_loss[key].append(total_losses[key] / len(self.dataloaders.valid))

    def eval_one_epoch(self, epoch):
        total_losses = {key: 0.0 for key in self.loss_keys}
        test_iterator = self._set_iterator(self.dataloaders.test, epoch, mode='test')

        if not os.path.exists(self.cfg.path.plot_file_path):
            os.makedirs(self.cfg.path.plot_file_path, exist_ok=True)

        data = {'static_data': [], 'temporal_data': []}
        data_hat = {'static_data': [], 'temporal_data': []}
        conditions = []
        s_latent_vectors = []
        t_latent_vectors = []
        with (torch.no_grad()):
            for batch in test_iterator:
                sc, tc, sn, tn, sl, tl, sm, tm, condition = self._get_batch(batch)
                seq_len = tc.size(1)

                (sc_data_target, sc_data_converted,
                 sb_data_target, sb_data_converted) = StaticVAETrainer.onehot_to_index_with_mask(sc,
                                                                                                self.static_categorical_card)
                (s_num_hat, s_cat_logits, s_bin_logits, s_listwise_logits, s_mask_logits,
                 s_mu, s_logvar, s_z, s_condition_hat) = self.static_ae(x_num=sn,
                                                                        x_cat=sc_data_converted,
                                                                        x_bin=sb_data_converted,
                                                                        x_listwise=sl,
                                                                        x_mask=sm,
                                                                        use_gumbel=self.use_gumbel)

                (tc_data_target, tc_data_converted,
                 tb_data_target, tb_data_converted) = TemporalVAETrainer.temporal_onehot_to_index_with_mask(tc,
                                                                                                           self.temporal_categorical_card)
                (t_num_hat, t_cat_logits, t_bin_logits, t_listwise_logits, t_mask_logits,
                 t_mu, t_logvar, t_z, t_condition_hat) = self.temporal_ae(x_num=tn,
                                                                          x_cat=tc_data_converted,
                                                                          x_bin=tb_data_converted,
                                                                          x_listwise=tl,
                                                                          x_mask=tm,
                                                                          use_gumbel=self.use_gumbel,
                                                                          z_s=s_z)

                z = torch.cat([s_z, t_z], dim=-1)

                out = self.model(z, condition=condition)
                fake = out['fake']
                disc_fake = out['disc_fake']
                disc_real = out['disc_real']

                disc_loss = self.disc_loss_fn(disc_fake, disc_real)
                gen_loss = self.gen_loss_fn(disc_fake, disc_real)

                idx = self.static_ae.decoder.fc1.in_features
                loss = disc_loss + gen_loss

                total_losses['Total_Loss'] += loss.item()
                total_losses['Disc_Loss'] += disc_loss.item()
                total_losses['Gen_Loss'] += gen_loss.item()
                self._set_iterator_postfix(test_iterator, total_losses['Total_Loss'], total_losses['Disc_Loss'], total_losses['Gen_Loss'])

                # real data
                static_x = self.get_static_target(sn, sc, sl, sm)
                data['static_data'].append(static_x)
                temporal_x = self.get_temporal_target(tn, tc, tl, tm)
                data['temporal_data'].append(temporal_x)

                # decoding
                idx = self.static_ae.decoder.fc1.in_features
                s_num_hat, s_cat_hat, s_bin_hat, s_listwise_hat, s_mask_hat = self.static_ae.decoder(fake[:, :idx], hard=True)
                t_num_hat, t_cat_hat, t_bin_hat, t_listwise_hat, t_mask_hat = self.temporal_ae.decoder(fake[:, idx:], seq_len=seq_len, hard=True, z_s=fake[:, :idx])

                s_act_x_hat = self.activate_static_hat(s_num_hat, s_cat_hat, s_bin_hat, s_listwise_hat, s_mask_hat)
                data_hat['static_data'].append(s_act_x_hat)
                t_act_x_hat = self.activate_temporal_hat(t_num_hat, t_cat_hat, t_bin_hat, t_listwise_hat, t_mask_hat)
                data_hat['temporal_data'].append(t_act_x_hat)

                conditions.append(condition)
                s_latent_vectors.append(fake[:, :idx])
                t_latent_vectors.append(fake[:, idx:])
                # t_latent_vectors.append(fake)

            data['static_data'] = torch.concatenate(data['static_data'], dim=0)
            data['temporal_data'] = torch.concatenate(data['temporal_data'], dim=0)
            data_hat['static_data'] = torch.concatenate(data_hat['static_data'], dim=0)
            data_hat['temporal_data'] = torch.concatenate(data_hat['temporal_data'], dim=0)
            conditions = torch.concatenate(conditions, dim=0)

            s_latent_vectors = torch.concatenate(s_latent_vectors, dim=0)
            t_latent_vectors = torch.concatenate(t_latent_vectors, dim=0)

            static_data = self.static_transformer.inverse_transform(data['static_data'].detach().cpu().numpy(), self.static_feature_info)
            static_data_hat = self.static_transformer.inverse_transform(data_hat['static_data'].detach().cpu().numpy(), self.static_feature_info)

            mask = data['temporal_data'] != self.cfg.dataloader.pad_value
            mask = mask.view(-1, data['temporal_data'].shape[-1]).detach().cpu().numpy()
            feature_dim = data['temporal_data'].shape[-1]
            temporal_data = self.temporal_transformer.inverse_transform(data['temporal_data'].view(-1, feature_dim).detach().cpu().numpy(), self.temporal_feature_info)
            temporal_data_hat = self.temporal_transformer.inverse_transform(data_hat['temporal_data'].view(-1, feature_dim).detach().cpu().numpy(), self.temporal_feature_info)

            s_latent_vectors = s_latent_vectors.detach().cpu().numpy()
            t_latent_vectors = t_latent_vectors.detach().cpu().numpy()

            conditions = conditions.detach().cpu().numpy()
            conditions_indices = conditions.argmax(axis=-1)
            conditions = pd.DataFrame(np.repeat(conditions, seq_len, axis=0),
                                      columns=['survived', 'icu_mortality', 'hospital_mortality'])

        icd_d_cols = [col for col in static_data.columns if (col.startswith(self.diagnoses_prefix)) and ('mask' not in col)]
        icd_p_cols = [col for col in static_data.columns if (col.startswith(self.procedure_prefix)) and ('mask' not in col)]
        proc_cols = [col for col in temporal_data.columns if (col.startswith(self.proc_prefix)) and ('mask' not in col)]

        # evaluate condition
        latent_vectors = np.concatenate([s_latent_vectors, t_latent_vectors], axis=-1)
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        tsne_results = tsne.fit_transform(latent_vectors)  # [n_samples, 2]
        condition_labels = {0: 'Survived',
                            1: 'ICU mortality',
                            2: 'Hospital mortality'}
        colors = {
            0: 'grey',
            1: 'red',
            2: 'green'
        }

        plt.figure(figsize=(8, 6))
        for cond_value in np.unique(conditions_indices):
            idx = conditions_indices == cond_value
            plt.scatter(
                tsne_results[idx, 0],
                tsne_results[idx, 1],
                s=15,
                alpha=0.6,
                label=condition_labels.get(cond_value, f'Class {cond_value}'),
                color=colors.get(cond_value, None)
            )

        plt.xlabel('tSNE 1')
        plt.ylabel('tSNE 2')
        plt.legend()
        plt.title('t-SNE of latent vectors by condition')
        plt.tight_layout()
        plt.savefig(os.path.join(self.cfg.path.plot_file_path, f'tsne_by_condition_{epoch}.png'))

        target_cols = 'icu_mortality'
        seq_len = self.cfg.dataloader.seq_len
        static_repeat = pd.DataFrame(np.repeat(static_data.values, seq_len, axis=0),
                                     columns=static_data.columns)
        total_data = pd.concat([static_repeat, temporal_data.reset_index(drop=True)], axis=1)
        static_repeat_hat = pd.DataFrame(np.repeat(static_data_hat[static_data.columns].values, seq_len, axis=0),
                                         columns=static_data.columns)
        total_data_hat = pd.concat([static_repeat_hat, temporal_data_hat.reset_index(drop=True)], axis=1)
        total_data = pd.concat([total_data, conditions], axis=1)
        total_data_hat = pd.concat([total_data_hat, conditions], axis=1)

        for col in (self.sc_cols+self.tc_cols):
            if col == target_cols:
                continue

            total_data['type'] = 'Real'
            total_data_hat['type'] = 'Synthetic'
            data = pd.concat([total_data, total_data_hat], axis=0)
            data = data.reset_index(drop=True)
            plt.figure(figsize=(8, 6))
            g = sns.catplot(data=data, x=col, y=target_cols, col='type',
                            kind='bar', height=6, aspect=0.5)
            for ax in g.axes.flat:
                ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
            plt.tight_layout()
            plt.savefig(
                os.path.join(self.cfg.path.plot_file_path, f'{col}_distribution_by_{target_cols}_{epoch}.png'))

        for col in (self.sn_cols+self.tn_cols):
            total_data['type'] = 'Real'
            total_data_hat['type'] = 'Synthetic'
            data = pd.concat([total_data, total_data_hat], axis=0)
            data = data.reset_index(drop=True)
            plt.figure(figsize=(8, 6))
            sns.boxplot(data=data, x='type', y=col, hue=target_cols)
            plt.legend(loc='upper right')
            plt.savefig(
                os.path.join(self.cfg.path.plot_file_path, f'{col}_distribution_by_{target_cols}_{epoch}.png'))

        for key in self.loss_keys:
            self.test_loss[key].append(total_losses[key] / len(self.dataloaders.test))

        for key in self.loss_keys:
            self.test_loss[key].append(total_losses[key] / len(self.dataloaders.test))

    def get_static_target(self, sn_data, sc_data, sl_data, static_mask):
        sc_target = []
        sb_target = []
        s_idx = 0
        for i, dim in enumerate(self.static_categorical_card[:-2]):
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

    def get_temporal_target(self, tn_data, tc_data, tl_data, temporal_mask):
        tc_target = []
        tb_target = []
        s_idx = 0
        for i, dim in enumerate(self.temporal_categorical_card[:-1]):
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

    def activate_static_hat(self, x_num_hat, x_cat_logits, x_bin_logits, x_listwise_logits, x_mask_logits):
        act_x_hat = []
        act_x_hat.append(x_num_hat)
        for sc_hat in x_cat_logits:
            _act_sc_hat = torch.softmax(sc_hat, dim=-1)
            _act_sc_hat = (_act_sc_hat >= self.logit_threshold).float()
            act_x_hat.append(_act_sc_hat)

        for sb_hat in x_bin_logits:
            if self.use_gumbel:
                _act_sb_hat = sb_hat
            else:
                _act_sb_hat = torch.sigmoid(sb_hat)
                _act_sb_hat = (_act_sb_hat >= self.logit_threshold).float()
            act_x_hat.append(_act_sb_hat)

        for sl_hat in x_listwise_logits:
            if self.use_gumbel:
                _act_sl_hat = sl_hat
            else:
                _act_sl_hat = torch.sigmoid(sl_hat)
                _act_sl_hat = (_act_sl_hat >= self.logit_threshold).float()
            act_x_hat.append(_act_sl_hat)

        for sm_hat in x_mask_logits:
            if self.use_gumbel:
                _act_sm_hat = sm_hat
            else:
                _act_sm_hat = torch.sigmoid(sm_hat)
                _act_sm_hat = (_act_sm_hat >= self.logit_threshold).float()
            act_x_hat.append(_act_sm_hat)
        act_x_hat = torch.cat(act_x_hat, dim=-1)

        return act_x_hat

    def activate_temporal_hat(self, x_num_hat, x_cat_logits, x_bin_logits, x_listwise_logits, x_mask_logits):
        act_x_hat = []
        act_x_hat.append(x_num_hat)
        for tc_hat in x_cat_logits:
            _act_tc_hat = torch.softmax(tc_hat, dim=-1)
            _act_tc_hat = (_act_tc_hat >= self.logit_threshold).float()
            act_x_hat.append(_act_tc_hat)

        for tb_hat in x_bin_logits:
            _act_tb_hat = torch.sigmoid(tb_hat)
            _act_tb_hat = (_act_tb_hat >= self.logit_threshold).float()
            act_x_hat.append(_act_tb_hat)

        for tl_hat in x_listwise_logits:
            if self.use_gumbel:
                _act_tl_hat = tl_hat
            else:
                _act_tl_hat = torch.sigmoid(tl_hat)
                _act_tl_hat = (_act_tl_hat >= self.logit_threshold).float()
            act_x_hat.append(_act_tl_hat)

        for tm_hat in x_mask_logits:
            if self.use_gumbel:
                _act_tm_hat = tm_hat
            else:
                _act_tm_hat = torch.sigmoid(tm_hat)
                _act_tm_hat = (_act_tm_hat >= self.logit_threshold).float()
            act_x_hat.append(_act_tm_hat)
        act_x_hat = torch.cat(act_x_hat, dim=-1)

        return act_x_hat


def load_embedders(cfg):
    from Utils.namespace import _load_yaml
    from Models.TabularEhrGen import StaticVAE
    from Models.TabularEhrGen import TemporalVAE

    checkpoint_saved_root = '/'.join(Path(cfg.path.ckpt_path).parts[:-1])
    static_embedder_checkpoint_path = os.path.join(checkpoint_saved_root,
                                                   cfg.train.static_ae.name,
                                                   cfg.train.static_ae.checkpoint)
    static_config = _load_yaml(os.path.join(static_embedder_checkpoint_path, 'config.yaml'))
    static_ae = StaticVAE.build_model(static_config.model.static_autoencoder,
                                 device=torch.device(f'cuda:{cfg.device_num}'))
    static_ae_checkpoint = torch.load(
        os.path.join(static_embedder_checkpoint_path, 'checkpoint.pth.tar'),
        map_location=f'cuda:{cfg.device_num}')
    static_ae.load_state_dict(static_ae_checkpoint['state_dict'])

    temporal_embedder_checkpoint_path = os.path.join(checkpoint_saved_root,
                                                     cfg.train.temporal_ae.name,
                                                     cfg.train.temporal_ae.checkpoint)
    temporal_config = _load_yaml(os.path.join(temporal_embedder_checkpoint_path, 'config.yaml'))
    temporal_ae = TemporalVAE.build_model(
        temporal_config.model.temporal_autoencoder,
        device=torch.device(f'cuda:{cfg.device_num}'))
    temporal_ae_checkpoint = torch.load(
        os.path.join(temporal_embedder_checkpoint_path, 'checkpoint.pth.tar'),
        map_location=f'cuda:{cfg.device_num}')
    temporal_ae.load_state_dict(temporal_ae_checkpoint['state_dict'])

    return static_ae, temporal_ae

def gan_trainer_main(cols):
    import config_manager

    from Datasets.dataset_k_mimic import KMIMICDataset as CustomDataset
    from Utils.reproducibility import lock_seed
    from Utils.train import set_loss_fn, set_scheduler, set_optimizer

    config_manager.load_config()
    cfg = config_manager.config
    cfg.log.wandb.name = f'{cfg.log.time}_{cfg.log.ipaddr}'
    cfg.log.wandb.tags = ['ConditionalGAN']

    lock_seed(seed=cfg.seed, multi_gpu=False, activate_cudnn=False)

    dataset_name = cfg.dataset.dataset_name
    fname = cfg.preprocess.preprocess_fname_suffix
    train_ratio, test_ratio = cfg.preprocess.train_valid_test_split_ratio
    dataset_fname = f'{dataset_name}_{fname}_{int(train_ratio * 10)}.h5'

    train_dataset = CustomDataset(cfg=cfg,
                                  dataset_name=dataset_name,
                                  dataset_fname=dataset_fname,
                                  mode='train',
                                  condition_col=cfg.data.condition_col,
                                  static_cols=cols)
    validation_dataset = CustomDataset(cfg=cfg,
                                       dataset_name=dataset_name,
                                       dataset_fname=dataset_fname,
                                       mode='val',
                                       condition_col=cfg.data.condition_col,
                                       static_cols=cols)
    test_dataset = CustomDataset(cfg=cfg,
                                 dataset_name=dataset_name,
                                 dataset_fname=dataset_fname,
                                 mode='test',
                                 condition_col=cfg.data.condition_col,
                                 static_cols=cols)

    # load embedders
    static_ae, temporal_ae = load_embedders(cfg)

    gen_output_dim = static_ae.decoder.fc1.in_features + temporal_ae.decoder.fc1.in_features
    # gen_output_dim = temporal_ae.decoder.fc1.in_features
    set_cfg(cfg, 'model.gan.generator.output_dim', gen_output_dim)

    # disc_input_dim = static_ae.decoder.fc1.in_features + temporal_ae.decoder.fc1.in_features + cfg.model.gan.condition_classes
    disc_input_dim = temporal_ae.decoder.fc1.in_features + cfg.model.gan.condition_classes
    set_cfg(cfg, 'model.gan.discriminator.input_dim', disc_input_dim)

    gen_input_dim = cfg.model.gan.generator.latent_dim + cfg.model.gan.condition_classes
    set_cfg(cfg, 'model.gan.generator.input_dim', gen_input_dim)

    model = build_model(cfg.model.gan, device=torch.device(f'cuda:{cfg.device_num}'))
    if cfg.train.general.init_weight:
        model.apply(model.init_weights)

    if cfg.train.general.distributed:
        world_size = torch.cuda.device_count()
    else:
        world_size = 1

    disc_loss_fn = set_loss_fn(cfg.train.loss)
    gen_loss_fn = set_loss_fn(cfg.train.gen_loss)
    disc_optimizer = set_optimizer(model.discriminator.parameters(),
                                   cfg.train.optimizer,
                                   apply_lookahead=cfg.train.optimizer.lookahead.flag)
    gen_optimizer = set_optimizer(model.generator.parameters(),
                                  cfg.train.gen_optimizer,
                                  apply_lookahead=cfg.train.gen_optimizer.lookahead.flag)
    disc_scheduler = set_scheduler(disc_optimizer, cfg.train.scheduler)
    gen_scheduler = set_scheduler(gen_optimizer, cfg.train.scheduler)

    trainer = GANTrainer(config=cfg,
                         rank=cfg.device_num,
                         world_size=world_size,
                         model=model,
                         collate_fn=None,
                         loss_fn=disc_loss_fn,
                         optimizer=disc_optimizer,
                         scheduler=disc_scheduler,
                         gen_loss_fn=gen_loss_fn,
                         gen_optimizer=gen_optimizer,
                         gen_scheduler=gen_scheduler,
                         train_dataset=train_dataset,
                         validation_dataset=validation_dataset,
                         test_dataset=test_dataset,
                         static_ae=static_ae,
                         temporal_ae=temporal_ae)
    trainer.run_epochs()


if __name__ == '__main__':
    cols = None
    gan_trainer_main(cols)