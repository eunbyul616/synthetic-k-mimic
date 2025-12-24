import torch


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

    return static_categorical_card, categorical_feature_info


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

    return temporal_categorical_card, categorical_feature_info


def get_static_info(dataset, static_transformer):
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


def get_temporal_info(dataset, temporal_transformer):
    tc_cols = dataset.tc_cols
    tn_cols = dataset.tn_cols
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


def unpack_static_batch(batch, device='cpu'):
    """
    Unpack static data batch.
    Args:
        batch: tuple, (sc_data, tc_data, sn_data, tn_data, sl
        device: str, device to move the data to.
    Returns:
        sc_data: Tensor, static categorical data.
        sn_data: Tensor, static numerical data.
        sl_data: Tensor, static label data.
        static_mask: Tensor, static data mask.
    """
    sc_data = batch[0].to(device)
    sn_data = batch[2].to(device)
    sl_data = batch[4].to(device)
    static_mask = batch[6].to(device)
    return sc_data, sn_data, sl_data, static_mask


def unpack_temporal_batch(batch, device='cpu'):
    """
    Unpack temporal data batch.
    Args:
        batch: tuple, (tc_data, tn_data, tl_data, temporal_mask)
        device: str, device to move the data to.
    Returns:
        tc_data: Tensor, temporal categorical data.
        tn_data: Tensor, temporal numerical data.           
        tl_data: Tensor, temporal label data.
        temporal_mask: Tensor, temporal data mask.
        condition: None
    """
    tc_data = batch[1].to(device)
    tn_data = batch[3].to(device)
    tl_data = batch[5].to(device)
    temporal_mask = batch[7].to(device)
    condition = None
    return tc_data, tn_data, tl_data, temporal_mask, condition


def get_feature_info(static_transformer, temporal_transformer):
    sc_feature_info = [
        info
        for info in static_transformer._data_manipulation_info_list
        if (info.column_type in ['Categorical', 'Binary', 'Listwise'])
           and ('_mask' not in info.column_name)
    ]
    sn_feature_info = [
        info
        for info in static_transformer._data_manipulation_info_list
        if info.column_type == 'Numerical'
    ]

    sm_feature_info = []
    mask_info = [
        info
        for info in static_transformer._data_manipulation_info_list
        if '_mask' in info.column_name
    ]
    for s_info in (sc_feature_info + sn_feature_info):
        for info in mask_info:
            if f'{s_info.column_name}_mask' == info.column_name:
                sm_feature_info.append(info)
                break

    tc_feature_info = [
        info
        for info in temporal_transformer._data_manipulation_info_list
        if (info.column_type in ['Categorical', 'Binary', 'Listwise'])
           and ('_mask' not in info.column_name)
    ]
    tn_feature_info = [
        info
        for info in temporal_transformer._data_manipulation_info_list
        if info.column_type == 'Numerical'
    ]

    tm_feature_info = []
    mask_info = [
        info
        for info in temporal_transformer._data_manipulation_info_list
        if '_mask' in info.column_name
    ]
    for tc_info in (tc_feature_info + tn_feature_info):
        for info in mask_info:
            if f'{tc_info.column_name}_mask' == info.column_name:
                tm_feature_info.append(info)
                break

    return sc_feature_info, sn_feature_info, sm_feature_info, tc_feature_info, tn_feature_info, tm_feature_info


def apply_static_activation(
        sc_cols, sc_binary_cols, sl_cols, 
        sc_hat, sn_hat, sm_hat,
        batch_size, logit_threshold=0.5):
    s_idx = 0
    act_sc_hat = []
    for i, col in enumerate(sc_cols + sl_cols):
        _x_hat = sc_hat[i]
        dim = _x_hat.shape[-1]

        if col in sc_cols:
            if col in sc_binary_cols:
                _act_x_hat = torch.sigmoid(_x_hat)
                _act_x_hat = (_act_x_hat >= logit_threshold).float()
                act_sc_hat.append(_act_x_hat)

            else:
                act_sc_hat.append(torch.softmax(_x_hat, dim=-1))

        elif col in sl_cols:
            _act_x_hat = torch.sigmoid(_x_hat)
            _act_x_hat = (_act_x_hat >= logit_threshold).float()
            act_sc_hat.append(_act_x_hat)

        s_idx += dim

    act_sc_hat = torch.cat(act_sc_hat, dim=-1)
    # act_sn_hat = torch.sigmoid(sn_hat)
    act_sn_hat = sn_hat
    act_sm_hat = torch.sigmoid(sm_hat)
    act_sm_hat = (act_sm_hat >= logit_threshold).float()

    return act_sc_hat, act_sn_hat, act_sm_hat


def apply_temporal_activation(
        tc_cols, tc_binary_cols, tl_cols,
        tc_hat, tn_hat, tm_hat,
        batch_size, seq_len, logit_threshold=0.5):
    s_idx = 0
    act_x_hat = []
    for i, col in enumerate(tc_cols + tl_cols):
        _x_hat = tc_hat[i]
        _x_hat = _x_hat.view(batch_size, seq_len, -1)
        dim = _x_hat.shape[-1]

        if col in tc_cols:
            if col in tc_binary_cols:
                _act_x_hat = torch.sigmoid(_x_hat)
                _act_x_hat = (_act_x_hat >= logit_threshold).float()
                act_x_hat.append(_act_x_hat)

            else:
                act_x_hat.append(torch.softmax(_x_hat, dim=-1))

        elif col in tl_cols:
            _act_x_hat = torch.sigmoid(_x_hat)
            _act_x_hat = (_act_x_hat >= logit_threshold).float()
            act_x_hat.append(_act_x_hat)

        s_idx += dim
    act_tc_hat = torch.cat(act_x_hat, dim=-1)
    # act_tn_hat = torch.sigmoid(tn_hat)
    act_tn_hat = tn_hat
    act_tm_hat = torch.sigmoid(tm_hat)
    act_tm_hat = (act_tm_hat >= logit_threshold).float()

    return act_tc_hat, act_tn_hat, act_tm_hat


def split_data_by_type(x_hat, sc_rep_dim, tc_rep_dim, sn_dim, tn_dim, sm_dim, tm_dim, seq_len):
    s_idx = 0
    # static features
    rep_dim = sc_rep_dim
    dim = sc_rep_dim + sn_dim
    sc_rep_hat = x_hat[:, s_idx:s_idx + rep_dim]
    sn_hat = x_hat[:, s_idx + rep_dim:s_idx + dim]
    s_idx += dim

    # temporal features
    rep_dim = tc_rep_dim
    dim = tc_rep_dim + (seq_len * tn_dim)
    tc_rep_hat = x_hat[:, s_idx:s_idx + rep_dim]
    tn_hat = x_hat[:, s_idx + rep_dim:s_idx + dim]
    s_idx += dim

    # mask loss
    sm_dim = sm_dim
    dim = sm_dim + (seq_len * tm_dim)
    static_mask_hat = x_hat[:, s_idx:s_idx + sm_dim]
    temporal_mask_hat = x_hat[:, s_idx + sm_dim:s_idx + dim]
    s_idx += dim

    return sc_rep_hat, tc_rep_hat, sn_hat, tn_hat, static_mask_hat, temporal_mask_hat