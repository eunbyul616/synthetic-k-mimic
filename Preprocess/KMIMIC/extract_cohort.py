import os
from Preprocess.KMIMIC.utils.utils import *
from Preprocess.KMIMIC.utils.file import *
import config_manager


@measure_runtime
def extract_cohort(cfg):
    """
    Extract cohort from the original data
    Args:
        cfg: configuration

    Returns:

    """
    dataset = cfg.preprocess.dataset
    if cfg.preprocess.debug:
        debug_num = cfg.preprocess.debug_num
    else:
        debug_num = None

    data_path = os.path.join(cfg.path.raw_data_path, dataset)
    preprocessed_data_path = os.path.join(cfg.path.preprocessed_data_path, dataset)
    key_cols = cfg.preprocess.keys

    # load data
    patients = read_patients_table(cfg=cfg, path=data_path, file_name='patients.parquet', nrows=debug_num)
    admissions = read_admissions_table(cfg=cfg, path=data_path, file_name='admissions.parquet', nrows=debug_num)
    stays = read_icustays_table(cfg, path=data_path, file_name='icustays.parquet', nrows=debug_num)

    admissions['admission_type'] = admissions['admission_type'].map(lambda x: 'Unknown' if x == 'nan' else x)

    # set cohort
    cohort = set_cohort(patients, admissions, stays, only_icu_stays=cfg.preprocess.cohort.only_icu_stays)
    if cfg.preprocess.verbose:
        print('START:\n\tsubject_id: {}\n\thadm_id: {}\n\tstay_id: {}'.format(
            cohort.subject_id.unique().shape[0],
            cohort.hadm_id.unique().shape[0],
            cohort.stay_id.unique().shape[0])
        )

    # remove ICU transfers
    if cfg.preprocess.cohort.remove_icu_transfers:
        cohort = remove_icustays_with_transfer(cohort)
        if cfg.preprocess.verbose:
            print('REMOVE ICU TRANSFERS:\n\tsubject_id: {}\n\thadm_id: {}\n\tstay_id: {}'.format(
                cohort.subject_id.unique().shape[0],
                cohort.hadm_id.unique().shape[0],
                cohort.stay_id.unique().shape[0])
            )

    # remove multiple stays per admission
    if cfg.preprocess.cohort.remove_multiple_stays_per_admission:
        cohort = remove_multiple_stays_per_admission(cohort)
        if cfg.preprocess.verbose:
            print('REMOVE MULTIPLE STAYS PER ADMIT:\n\tsubject_id: {}\n\thadm_id: {}\n\tstay_id: {}'.format(
                cohort.subject_id.unique().shape[0],
                cohort.hadm_id.unique().shape[0],
                cohort.stay_id.unique().shape[0])
            )

    # remove patients age < 18 or age > 90
    # If a patient’s anchor_age is over 89 in the anchor_year then their anchor_age is set to 91.
    if cfg.preprocess.cohort.remove_patients_on_age:
        cohort = remove_patients_on_age(cohort, min_age=15, max_age=100)
        if cfg.preprocess.verbose:
            print('REMOVE PATIENTS AGE < 15 :\n\tsubject_id: {}\n\thadm_id: {}\n\tstay_id: {}'.format(
                cohort.subject_id.unique().shape[0],
                cohort.hadm_id.unique().shape[0],
                cohort.stay_id.unique().shape[0])
            )

    # remove stays exceeding the maximum length of stay
    if cfg.preprocess.cohort.remove_stays_on_los:
        cohort = remove_stays_on_los(cohort, min_los=cfg.preprocess.los.min, max_los=cfg.preprocess.los.max)
        if cfg.preprocess.verbose:
            print('REMOVE STAYS EXCEEDING THE MAXIMUM LENGTH OF STAY:\n\tsubject_id: {}\n\thadm_id: {}\n\tstay_id: {}'.format(
                cohort.subject_id.unique().shape[0],
                cohort.hadm_id.unique().shape[0],
                cohort.stay_id.unique().shape[0])
            )

    cohort = cohort.drop_duplicates(subset=key_cols, keep='first')

    # save cohort file
    save_csv(cohort, preprocessed_data_path, 'cohort.csv')

    if cfg.preprocess.verbose:
        # check duplicated value
        check_duplicated_value(cohort)

        print('COMPLETE SETTING COHORT AND SAVE FILE:\n\tsubject_id: {}\n\thadm_id: {}\n\tstay_id: {}'.format(
            cohort.subject_id.unique().shape[0],
            cohort.hadm_id.unique().shape[0],
            cohort.stay_id.unique().shape[0])
        )
        print('LOS MIN: ', cohort['los'].min())
        print('LOS MEAN: ', cohort['los'].mean())
        print('LOS MAX: ', cohort['los'].max())


if __name__ == "__main__":
    config_manager.load_config()
    cfg = config_manager.config

    # extract cohort
    extract_cohort(cfg=cfg)
