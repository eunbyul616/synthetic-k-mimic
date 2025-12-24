from Preprocess.KMIMIC.extract_cohort import extract_cohort
from Preprocess.KMIMIC.extract_features import *
from Preprocess.KMIMIC.extract_static_features import extract_static_features
from Preprocess.KMIMIC.extract_temporal_features import extract_temporal_features
import config_manager


if __name__ == "__main__":
    config_manager.load_config()
    cfg = config_manager.config

    # extract cohort
    extract_cohort(cfg=cfg)

    if cfg.preprocess.flag.diagnoses_icd:
        print("Extracting diagnoses ICD...")
        extract_diagnoses_icd(cfg=cfg)

    if cfg.preprocess.flag.procedures_icd:
        print("Extracting procedures ICD...")
        extract_procedures_icd(cfg=cfg)

    if cfg.preprocess.flag.transfers:
        print("Extracting transfers...")
        extract_transfers(cfg=cfg)

    if cfg.preprocess.flag.emar:
        print("Extracting emar...")
        extract_emar(cfg=cfg)

    if cfg.preprocess.flag.chartevents:
        print("Extracting chartevents...")
        chartevents = extract_chartevents(cfg=cfg)

    if cfg.preprocess.flag.outputevents:
        print("Extracting outputevents...")
        outputevents = extract_outputevents(cfg=cfg)

    if cfg.preprocess.flag.labevents:
        print("Extracting labevents...")
        labevents = extract_labevents(cfg=cfg)

    if cfg.preprocess.flag.procedureevents:
        print("Extracting procedureevents...")
        extract_procedureevents(cfg=cfg)

    if cfg.preprocess.flag.inputevents:
        print("Extracting inputevents...")
        extract_inputevents(cfg=cfg)

    save_fname = 'K_MIMIC_preprocessed.h5'
    extract_static_features(cfg, save_fname=save_fname)
    extract_temporal_features(cfg, save_fname=save_fname)

