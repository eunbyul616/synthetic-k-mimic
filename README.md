# Synthetic K-MIMIC: Korean ICU Registry Synthesized for Compliance with Korean Data Regulation

This repository provides code to **generate and evaluate synthetic ICU EHR time-series** derived from the K-MIMIC, with an emphasis on **privacy-aware release and downstream utility**.

---
Then we **jointly generate** synthetic patient trajectories by combining:
1) **Static Variational Autoencoder (Static VAE)**  
   Encodes static features into a latent representation and reconstructs static distributions.

2) **Temporal Variational Autoencoder (Temporal VAE)**  
   Encodes time-series features into a latent representation and reconstructs sequences.

3) **Conditional GAN**  
   Generates *mortality-conditioned* latent embeddings, which are decoded back to synthetic ICU records.

---

## Environment
- Python **3.10**
- CUDA **12.4**
- See `requirements.txt` for Python dependencies.

---

## Installation
Clone and install dependencies:

```bash
git clone <repository_url>
cd <repository_folder>
pip install -r requirements.txt
```

---

## Configuration (Hydra)
This project uses Hydra-based configuration.

### 1) Modify `rootpath` in `Conf/common/path.yaml`
Set the project root path used for saving outputs (checkpoints, logs, artifacts):

```yaml
rootpath: /path/to/your/project
```

### 2) Modify `hydra.run.dir` in `Conf/config.yaml`
Set Hydra working directory:

```yaml
hydra:
  run:
    dir: /path/to/your/project
```

---

## Usage

### 1. **Prepare Your Data**
   - Ensure your dataset is preprocessed and categorized into static and temporal features.
   
### 2. **Train Models**
   - You train the modules in the following order, because later stages load checkpoints from earlier stages:
   
   **1) Static Variational Autoencoder:**
   ```bash
   python -m Trainer.TabularEhrGen.train_static_vae train=tabular_ehr_gen/static_vae 
   ```
   
   **2) Temporal Variational Autoencoder:**
   ```bash
   python -m Trainer.TabularEhrGen.train_temporal_vae train=tabular_ehr_gen/temporal_vae 
   ```

   **3) Conditional GAN:**
   ```bash
   python -m Trainer.TabularEhrGen.train_conditional_gan train=tabular_ehr_gen/conditional_gan train.static_ae.checkpoint='/path/to/checkpoint/static_vae/' train.temporal_ae.checkpoint='/path/to/checkpoint/temporal_vae/'
   ```
   
### 3. **Generate Synthetic Data**
   Run the following command to generate synthetic data.
   
   ```bash
   python -m Inference.inference_tabular_ehr_gan evaluation.checkpoint='/path/to/checkpoint/gan/'
   ```

### 3. **Evaluation**

#### 1) Fidelity

   ```bash
   python -m Evaluation.Fidelity.evaluate_fidelity evaluation.checkpoint='/path/to/checkpoint/gan/'
   ```

#### 2) Utility

   ```bash
   python -m Evaluation.Utility.evaluate_utility evaluation.checkpoint='/path/to/checkpoint/gan/'
   ```

#### 3) Privacy
- Model-based metrics
    ```bash
    python -m Evaluation.Privacy.evaluete_privacy_model_based evaluation.checkpoint='/path/to/checkpoint/gan/'
    ```

- Model-free metrics
    ```bash
    python -m Evaluation.Privacy.evaluete_privacy_model_free evaluation.checkpoint='/path/to/checkpoint/gan/'
    ```

---

## Dataset
- **K-MIMIC**: available via the Korea Health Data Platform (KHDP) after ethical approval.
  - https://khdp.net/database/data-search-detail/664/K-MIMIC/1.0.0
- **Synthetic K-MIMIC (v2.0.0)**: publicly available via KHDP.
  - https://khdp.net/database/data-search-detail/723/SYN-ICU/2.0.0 
  
---
