# MAAT: Mamba Adaptive Anomaly Transformer
Authors: Abdellah Zakaria Sellam*, Ilyes Benaissa, Abdelmalik Taleb-Ahmed, Luigi
Patrono, Cosimo Distante

## Abstract
Anomaly detection in time series poses a critical challenge in industrial moni-
toring, environmental sensing, and infrastructure reliability, where accurately
distinguishing anomalies from complex temporal patterns remains an open
problem. While existing methods, such as the Anomaly Transformer lever-
aging multi-layer association discrepancy between prior and series distribu-
tions and DCdetector employing dual-attention contrastive learning, have
advanced the field, critical limitations persist. These include sensitivity
to short-term context windows, computational inefficiency, and degraded
performance under noisy and non-stationary real-world conditions. To ad-
dress these challenges, we present MAAT (Mamba Adaptive Anomaly Trans-
former), an enhanced architecture that refines association discrepancy model-
ing and reconstruction quality for more robust anomaly detection. Our work
introduces two key contributions to the existing Anomaly transformer ar-
chitecture: Sparse Attention, which computes association discrepancy more
efficiently by selectively focusing on the most relevant time steps. This re-
duces computational redundancy while effectively capturing long-range de-
pendencies critical for discerning subtle anomalies. A Mamba-Selective State
Space Model (Mamba-SSM) is also integrated into the reconstruction mod-
ule. A skip connection bridges the original reconstruction and the Mamba-
SSM output, while a Gated Attention mechanism adaptively fuses features
from both pathways. This design balances fidelity and contextual enhancement dynamically, 
improving anomaly localization and overall detection performance. 
Extensive experiments on benchmark datasets demonstrate that
MAAT significantly outperforms prior methods, achieving superior anomaly
distinguishability and generalization across diverse time series applications.
By addressing the limitations of existing approaches, MAAT sets a new stan-
dard for unsupervised time series anomaly detection in real-world scenarios.

Keywords:  MAAT, Transformer, Association Discrepancy, Gated Attention, Mamba-SSM, Sparse Attention, Anomaly Detection, Unsupervised Learning

## Architecture
![alt text](img/full_maat.png)

## Results
![alt text](img/result.png)

## Get Start
1. Create new virtual environement with Python 3.10 .
2. Clone the repo.
3. Install the requirements using: ```pip install -r requirements.txt```.
4. Download data.
5. Train and evaluate. using the scripts in ./script folder:
```bash
bash ./scripts/SMD.sh
bash ./scripts/MSL.sh
bash ./scripts/SMAP.sh
bash ./scripts/PSM.sh
bash ./scripts/SWAT.sh
bash ./scripts/NIPS_TS_Swan.sh
bash ./scripts/NIPS_TS_Water.sh
```
## Citation
If you find this repo useful, please cite our paper.

```
@misc{sellam2025maatmambaadaptiveanomaly,
      title={MAAT: Mamba Adaptive Anomaly Transformer with association discrepancy for time series}, 
      author={Abdellah Zakaria Sellam and Ilyes Benaissa and Abdelmalik Taleb-Ahmed and Luigi Patrono and Cosimo Distante},
      year={2025},
      eprint={2502.07858},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2502.07858}, 
}
```
## Contact
If you have any question or want to use the code, please contact z.abdellah.sellam@gmail.com, ilyesbenaissa7429@gmail.com
