# FraudWhistler
This is an official implementation of the USENIX Security 2024 paper **FraudWhistler: A Resilient, Robust and Plug-and-play Adversarial Example Detection Method for Speaker Recognition.** ([Preprint Version](https://www.usenix.org/conference/usenixsecurity24/presentation/wang-kun))

**Abstract:** With the in-depth integration of deep learning, state-of-the-art speaker recognition systems have achieved breakthrough progress. However, the intrinsic vulnerability of deep learning to Adversarial Example (AE) attacks has brought new severe threats to real-world speaker recognition systems. In this paper, we propose FraudWhistler, a practical AE detection system, which is resilient to various AE attacks, robust in complex physical environments, and plug-and-play for deployed systems. Its basic idea is to make use of an intrinsic characteristic of AE, i.e., the instability of model prediction for AE, which is totally different from benign samples. FraudWhistler generates several audio variants for the original audio sample with some distortion techniques, obtains multiple outputs of the speaker recognition system for these audio variants, and based on that FraudWhistler extracts some statistics representing the instability of the original audio sample and further trains a one-class SVM classifier to detect adversarial example. Extensive experimental results show that FraudWhistler achieves 98.7% accuracy on AE detection outperforming SOTA works by 13%, and 84% accuracy in the worst case against an adaptive adversary.

## Environment
1. Python 3.11
2. PyTorch 2.0.1
3. CUDA >= 11.7

You can run the following commands to create a new environments for running the codes with Anaconda:

```
conda env create -f env.yaml
conda active fraudwhistler
pip install -r requirements.txt
```

## Project Overview
### Key Component
The main function of FraudWhistler is implemented by following codes:
```
├── attack
│   ├── __init__.py
│   ├── adaptive.py
│   ├── attacker.py
│   ├── basic.py
│   ├── imperceptible.py
│   └── universal.py
├── detect
│   ├── __init__.py
│   └── detector.py
├── prepare_adv_data.py
├── prepare_data.py
├── run_adap_adv.py
└── run_det.py
```
**attack:** Implementation of five kinds of AE-generating algorithms.\
**detect:** Implementation of detection methods including FraudWhistler.\
**prepare\_data.py:** Implemented as to extract features on benign audio samples using FraudWhistler.\
**prepare\_adv\_data.py:** Implemented as to extract features on adversarial examples using FraudWhistler.\
**run\_det.py:** Train an one-class SVM on benign datasets and perform detection on AEs.\
**run\_adap\_adv.py:** Perform detection on adaptive AEs.

### Auxiliary Files
There are also some auxiliary files listed as follows:
```
├── attack_pairs.json
├── config
│   ├── attack.yaml
│   └── defense.yaml
├── detection.json
├── enroll
├── enroll.json
├── enroll_wavs.json
├── env.yaml
└── prepare_dataset.sh
```
**attack_pairs.json:** Defines the attack pairs.\
**config:** Attack and defense parameter setting in yaml files.\
**detection.json:** Experiment evaluation configuration.\
**enroll:** The speaker embedding of enrolled users.\
**enroll.json:** Store the spk\_id and corresponding embedding location.\
**enroll_wavs.json:** Store the specific wav files used in enrollment.\
**env.yaml:** Used to recreate experiment conda environment.\
**prepare_dataset.sh:** Used to download necessary dataset for evaluation.

## Dataset Preparing
Run the following command to download dataset:
```
sh prepare_dataset.sh
```

## Instructions to Run the Codes
We provide step-by-step instructions to generate the results in our paper.

**Notice:** Running the experiment may take several hours or more, depending on your hardware device.

Our deployed setup is as follows:
* 40 Intel Xeon Silver 4210R CPU
* 256GB RAM
* Four 48GB NVIDIA RTX A6000 GPU
* Ubuntu hirsute 21.04

### 0. General setup
Activate the conda environment:
```sh
conda activate fraudwhistler
```

### 1. Prepare Training Dataset
Run the following commands to prepare training dataset on benign samples:
```
python prepare_data.py
```

### 2. Evaluate on Static Attacks
Run the following commands to prepare eval dataset on adversarial examples:
```
python prepare_adv_data.py
```
Evaluate FraudWhistler on static attacks:
```
python run_det.py 3
```
The result output is stored in *Results/det.txt*, run the following command to keep watch on the real-time output:
```
tail -f Results/det.txt
```
and show the overall results after the program is finished:
```
cat Results/det.txt
```


### 3. Evaluate on Adaptive Attack
Run the following commands to evaluate FraudWhistler on adaptive attacks: 
```
python run_adap_adv.py 3
```
Similarly, the results are stored in *Results/det_adap.txt*, run the following command to keep watch on the real-time output:
```
tail -f Results/det_adap.txt
```
and show the final results after the program is finished:
```
cat Results/det_adap.txt
```
The details of the adaptive algorithm design could be found in the paper, and the raw anonymised results of our audibility study is in *listening\_res.csv*.

### 4. More Evaluation Experiments
To play with FraudWhistler, you could customize the attack and defense parameters. There are mainly two parts you could customize by yourself, i.e., *attack.yaml* and *defense.yaml*.

In *attack.yaml*, you could customize the strength of five implemented AE-generating algorithms and the proposed adaptive attack algorithm. 
```
fgsm_epsilon: 0.001

pgd_epsilon: 0.001
pgd_num_iters: 100
pgd_step_size: 0.00001

cw_num_iters: 1000
cw_lr: 0.001
cw_const: 0.01
cw_norm_power: 2

fm_num_iters: 1000
fm_lr: 0.001
fm_const: 0.01

univ_epsilon: 0.01
univ_num_iters: 100
univ_step_size: 0.0001
univ_confidence: 0.4
univ_length: 16000
univ_num_samples: 15

reverb: "reverb.csv"

eot_num_iters: 10
adap_epsilon: 0.1
adap_num_iters: 100
adap_step_size: 0.001
```

In *defense.yaml*, you could customize the parameters of distortion methods integrated in FraudWhistler.
```
# For add noise
snr: 10
# For reverbration
reverb: "reverb.csv"
scale_factor: 2.5
# For speedperturb
speedrange:
  - 90
  - 110
# For Codec
param:
  format: flac
  bits_per_sample: 8
  channels_first: False
# For Quantization
qtbits: 8
# For Resample
resample_rate: 6000
# For clip
clip_low: 0.1
clip_high: 0.1
# For DropChunk
chunk_drop_count_low: 50
chunk_drop_count_high: 150
noise_factor: 0.2
# For DropFreq
freq_drop_count_low: 10
freq_drop_count_high: 15
# For Filtering
lowpass_factor: 1.5
highpass_factor: 0.1
# For MelReconstruct
n_mels: 40
# For Smoothing
kernel_size: 7
# For LPC Reconstruct
order: 20
# For Pitch Shift
steps: 1
# For Time Shift
ratio: 0.5
# For time dependency
prob: 0.5
```

## Deploy FraudWhistler on New SR/Dataset

### 1. SR Enrollment
To empoly FraudWhistler on a new speaker recognition system or a new dataset, it needs to first enroll users for the SR system. Specifically, you need to generate the following files:
**enroll.json:** In this file, the pairs of (spkid, corresponding enroll audios) are stored.\
**enroll:** In this directory, speaker embeddings of each enrolled users are stored with the file name in the format of *spkid.pt*.\
**enroll_wavs.json:** This file stores the list of wav files used for each enrolled user to generate speaker embedding in the format of (spkid, [list of wav files]).

### 2. Attack Pairs
Second, to evaluate FraudWhistler, you need to generate an list of attack pairs like (non-enroll wav, target spkid). The file attack\_pairs.json is an example for your reference.

### 3. Training Dataset
With enrollment and attack pairs prepared, you need to call *prepare\_data.py* to generate training dataset on benign audio samples.

### 4. Evaluation Dataset
After generating training dataset, you could evaluate FraudWhistler on adversarial examples. Towards this end, you need to call existing adversari example-generating algorithms and FraudWhistler distortion module to build a evaluation dataset which is wrapped in *prepare_adv_data.py*.

### 5. Perform Detection
Finally, you could run the script to perform detection with following command:
```
python run_det.py 3
```
