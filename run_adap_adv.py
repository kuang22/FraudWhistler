import numpy as np
import torch
import torchaudio
from speechbrain.pretrained import SpeakerRecognition
import yaml
import json
import utils.util as util
from detect import detector
from sklearn import svm
from attack import adaptive
import sys
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')
with open("detection.json") as f:
  detection_config = json.load(f)
  dets = detection_config['dets']
  det_name_map = detection_config['det_name_map']
  res_dir = detection_config['res_dir']

if __name__ == "__main__":
    fpr = int(sys.argv[1]) / 100
    sys.stdout = open(f"{res_dir}/det_adap.txt", "w")
    X_train = {}
    X_test = {}
    with open(f"{res_dir}/scores_data.npy", "rb") as f:
        for det in dets:
          X_train[det] = np.load(f)
          X_test[det] = np.load(f)
        test_asr_succ = np.load(f)
    # -------------FraudWhistler SVM
    X_train = X_train['fw']
    X_test = X_test['fw']
    clf = svm.OneClassSVM(nu=fpr, kernel='rbf')
    clf.fit(X_train)
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)
    # ---- Prepare detector and adaptive attacker
    if torch.cuda.is_available():
        ver_ecapa = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb", run_opts={'device': 'cuda'})
        device = torch.device('cuda')
    else:
        ver_ecapa = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb", run_opts={'device': 'cpu'})
        device = torch.device('cpu')
    with open("config/attack.yaml") as f:
        config = yaml.safe_load(f)
    with open("config/defense.yaml") as f:
        d_config = yaml.safe_load(f)

    epsilon_lst = [0.001, 0.002, 0.004, 0.006, 0.008, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1]
    for epsilon in epsilon_lst:
        print(f"------------------------epsilon_{epsilon}")
        config['adap_epsilon'] = epsilon
        config['adap_step_size'] = epsilon / config['adap_num_iters']
        fraudwhistler = detector.FraudWhistler(ver_ecapa, d_config)
        ae = adaptive.AdapFW(ver_ecapa, config, fraudwhistler)
        # ------ Evaluate
        with open("attack_pairs.json") as f:
            adv_pairs = json.load(f)
        succs = []
        X_scores = []
        snr = []
        pesq = []
        for target in tqdm(adv_pairs):
            for sample in adv_pairs[target]:
                target_emb = torch.load(f"enroll/{target}.pt")
                adv_ori = util.load_audio([sample])[0].to(device)
                ori_succ, _, adap_adv, auditory = ae.attack(adv_ori, target_emb)
                succs.append(ori_succ)
                snr.append(auditory['SNR'])
                pesq.append(auditory['PESQ'])
                adap_adv = adap_adv.to(device)
                adap_scores = fraudwhistler.extract_vec(adap_adv, target_emb)
                X_scores.append(adap_scores)
                # adap_adv = adap_adv.unsqueeze(0).cpu()
                # audio_name = f"{target}_{sample.split('/')[-1]}"
                # torchaudio.save(f"Results/adap/{audio_name}", adap_adv, 16000, format='wav', bits_per_sample=32)
        succs = np.array(succs)
        X_scores = np.array(X_scores)
        snr = np.array(snr)
        pesq = np.array(pesq)
        print(f"SNR: {np.mean(snr):.2f}         PESQ: {np.mean(pesq):.2f}")
        y_pred_adap = clf.predict(X_scores)
        n_succ_adap = succs[(succs == 1) * (y_pred_adap == 1)].size
        asr = n_succ_adap / succs.size * 100.0
        print(f"Adaptive Attack Success Rate: {asr:.2f}%")
        print(f"Robust accuracy of whole system on AEs: {(1 - n_succ_adap / y_pred_adap.size) * 100:.2f}%")
