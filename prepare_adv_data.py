import os
import warnings
import torch
import torch.multiprocessing as mp
from speechbrain.pretrained import SpeakerRecognition
from attack import basic, imperceptible, universal, adaptive
import yaml
import json
import random
import utils.util as util
import numpy as np
from detect import detector
from tqdm import tqdm
# Global data and code
warnings.filterwarnings('ignore')
with open("detection.json") as f:
  detection_config = json.load(f)
  dets = detection_config['dets']
  det_name_map = detection_config['det_name_map']
  aes = detection_config['aes']
  ae_name_map = detection_config['ae_name_map']
  res_dir = detection_config['res_dir']

success = {}
scores = {}
succ_scores = {}

def gen_adv_data(alg, is_univ, success, scores, succ_scores):
  if torch.cuda.is_available():
    ver_ecapa = SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={'device': 'cuda'})
    device = torch.device('cuda')
  else:
    ver_ecapa = SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={'device': 'cpu'})
    device = torch.device('cpu')

  with open("config/attack.yaml") as f:
    config = yaml.safe_load(f)
  with open("config/defense.yaml") as f:
    d_config = yaml.safe_load(f)

  detectors = {}
  for det in dets:
    exec(f"detectors[det] = \
                detector.{det_name_map[det]}(ver_ecapa, d_config)")

  ae = {}
  exec(f"ae['{alg}'] = {ae_name_map[alg]}(ver_ecapa, config)")
  ae = ae[alg]

  print(f"Sub process: {ae.name}")
  with open("attack_pairs.json") as f:
    adv_pairs = json.load(f)
  # Pre-load data
  target_embs = {}
  for target in adv_pairs:
    p_target = f"enroll/{target}.pt"
    target_embs[target] = torch.load(p_target, map_location=device)
  # For attack source audios
  ori_audios = {}
  for target in adv_pairs:
    audios = util.load_audio(adv_pairs[target])
    ori_audios[target] = {}
    for idx, sample in enumerate(adv_pairs[target]):
      ori_audios[target][sample] = audios[idx].to(device)

  success_lst = success[alg]

  for target in tqdm(adv_pairs):
    for sample in adv_pairs[target]:
      p_adv = sample
      target_emb = target_embs[target]
      adv_ori = ori_audios[target][p_adv]
      if is_univ:
        dir_adv = '/'.join(p_adv.split('/')[:-1])
        with os.popen(f"ls {dir_adv}") as f:
          samples = f.read().split('\n')
          samples.remove('')
          if p_adv in samples:
            samples.remove(p_adv)
          p_samples = random.sample(
              samples, config['univ_num_samples'])
          p_samples = [f"{dir_adv}/{p}" for p in p_samples]
        adv_wavs = util.load_audio(p_samples)
        adv_wavs, adv_lens = util.pad_patch(adv_wavs, device)
        ae_succ, _, ae_adv, _ = ae.attack(
            adv_ori, adv_wavs, adv_lens, target_emb)
      else:
        ae_succ, _, ae_adv, _ = ae.attack(adv_ori, target_emb)
      ae_adv = ae_adv.to(device)

      success_lst.append(ae_succ.item())

      for det in dets:
        det_scores = detectors[det].extract_vec(ae_adv, target_emb)
        scores[det][alg].append(det_scores)
        if ae_succ.item() is True:
          succ_scores[det][alg].append(det_scores)

  print(f"Sub process: {ae.name} Done")


if __name__ == "__main__":
  for det in dets:
    scores[det] = {}
    succ_scores[det] = {}
  for alg in aes:
    success[alg] = []
    for det in dets:
      scores[det][alg] = []
      succ_scores[det][alg] = []

  # Multiple Process
  if 'univ' in aes:
    gen_adv_data(alg, True, success, scores, succ_scores)

  for alg in set(aes)-{'univ'}:
    gen_adv_data(alg, False, success, scores, succ_scores)

  for alg in aes:
    success[alg] = np.array(success[alg])
  print("Success Rate:")
  for alg in aes:
    success[alg] = np.array(success[alg])
    succrate = np.count_nonzero(success[alg]) / success[alg].size * 100
    print(f"{alg.upper()}: {succrate:.5f}%")

  if not os.path.exists(res_dir):
      os.makedirs(res_dir)
  with open(f"{res_dir}/scores_adv_data.npy", "wb") as f:
    for det in dets:
      for alg in aes:
        np.save(f, np.array(scores[det][alg]))
        np.save(f, np.array(succ_scores[det][alg]))
