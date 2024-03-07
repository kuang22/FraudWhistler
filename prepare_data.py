import os
import json
import yaml
import random
import warnings
import argparse
import torch
import torch.nn as nn
import utils.util as util
import numpy as np
from detect import detector
from speechbrain.pretrained import SpeakerRecognition
from tqdm import tqdm
# Global data and code
warnings.filterwarnings('ignore')
with open("detection.json") as f:
  detection_config = json.load(f)
  dets = detection_config['dets']
  det_name_map = detection_config['det_name_map']
  res_dir = detection_config['res_dir']


if __name__ == "__main__":
  # Prepare training and test pairs (clean trials based on enrolled users)
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "-n", "--number",
      help="set the number of samples for each enrolled user",
      type=int,
      default=70)
  parser.add_argument(
      "-s", "--size",
      help="set the size(number) of test samples for each enrolled user",
      type=int,
      default=30)
  parser.add_argument(
      "-d", "--seed",
      help="set the value of seed for radom function",
      type=int,
      default=91225)
  args = parser.parse_args()
  random.seed(args.seed)
  train_pairs = {}
  test_pairs = {}
  with open('enroll_wavs.json') as f:
    enroll_spk_wavs = json.load(f)
  enroll_spk_ids = list(enroll_spk_wavs.keys())

  for spk_id in enroll_spk_ids:
    train_pairs[spk_id] = []
    test_pairs[spk_id] = []
    with os.popen(f"ls dataset/ori/{spk_id}") as f:
      wavs = f.read().split('\n')[:-1]
      wavs = [f'dataset/ori/{spk_id}/' + name for name in wavs]
      wavs = list(set(wavs) - set(enroll_spk_wavs[spk_id]))
      wavs_fnames = random.sample(wavs, args.number)
      train_pairs[spk_id] = wavs_fnames[:]
      wavs = list(set(wavs) - set(wavs_fnames))
      test_pairs[spk_id] = random.sample(wavs, args.size)
  # Prepare training and test data based on Destructor
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
  with open("config/defense.yaml") as f:
    config = yaml.safe_load(f)

  detectors = {}
  for det in dets:
    exec(f"detectors[det] = \
             detector.{det_name_map[det]}(ver_ecapa, config)")

  # Pre-load data
  enroll_embs = {}
  print("Loading enroll embeddings:")
  for enroll in tqdm(train_pairs):
    enroll_embs[enroll] = torch.load(
        f"enroll/{enroll}.pt", map_location=device)
  # For train pairs
  train_audios = {}
  print("Loading train audios:")
  for enroll in tqdm(train_pairs):
    audios = util.load_audio(train_pairs[enroll])
    train_audios[enroll] = {}
    for idx, test in enumerate(train_pairs[enroll]):
      train_audios[enroll][test] = audios[idx].to(device)
  # For test pairs
  test_audios = {}
  print("Loading test audios:")
  for enroll in tqdm(test_pairs):
    audios = util.load_audio(test_pairs[enroll])
    test_audios[enroll] = {}
    for idx, test in enumerate(test_pairs[enroll]):
      test_audios[enroll][test] = audios[idx].to(device)

  X_train = {}
  for det in dets:
    X_train[det] = []
  for enroll in tqdm(train_pairs):
    enroll_emb = enroll_embs[enroll]
    for test in train_pairs[enroll]:
      x = train_audios[enroll][test]
      for det in dets:
        det_scores = detectors[det].extract_vec(x, enroll_emb)
        X_train[det].append(det_scores)

  scorer = nn.CosineSimilarity()
  threshold = config['threshold']
  test_asr_succ = []
  X_test = {}
  for det in dets:
    X_test[det] = []
  for enroll in tqdm(test_pairs):
    enroll_emb = enroll_embs[enroll]
    for test in test_pairs[enroll]:
      x = test_audios[enroll][test]
      test_emb = ver_ecapa.encode_batch(
          x, normalize=True).detach().squeeze(dim=1)
      sim_score = scorer(test_emb, enroll_emb)
      succ = True if sim_score >= threshold else False
      test_asr_succ.append(succ)

      for det in dets:
        det_scores = detectors[det].extract_vec(x, enroll_emb)
        X_test[det].append(det_scores)

  if not os.path.exists(res_dir):
      os.makedirs(res_dir)
  with open(f"{res_dir}/scores_data.npy", "wb") as f:
    for det in dets:
      np.save(f, np.array(X_train[det]))
      np.save(f, np.array(X_test[det]))
    np.save(f, np.array(test_asr_succ))
