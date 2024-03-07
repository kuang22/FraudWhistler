import torch
import torchaudio
from speechbrain.dataio.preprocess import AudioNormalizer


def load_audio(paths):
  """paths is list"""
  wavs = []
  normalizer = AudioNormalizer()
  for p in paths:
    sig, sr = torchaudio.load(p, channels_first=False)
    wav = normalizer(sig, sr)
    wavs.append(wav.cpu())
  """Return: wavs (list of Tensors)"""
  """Tensor -> [Batch, Time]"""
  return wavs


def pad_patch(wavs, device):
  lens = []
  for idx in range(len(wavs)):
    lens.append(len(wavs[idx]))
  for idx in range(len(wavs)):
    wavs[idx] = torch.nn.functional.pad(
        wavs[idx], (0, max(lens) - lens[idx]), 'constant', 0).tolist()
  lens = torch.Tensor(lens)
  lens = lens / max(lens)
  wavs = torch.Tensor(wavs)
  wavs = wavs.to(device)
  lens = lens.to(device)
  """wavs -> [Batch, Time]"""
  return wavs, lens
