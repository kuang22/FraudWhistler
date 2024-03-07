import torch
import numpy as np
from scipy import signal
from .attacker import Attacker

sr = 16000
win_len = 2048
hop_len = 512
n_fft = 2048
window = torch.hamming_window(win_len)


def calc_norm_psd(wave):
  c_spec = torch.stft(wave, n_fft, hop_len, win_len,
                      window=window.to(wave.device),
                      pad_mode='constant',
                      onesided=True, return_complex=True)
  magnitude = c_spec.abs().squeeze()
  z = magnitude / n_fft
  psd = 10 * torch.log10(z * z + 1e-20)
  psd_max = psd.max()
  psd = 96 - psd_max + psd
  return psd, psd_max


def ath_f(f):
  f_ = f * 1e-3
  return 3.64 * torch.pow(f_, -0.8) \
      - 6.5 * torch.exp(-0.6 * torch.pow(f_ - 3.3, 2)) \
      + 1e-3 * torch.pow(f_, 4)


def ath_k(k):
  return ath_f(k / n_fft * sr)


def bark_f(f):
  return 13 * torch.arctan(0.76 * f * 1e-3) \
      + 3.5 * torch.arctan(torch.pow(f / 7500., 2))


def bark_k(k):
  return bark_f(k / n_fft * sr)


bins = torch.arange(0, n_fft // 2 + 1)
BARK = bark_k(bins)
ATH = ath_k(bins)
ATH = torch.where(BARK > 1, ATH, torch.FloatTensor([float('-inf')]))


def two_slops(bark_psd_index):
  delta_m = - 6.025 - 0.275 * bark_psd_index[:, 0]
  Ts = []
  for tone in range(bark_psd_index.shape[0]):
    bark_masker = bark_psd_index[tone, 0]
    dz = BARK - bark_masker
    sf = torch.zeros_like(dz)
    index = torch.where(dz > 0)[0][0]
    sf[:index] = 27 * dz[:index]
    sf[index:] = (-27 + 0.37 *
                  max(bark_psd_index[tone, 1] - 40, 0)) * dz[index:]
    T = bark_psd_index[tone, 1] + delta_m[tone] + sf
    Ts.append(T)
  return torch.vstack(Ts)


def calc_thresh(psd):
  # return ATH
  # local maximum
  masker_index = signal.argrelextrema(psd.numpy(), np.greater)[0]
  masker_index = torch.from_numpy(masker_index)
  try:
    # remove boundaries
    if masker_index[0] == 0:
      masker_index = masker_index[1:]
    if masker_index[-1] == len(psd) - 1:
      masker_index = masker_index[:-1]
    # larger than ATH
    masker_index = masker_index[psd[masker_index] > ATH[masker_index]]
    # smooth
    psd_k = torch.pow(10, psd[masker_index] / 10.)
    psd_k_prev = torch.pow(10, psd[masker_index - 1] / 10.)
    psd_k_post = torch.pow(10, psd[masker_index + 1] / 10.)
    psd_m = 10 * torch.log10(psd_k_prev + psd_k + psd_k_post)
    # local maximum with [-0.5Bark, 0.5Bark]
    bark_m = BARK[masker_index]
    bark_psd_index = torch.vstack([bark_m, psd_m, masker_index]).T
    cur, next = 0, 1
    while next < bark_psd_index.shape[0]:
      if next >= bark_psd_index.shape[0]:
        break
      if bark_psd_index[cur, 2] == -1:
        break
      while bark_psd_index[next, 0] - bark_psd_index[cur, 0] < 0.5:
        if bark_psd_index[next, 1] > bark_psd_index[cur, 1]:
          bark_psd_index[cur, 2] = -1
          cur = next
          next = cur + 1
        else:
          bark_psd_index[next, 2] = -1
          next += 1
        if next >= bark_psd_index.shape[0]:
          break
      cur = next
      next = cur + 1
    bark_psd_index = bark_psd_index[bark_psd_index[:, 2] != -1]
    # individual threshold
    Ts = two_slops(bark_psd_index)
    # global threshold
    Gs = torch.pow(10, ATH / 10.) + \
        torch.sum(torch.pow(10, Ts / 10.), dim=0)
    return 10 * torch.log10(Gs)
  except Exception as err:
    print(err)
    return ATH


def generate_threshold(wave):
  psd, psd_max = calc_norm_psd(wave)
  H = []
  for i in range(psd.shape[1]):
    H.append(calc_thresh(psd[:, i]))
  H = torch.vstack(H).T
  return H, psd_max


class FM(Attacker):
  def __init__(self, model, config):
    super().__init__(model, config)
    self.num_iters = config['fm_num_iters']
    self.lr = config['fm_lr']
    self.const = config['fm_const']
    self.name = f'FM_{self.const:.5f}'

  def psycho_loss(self, perturb, H, psd_max):
    c_spec = torch.stft(perturb, n_fft, hop_len, win_len,
                        window=window.to(perturb.device),
                        pad_mode='constant',
                        onesided=True, return_complex=True)
    magnitude = c_spec.abs().squeeze()
    z = magnitude / n_fft
    psd_perturb = 10 * torch.log10(z * z + 1e-20)
    psd_perturb = 96 - psd_max + psd_perturb
    return torch.mean(torch.clamp(psd_perturb - H, 0))

  def generate(self, x, enroll_emb):
    self.model.eval()
    self.model.zero_grad()
    H, psd_max = generate_threshold(x.cpu())
    H, psd_max = H.to(x.device), psd_max.to(x.device)
    perturb = torch.zeros_like(
        x, dtype=torch.float, requires_grad=True, device=x.device)
    torch.nn.init.xavier_normal_(perturb.unsqueeze(0), 0.1)
    optimizer = torch.optim.Adam([perturb], lr=self.lr)
    for _ in range(self.num_iters):
      _x = x + perturb
      test_emb = self.model.encode_batch(
          _x, normalize=True).squeeze(dim=1)
      loss1 = self.psycho_loss(perturb, H, psd_max)
      loss2 = - self.scorer(test_emb, enroll_emb)
      loss = loss1 + self.const * loss2
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
    return _x
