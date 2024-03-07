import random
import torch
import torchaudio
import torch.nn as nn
import torchaudio.functional as F
import torchaudio.transforms as T
import speechbrain.processing.speech_augmentation as Aug
import librosa
import numpy as np
import pandas as pd


class Destructor():
  def __init__(self, model, config) -> None:
    self.scorer = nn.CosineSimilarity()
    self.model = model
    # When constitute fraudwhistler -> no need to use these two
    self.threshold = config['threshold']

  def perturb(self, x):
    pass

  def destruct(self, x, enroll_emb):
    x_clean = x.detach().clone()
    x_perturb = self.perturb(x_clean.unsqueeze(0))
    clean_emb = self.model.encode_batch(
        x_clean, normalize=True).detach().squeeze(1)
    perturb_emb = self.model.encode_batch(
        x_perturb, normalize=True).detach().squeeze(1)
    score_clean = self.scorer(clean_emb, enroll_emb).item()
    score_perturb = self.scorer(perturb_emb, enroll_emb).item()
    scores = [score_clean, score_perturb]
    return scores

  def get_diff(self, x, enroll_emb):
    scores = self.destruct(x, enroll_emb)
    cross = np.sign((scores[0] - self.threshold) *
                    (self.threshold - scores[1])).item()
    diff = scores[0] - scores[1]
    return [diff, cross]

  def get_score_diff(self, x, enroll_emb):
    scores = self.destruct(x, enroll_emb)
    diff = scores[0] - scores[1]
    return diff

  def get_score_diff_plus(self, x, enroll_emb):
    scores = self.destruct(x, enroll_emb)
    cross = np.sign((scores[0] - self.threshold) *
                    (self.threshold - scores[1])).item()
    if cross == 1:
      diff = scores[0] - scores[1] + 1
    else:
      diff = scores[0] - scores[1]
    return diff


class Noisifier(Destructor):
  def __init__(self, model, config) -> None:
    super().__init__(model, config)
    self.name = "Noisifier"
    self.snr = config['snr']
    self.noisifier = Aug.AddNoise(
        normalize=True,
        snr_low=self.snr,
        snr_high=self.snr).to(model.device)
    self.fullname = self.name + '_' + str(self.snr)

  def perturb(self, x):
    return self.noisifier(x, torch.ones(1).to(x.device))


class Reverber(Destructor):
  def __init__(self, model, config) -> None:
    super().__init__(model, config)
    self.name = "Reverber"
    self.reverb_file = config['reverb']
    self.scale_factor = config['scale_factor']
    self.rirs = pd.read_csv(self.reverb_file)
    self.fullname = self.name + '_' + str(self.scale_factor)

  def perturb(self, x):
    idx = random.randrange(0, len(self.rirs))
    rir_raw, _ = torchaudio.load(self.rirs.iloc[idx, 2])
    rir = rir_raw[0, :].unsqueeze(0)
    rir = (rir / torch.norm(rir, p=2)).to(x.device)
    res = F.fftconvolve(x, rir, mode="same").to(x.device)
    return res


class SpeedChanger(Destructor):
  def __init__(self, model, config) -> None:
    super().__init__(model, config)
    self.name = "SpeedChanger"
    self.speedrange = config['speedrange']
    self.samplerate = config['samplerate']
    self.speedchanger = Aug.SpeedPerturb(
        orig_freq=self.samplerate, speeds=self.speedrange).to(model.device)
    self.fullname = self.name + '_' + str(self.speedrange)

  def perturb(self, x):
    return self.speedchanger(x)


class Codec(Destructor):
  def __init__(self, model, config) -> None:
    super().__init__(model, config)
    self.name = "Codec"
    self.params = config['param']
    self.samplerate = config['samplerate']
    self.fullname = self.name + '_' + self.params['format']

  def perturb(self, x):
    _x = x.detach().clone()
    _x = _x.cpu().squeeze(0).unsqueeze(1)
    _x = F.apply_codec(_x, self.samplerate, **self.params).to(x.device)
    return _x.squeeze(1).unsqueeze(0)


class Quant(Destructor):
  def __init__(self, model, config) -> None:
    super().__init__(model, config)
    self.name = "Quant"
    self.qtbits = config['qtbits']
    self.fullname = self.name + '_' + str(self.qtbits)

  def perturb(self, x):
    return (x * 2**(self.qtbits - 1)).int().float() / 2**(self.qtbits - 1)


class Resample(Destructor):
  def __init__(self, model, config) -> None:
    super().__init__(model, config)
    self.name = "Resample"
    self.samplerate = config['samplerate']
    self.resample_rate = config['resample_rate']
    self.down_sampler = T.Resample(
        self.samplerate, self.resample_rate).to(model.device)
    self.up_sampler = T.Resample(
        self.resample_rate, self.samplerate).to(model.device)
    self.fullname = self.name + '_' + str(self.resample_rate)

  def perturb(self, x):
    return self.up_sampler(self.down_sampler(x))


class Clip(Destructor):
  def __init__(self, model, config) -> None:
    super().__init__(model, config)
    self.name = "Clip"
    self.clip_low = config['clip_low']
    self.clip_high = config['clip_high']
    self.clipper = Aug.DoClip(clip_low=self.clip_low,
                              clip_high=self.clip_high).to(model.device)
    self.fullname = self.name + '_' + \
        str(self.clip_low) + '_' + str(self.clip_high)

  def perturb(self, x):
    return self.clipper(x)


class DropChunk(Destructor):
  def __init__(self, model, config) -> None:
    super().__init__(model, config)
    self.name = "DropChunk"
    self.drop_count_low = config['chunk_drop_count_low']
    self.drop_count_high = config['chunk_drop_count_high']
    self.noise_factor = config['noise_factor']
    self.dropper = Aug.DropChunk(
        drop_count_low=self.drop_count_low,
        drop_count_high=self.drop_count_high,
        noise_factor=self.noise_factor).to(model.device)
    self.fullname = self.name + '_' + \
        str(self.drop_count_low) + '_' + str(self.drop_count_high)

  def perturb(self, x):
    return self.dropper(x, torch.ones(1).to(x.device))


class DropFreq(Destructor):
  def __init__(self, model, config) -> None:
    super().__init__(model, config)
    self.name = "DropFreq"
    self.drop_count_low = config['freq_drop_count_low']
    self.drop_count_high = config['freq_drop_count_high']
    self.dropper = Aug.DropFreq(
        drop_count_low=self.drop_count_low,
        drop_count_high=self.drop_count_high).to(model.device)
    self.fullname = self.name + '_' + \
        str(self.drop_count_low) + '_' + str(self.drop_count_high)

  def perturb(self, x):
    return self.dropper(x)


class Filtering(Destructor):
  def __init__(self, model, config) -> None:
    super().__init__(model, config)
    self.name = "Filtering"
    self.samplerate = config['samplerate']
    self.lowpass_factor = config['lowpass_factor']
    self.highpass_factor = config['highpass_factor']
    self.transform = T.SpectralCentroid(self.samplerate).to(model.device)
    self.fullname = self.name + '_' + \
        str(self.lowpass_factor) + '_' + str(self.highpass_factor)

  def perturb(self, x):
    centroids = self.transform(x)
    cent = centroids.median()
    _x = F.lowpass_biquad(x, self.samplerate, self.lowpass_factor * cent)
    _x = F.lowpass_biquad(_x, self.samplerate, self.lowpass_factor * cent)
    return _x


class MelReconstruct(Destructor):
  def __init__(self, model, config) -> None:
    super().__init__(model, config)
    self.samplerate = config['samplerate']
    self.name = "MelReconstruct"
    self.n_mels = config['n_mels']
    self.melspec_transform = T.MelSpectrogram(
        self.samplerate, n_mels=self.n_mels).to(model.device)
    self.invmel_transform = T.InverseMelScale(
        201, n_mels=self.n_mels).to(model.device)
    self.fullname = self.name + '_' + str(self.n_mels)

  def perturb(self, x):
    melspec = self.melspec_transform(x)
    spec = self.invmel_transform(melspec)
    griffin = T.GriffinLim(length=x.shape[-1]).to(x.device)
    return griffin(spec)


class Smoothing(Destructor):
  def __init__(self, model, config) -> None:
    super().__init__(model, config)
    self.name = "Smoothing"
    self.kernel_size = config['kernel_size']
    self.fullname = self.name + '_' + str(self.kernel_size)

  def perturb(self, x):
    win_len = self.kernel_size
    pad_len = (win_len - 1) // 2
    _x = torch.nn.functional.pad(x, (pad_len, pad_len))
    _x[..., :pad_len] = torch.cat(
        pad_len * [_x[..., pad_len].unsqueeze(-1)], dim=-1)
    _x[..., -pad_len:] = torch.cat(
        pad_len * [_x[..., -pad_len-1].unsqueeze(-1)], dim=-1)
    roll = _x.unfold(-1, win_len, 1)
    _x, _ = torch.median(roll, -1)
    return _x


class LPCReconstruct(Destructor):
  def __init__(self, model, config) -> None:
    super().__init__(model, config)
    self.name = "LPCReconstruct"
    self.order = config['order']
    self.fullname = self.name + '_' + str(self.order)

  def perturb(self, x):
    _x = x.detach().clone()
    _x = _x.cpu().squeeze(0).numpy()
    a = librosa.lpc(_x, order=self.order-1)
    e = np.random.normal(0, 0.5, _x.shape[0])
    for idx in range(self.order, _x.shape[0]):
      _x[idx] = np.sum([a[k-1]*e[idx-k]
                       for k in range(1, self.order+1)]) + e[idx]
    _x = torch.from_numpy(np.clip(_x, -1, 1))
    _x = _x.unsqueeze(0)
    _x = _x.to(x.device)
    return _x


class PitchShift(Destructor):
  def __init__(self, model, config) -> None:
    super().__init__(model, config)
    self.name = "PitchShift"
    self.samplerate = config['samplerate']
    self.steps = config['steps']
    self.fullname = self.name + '_' + str(self.steps)

  def perturb(self, x):
    _x = F.pitch_shift(x, self.samplerate, self.steps)
    return _x


class TimeShift(Destructor):
  def __init__(self, model, config) -> None:
    super().__init__(model, config)
    self.name = "TimeShift"
    self.ratio = config['ratio']
    self.fullname = self.name + '_' + str(self.ratio)

  def perturb(self, x):
    _x = x.clone()
    inter = int(_x.size(dim=1) * self.ratio)
    _x[0, -inter:] = x[0, :inter]
    _x[0, :-inter] = x[0, inter:]
    return _x


class TimeDepend(Destructor):
  def __init__(self, model, config) -> None:
    super().__init__(model, config)
    self.name = "TimeDepend"
    self.problist = [0.5, 0.8]

  def perturb(self, x):
    cut_len = int(x.size(dim=1) * self.prob)
    return x[0, :cut_len]

  def extract_vec(self, x, enroll_emb):
    vec = np.array([])
    for prob in self.problist:
      self.prob = prob
      vec = np.append(vec, self.get_score_diff_plus(x, enroll_emb))
    return vec


class WaveGuard():
  def __init__(self, model, config) -> None:
    self.model = model
    self.name = "WaveGuard"
    self.modules = []
    self.modules.append(MelReconstruct(self.model, config))
    self.modules.append(LPCReconstruct(self.model, config))

  def extract_vec(self, x, enroll_emb):
    vec = np.array([])
    for module in self.modules:
      vec = np.append(vec, module.get_score_diff_plus(x, enroll_emb))
    return vec


class FraudWhistler():
  def __init__(self, model, config) -> None:
    self.model = model
    self.name = "FraudWhistler"
    self.modules = []
    self.modules.append(Quant(self.model, config | {'qtbits': 7}))
    self.modules.append(Quant(self.model, config | {'qtbits': 8}))
    self.modules.append(Noisifier(self.model, config | {'snr': 1}))
    self.modules.append(Noisifier(self.model, config | {'snr': 10}))
    self.modules.append(Reverber(self.model, config))
    self.modules.append(Codec(self.model, config))
    self.modules.append(DropChunk(self.model, config))
    self.modules.append(DropFreq(self.model, config))

  def extract_vec(self, x, enroll_emb):
    vec = np.array([])
    for module in self.modules:
      vec = np.append(vec, module.get_score_diff_plus(x, enroll_emb))
    index = [0, 2]
    new_vec = np.delete(vec, index)
    d_var = np.var(new_vec)
    d_max = np.max(new_vec)
    d_min = np.min(new_vec)
    d_mean = np.mean(new_vec)
    d_range = d_max - d_min
    new_vec = np.append(new_vec, [d_var, d_range, d_mean, d_max])
    vec = np.append(new_vec, [vec[0]-vec[1], vec[2] - vec[3]])
    return vec
