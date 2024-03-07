import math
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
import torch.nn as nn


class Attacker(nn.Module):
  def __init__(self, model, config):
    super().__init__()
    self.model = model
    self.scorer = nn.CosineSimilarity()
    self.threshold = config['threshold']

  def generate(self, x, enroll_emb):
    pass

  def attack(self, x, enroll_emb):
    _x = self.generate(x, enroll_emb)
    adv_emb = self.model.encode_batch(
        _x, normalize=True).detach().squeeze(dim=1)
    score = self.scorer(adv_emb, enroll_emb)
    is_success = score >= self.threshold
    dic_auditory = self.auditory(_x, x)
    return is_success, score, _x, dic_auditory

  def auditory(self, _x, x):
    delta = _x.detach() - x.detach()
    wb_pesq = PerceptualEvaluationSpeechQuality(16000, 'wb')
    dic_auditory = {}
    dic_auditory['Infinity'] = int(delta.abs().max().item() * 2**16)
    dic_auditory['Loudness'] = 20 * \
        math.log(delta.abs().max().item() / _x.abs().max().item(), 10)
    dic_auditory['SNR'] = 10 * \
        math.log(_x.square().mean().item() /
                 delta.square().mean().item(), 10)
    dic_auditory['PESQ'] = wb_pesq(_x, x).item()
    return dic_auditory
