import math
import torch
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
import torch.nn as nn


class AdapAttacker():
  def __init__(self, model, config, destructor) -> None:
    self.model = model
    self.scorer = nn.CosineSimilarity()
    self.threshold = config['threshold']
    self.epsilon = config['pgd_epsilon']
    self.num_iters = config['pgd_num_iters']
    self.step_size = config['pgd_step_size']
    self.direct = config['adap_direct']
    self.bpda = config['adap_bpda']
    self.eot = config['adap_eot']
    self.eot_num_iters = config['eot_num_iters']
    self.destructor = destructor
    self.name = f'PGD_{self.epsilon:.5f}_{self.destructor.name}'

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

  def generate(self, x, enroll_emb):
    self.model.eval()
    self.model.zero_grad()
    _x = x.detach().clone()
    _x = _x.to(self.model.device)
    for _ in range(self.num_iters):
      if self.destructor.name in self.direct:
        self.model.zero_grad()
        _x.requires_grad_()
        mid = self.destructor.perturb(_x.unsqueeze(0))
        mid_emb = self.model.encode_batch(
            mid, normalize=True).squeeze(1)
        adv_emb = self.model.encode_batch(
            _x, normalize=True).squeeze(1)
        score = self.scorer(adv_emb, enroll_emb) + \
            self.scorer(mid_emb, enroll_emb)
        loss = - score
        loss.backward()
        grad = _x.grad.detach()
      elif self.destructor.name in self.bpda:
        self.model.zero_grad()
        mid = self.destructor.perturb(_x.unsqueeze(0)).squeeze(0)
        mid.requires_grad_()
        _x.requires_grad_()
        mid_emb = self.model.encode_batch(
            mid, normalize=True).squeeze(1)
        adv_emb = self.model.encode_batch(
            _x, normalize=True).squeeze(1)
        score = self.scorer(adv_emb, enroll_emb) + \
            self.scorer(mid_emb, enroll_emb)
        loss = - score
        loss.backward()
        grad = _x.grad.detach() + mid.grad.detach()
      elif self.destructor.name in self.eot:
        _x.requires_grad_()
        self.model.zero_grad()
        adv_emb = self.model.encode_batch(
            _x, normalize=True).squeeze(1)
        loss = - self.scorer(adv_emb, enroll_emb)
        loss.backward()
        grad1 = _x.grad.detach()
        grads = []
        for _ in range(self.eot_num_iters):
          self.model.zero_grad()
          mid = _x.detach().clone().requires_grad_()
          midp = self.destructor.perturb(mid.unsqueeze(0))
          mid_emb = self.model.encode_batch(
              midp, normalize=True).squeeze(1)
          loss = - self.scorer(mid_emb, enroll_emb)
          loss.backward()
          grads.append(mid.grad.detach().clone().tolist())
        grad2 = torch.tensor(grads).mean(dim=0).to(_x.device)
        grad = grad1 + grad2
      else:
        print("Error: Unknown destructor")
      _x = _x - self.step_size * grad.sign()
      _x = x + torch.clamp(_x - x, -self.epsilon, self.epsilon)
      _x = torch.clamp(_x.detach(), -1, 1)
    return _x

  def attack(self, x, enroll_emb):
    _x = self.generate(x, enroll_emb)
    _x_perturbed = self.destructor.perturb(_x.unsqueeze(0)).squeeze(0)
    adv_emb = self.model.encode_batch(_x, normalize=True).squeeze(dim=1)
    adv_emb_perturbed = self.model.encode_batch(
        _x_perturbed, normalize=True).squeeze(dim=1)
    score1 = self.scorer(adv_emb, enroll_emb)
    score2 = self.scorer(adv_emb_perturbed, enroll_emb)
    is_success = score1 >= self.threshold and score2 >= self.threshold
    dic_auditory = self.auditory(_x, x)
    return is_success, score1, score2,  _x, dic_auditory
