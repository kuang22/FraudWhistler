import torch
from .attacker import Attacker


class FGSM(Attacker):
  def __init__(self, model, config):
    super().__init__(model, config)
    self.epsilon = config['fgsm_epsilon']
    self.name = f'FGSM_{self.epsilon:.5f}'

  def generate(self, x, enroll_emb):
    self.model.eval()
    self.model.zero_grad()
    _x = x.detach().clone()
    _x.requires_grad_()
    adv_emb = self.model.encode_batch(_x, normalize=True).squeeze(dim=1)
    loss = - self.scorer(adv_emb, enroll_emb)
    loss.backward()
    grad = _x.grad.detach()
    _x = _x - self.epsilon * grad.sign()
    _x = torch.clamp(_x.detach(), -1, 1)
    return _x


class PGD(Attacker):
  def __init__(self, model, config):
    super().__init__(model, config)
    self.epsilon = config['pgd_epsilon']
    self.num_iters = config['pgd_num_iters']
    self.step_size = config['pgd_step_size']
    self.name = f'PGD_{self.epsilon:.5f}'

  def generate(self, x, enroll_emb):
    self.model.eval()
    self.model.zero_grad()
    _x = x.detach().clone()
    for _ in range(self.num_iters):
      _x.requires_grad_()
      self.model.zero_grad()
      test_emb = self.model.encode_batch(
          _x, normalize=True).squeeze(dim=1)
      loss = - self.scorer(test_emb, enroll_emb)
      loss.backward()
      grad = _x.grad.detach()
      _x = _x - self.step_size * grad.sign()
      _x = x + torch.clamp(_x - x, -self.epsilon, self.epsilon)
      _x = torch.clamp(_x.detach(), -1, 1)
    return _x


class CW(Attacker):
  def __init__(self, model, config):
    super().__init__(model, config)
    self.num_iters = config['cw_num_iters']
    self.lr = config['cw_lr']
    self.const = config['cw_const']
    self.norm_power = config['cw_norm_power']
    self.name = ('CW_2' if self.norm_power ==
                 2 else 'CW_inf') + f'_{self.const:.5f}'

  def lp_norm(self, delta):
    if self.norm_power == 2:
      return torch.norm(delta, p=self.norm_power, dim=-1).mean()
    else:
      return torch.max(torch.abs(delta)).mean()

  def generate(self, x, enroll_emb):
    self.model.eval()
    self.model.zero_grad()
    w = torch.zeros_like(x, dtype=torch.float,
                         requires_grad=True, device=x.device)
    optimizer = torch.optim.Adam([w], lr=self.lr)
    for _ in range(self.num_iters):
      _x = 0.5 * (torch.tanh(w) + 1)
      test_emb = self.model.encode_batch(
          _x, normalize=True).squeeze(dim=1)
      loss1 = self.lp_norm(_x-x)
      loss2 = - self.scorer(test_emb, enroll_emb)
      loss = loss1 + self.const * loss2
      loss = loss
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
    return _x
