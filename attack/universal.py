import torch
from .attacker import Attacker


class UNIV(Attacker):
  def __init__(self, model, config):
    super().__init__(model, config)
    self.epsilon = config['univ_epsilon']
    self.num_iters = config['univ_num_iters']
    self.step_size = config['univ_step_size']
    self.confidence = config['univ_confidence']
    self.length = config['univ_length']
    self.num_samples = config['univ_num_samples']
    self.name = f'UNIV_{self.epsilon:.5f}'

  def generate(self, wavs, lens, enroll_emb):
    self.model.eval()
    self.model.zero_grad()
    perturb = torch.zeros(self.length, dtype=torch.float,
                          requires_grad=True, device=wavs.device)
    wav_len = wavs.size()[1]
    repeat_time = int(wav_len / self.length) + 1
    num_samples = self.num_samples
    for _ in range(self.num_iters):
      perturb.requires_grad_()
      perturb_prime = perturb.repeat(repeat_time)[:wav_len]
      perturb_primes = perturb_prime.unsqueeze(0).repeat(num_samples, 1)
      _wavs = wavs.detach() + perturb_primes
      test_embs = self.model.encode_batch(
          _wavs, lens, normalize=True).squeeze(dim=1)
      losses = self.scorer(test_embs, enroll_emb.repeat(num_samples, 1))
      thetas = self.threshold * \
          torch.ones(num_samples, device=losses.device)
      confidences = torch.tensor(
          [self.confidence] * num_samples, device=losses.device)
      loss = torch.sum(torch.max(thetas - losses, - confidences))
      loss.backward()
      grad = perturb.grad.detach()
      perturb = perturb - self.step_size * grad.sign()
      perturb = torch.clamp(perturb.detach(),
                            -self.epsilon, self.epsilon)
    return perturb

  def attack(self, x, wavs, lens, enroll_emb):
    perturb = self.generate(wavs, lens, enroll_emb)
    wav_len = len(x)
    repeat_time = int(wav_len / self.length) + 1
    _x = x.detach() + perturb.detach().repeat(repeat_time)[:wav_len]
    adv_emb = self.model.encode_batch(
        _x, normalize=True).detach().squeeze(dim=1)
    score = self.scorer(adv_emb, enroll_emb)
    is_success = score >= self.threshold
    dic_auditory = self.auditory(_x, x)
    return is_success, score, _x, dic_auditory
