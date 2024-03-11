import math
import torch
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
import torch.nn as nn


class AdapFW():
    def __init__(self, model, config, detector) -> None:
        self.model = model
        self.scorer = nn.CosineSimilarity()
        self.threshold = config['threshold']
        self.epsilon = config['adap_epsilon']
        self.num_iters = config['adap_num_iters']
        self.step_size = config['adap_step_size']
        self.direct = config['adap_direct']
        self.bpda = config['adap_bpda']
        self.eot = config['adap_eot']
        self.eot_num_iters = config['eot_num_iters']
        self.detector = detector

    def auditory(self, _x, x):
        delta = _x.detach() - x.detach()
        wb_pesq = PerceptualEvaluationSpeechQuality(16000, 'wb')
        dic_auditory = {}
        dic_auditory['Infinity'] = int(delta.abs().max().item() * 2**16)
        dic_auditory['Loudness'] = 20 * \
            math.log(delta.abs().max().item() / _x.abs().max().item(), 10)
        dic_auditory['SNR'] = 10 * \
            math.log(_x.square().mean().item() / delta.square().mean().item(), 10)
        dic_auditory['PESQ'] = wb_pesq(_x, x).item()
        return dic_auditory

    def generate(self, x, enroll_emb):
        self.model.eval()
        self.model.zero_grad()
        _x = x.detach().clone()
        for _ in range(self.num_iters):
            all_grads = []
            for destructor in self.detector.modules:
                if destructor.name in self.direct:
                    self.model.zero_grad()
                    _x.requires_grad_()
                    mid = destructor.perturb(_x.unsqueeze(0))
                    mid_emb = self.model.encode_batch(mid, normalize=True).squeeze(1)
                    adv_emb = self.model.encode_batch(_x, normalize=True).squeeze(1)
                    score = self.scorer(adv_emb, enroll_emb) + self.scorer(mid_emb, enroll_emb)
                    loss = - score
                    loss.backward()
                    grad = _x.grad.detach()
                elif destructor.name in self.bpda:
                    self.model.zero_grad()
                    mid = destructor.perturb(_x.unsqueeze(0)).squeeze(0)
                    mid.requires_grad_()
                    _x.requires_grad_()
                    mid_emb = self.model.encode_batch(mid, normalize=True).squeeze(1)
                    adv_emb = self.model.encode_batch(_x, normalize=True).squeeze(1)
                    score = self.scorer(adv_emb, enroll_emb) + self.scorer(mid_emb, enroll_emb)
                    loss = - score
                    loss.backward()
                    grad = _x.grad.detach() + mid.grad.detach()
                elif destructor.name in self.eot:
                    _x.requires_grad_()
                    self.model.zero_grad()
                    adv_emb = self.model.encode_batch(_x, normalize=True).squeeze(1)
                    loss = - self.scorer(adv_emb, enroll_emb)
                    loss.backward()
                    grad1 = _x.grad.detach()
                    grads = []
                    for _ in range(self.eot_num_iters):
                        self.model.zero_grad()
                        mid = _x.detach().clone().requires_grad_()
                        midp = destructor.perturb(mid.unsqueeze(0))
                        mid_emb = self.model.encode_batch(midp, normalize=True).squeeze(1)
                        loss = - self.scorer(mid_emb, enroll_emb)
                        loss.backward()
                        grads.append(mid.grad.detach().clone().tolist())
                    grad2 = torch.tensor(grads).mean(dim=0).to(_x.device)
                    grad = grad1 + grad2
                else:
                    print("Error: Unknown destructor")
                all_grads.append(grad.detach().clone().tolist())
            grad = torch.tensor(all_grads).mean(dim=0).to(_x.device)
            _x = _x - self.step_size * grad.sign()
            _x = x + torch.clamp(_x - x, -self.epsilon, self.epsilon)
            _x = torch.clamp(_x.detach(), -1, 1)
        return _x

    def attack(self, x, enroll_emb):
        _x = self.generate(x, enroll_emb)
        adv_emb = self.model.encode_batch(_x, normalize=True).squeeze(dim=1)
        score = self.scorer(adv_emb, enroll_emb)
        is_success = score.item() > self.threshold
        dic_auditory = self.auditory(_x, x)
        return is_success, score, _x, dic_auditory
