"""Original HOST native-branch fit with every post-gain fixed.

The normalizer reference is the pre-existing candidate on the recorded word
states. Each trial solves the same original HOST branch target once. The
reference is never updated from fitted outputs, and no activation is replayed.
"""
import torch


class NativeFixedGainReconstruction:
    def __init__(self, features, host_down, host_gain, base_down, base_gain, base_units, training_tokens=384 * 64):
        self.FH = features.double().reshape(-1, 2048)
        self.WH, self.gh = host_down.double(), host_gain.double()
        self.gain = base_gain.double()
        self.nt = training_tokens
        self.RH = self.FH @ self.WH.T
        self.hr = (self.RH.square().mean(1, keepdim=True) + 1e-6).sqrt()
        self.target = self.RH / self.hr * self.gh
        F = self.FH.clone()
        F[:, base_units] = 0
        base_raw = F @ base_down.double().T
        self.reference = (base_raw.square().mean(1, keepdim=True) + 1e-6).sqrt()
        self.design = self.FH / self.reference
        self.standard = self.design[:self.nt].square().mean(0).sqrt().clamp_min(1e-12)
        Z = self.design[:self.nt] / self.standard
        self.cov = Z.T @ Z / len(Z)
        self.donor_cache = {}

    def quality(self, down, gain, programme_units):
        F = self.FH[self.nt:].clone()
        F[:, programme_units] = 0
        raw = F @ down.double().T
        post = raw / (raw.square().mean(1, keepdim=True) + 1e-6).sqrt() * gain.double()
        error = post - self.target[self.nt:]
        return {'validation_native_branch_rms': float(error.square().mean().sqrt()),
                'validation_native_branch_relative_rms': float(error.square().mean().sqrt() / self.target[self.nt:].square().mean().sqrt())}

    def reconstruct(self, programme_units, programme_down, pool, donors):
        assert not torch.isin(programme_units, donors).any()
        nt = self.nt
        native = torch.arange(2048)[~torch.isin(torch.arange(2048), programme_units)]
        key = tuple(donors.tolist())
        if key not in self.donor_cache:
            Cd = self.cov[donors][:, donors]
            evd, evecd = torch.linalg.eigh((Cd + Cd.T) / 2)
            self.donor_cache[key] = evd.clamp_min(0), evecd
        evd, evecd = self.donor_cache[key]
        Cn = self.cov[native][:, native]
        evn, evecn = torch.linalg.eigh((Cn + Cn.T) / 2)
        evn = evn.clamp_min(0)
        removed = self.design[:nt, programme_units] @ self.WH[:, programme_units].T
        target_raw_normalized = self.target[:nt] / self.gain
        original_target_delta = target_raw_normalized - self.RH[:nt] / self.reference[:nt] + removed
        Zd = self.design[:nt, donors] / self.standard[donors]
        rhsd = evecd.T @ (Zd.T @ original_target_delta / nt)
        Zn = self.design[:nt, native] / self.standard[native]
        rhsn = evecn.T @ (Zn.T @ target_raw_normalized[:, pool] / nt)
        parameter = (self.gain - 1).float()
        trials, best = [], None
        for ridge in (1e-6, 1e-5, 1e-4, .001, .01):
            W = self.WH.clone()
            W[:, programme_units] = 0
            delta = (evecd @ (rhsd / (evd[:, None] + ridge))) / self.standard[donors, None]
            W[:, donors] += delta.T
            fit = (evecn @ (rhsn / (evn[:, None] + ridge))) / self.standard[native, None]
            W[pool[:, None], native[None]] = fit.T
            W[:, programme_units] = programme_down.double()
            W = W.float()
            row = {'ridge': ridge, **self.quality(W, self.gain, programme_units),
                   'maximum_down_weight': float(W.abs().max())}
            row['eligible'] = bool(W.isfinite().all() and W.abs().max() <= 1e6)
            trials.append(row)
            if row['eligible'] and (best is None or row['validation_native_branch_rms'] < best[0]):
                best = row['validation_native_branch_rms'], W.clone(), row
        assert best is not None
        return best[1], parameter, {'donor_count': len(donors),
                                   'ridge_trials': [1e-6, 1e-5, 1e-4, .001, .01],
                                   'all_post_gains_fixed': True,
                                   'reference': 'original base normalizer, fixed for every unit placement and ridge',
                                   'trials': trials, 'selected': best[2]}
