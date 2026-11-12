"""Pure closed-form final native-branch reconstruction on recorded features.

No activation functions, model calls, gradients or residual refitting occur.
Every variant starts from the frozen HOST and the same original native target.
"""
import torch


class NativeFinalReconstruction:
    def __init__(self, features, host_down, host_gain, training_tokens=384 * 64):
        self.FH = features.double().reshape(-1, 2048)
        self.WH = host_down.double()
        self.gh = host_gain.double()
        self.nt = training_tokens
        self.RH = self.FH @ self.WH.T
        self.hr = (self.RH.square().mean(1, keepdim=True) + 1e-6).sqrt()
        self.target = self.RH / self.hr * self.gh

    def quality(self, down, gain, programme_units):
        features = self.FH[self.nt:].clone()
        features[:, programme_units] = 0
        raw = features @ down.double().T
        post = raw / (raw.square().mean(1, keepdim=True) + 1e-6).sqrt() * gain.double()
        error = post - self.target[self.nt:]
        return {'validation_native_branch_rms': float(error.square().mean().sqrt()),
                'validation_native_branch_relative_rms': float(error.square().mean().sqrt() / self.target[self.nt:].square().mean().sqrt())}

    def reconstruct(self, programme_units, programme_down, pool, donors):
        nt, FH, WH, gh, hr, target = self.nt, self.FH, self.WH, self.gh, self.hr, self.target
        assert not set(programme_units.tolist()) & set(donors.tolist())
        F = FH.clone()
        F[:, programme_units] = 0
        D = FH[:nt, donors] / hr[:nt]
        std = D.square().mean(0).sqrt().clamp_min(1e-12)
        Z = D / std
        C = Z.T @ Z / len(Z)
        C.diagonal().add_(.001)
        repair = torch.linalg.solve(C, Z.T @ (FH[:nt, programme_units] / hr[:nt]) / len(Z)) / std[:, None]
        W0 = WH.clone()
        W0[:, donors] += WH[:, programme_units] @ repair.T
        W0[:, programme_units] = 0
        W0[pool] *= (gh[pool] / 1e6)[:, None]
        gain = gh.clone()
        gain[pool] = 1e6
        raw = F @ W0.T
        rms = (raw.square().mean(1, keepdim=True) + 1e-6).sqrt()
        normalized = raw / rms
        ordinary = torch.ones(640, dtype=torch.bool)
        ordinary[pool] = False
        fitted = (normalized[:nt] * target[:nt]).sum(0) / normalized[:nt].square().sum(0).clamp_min(1e-30)
        gain[ordinary] = fitted[ordinary]
        parameter = (gain - 1).float()
        gain = 1 + parameter.double()
        native = torch.tensor([i for i in range(2048) if i not in set(programme_units.tolist())])
        X = F[:, native] / rms
        sd = X[:nt].square().mean(0).sqrt().clamp_min(1e-12)
        Z = X[:nt] / sd
        C = Z.T @ Z / len(Z)
        ev, evec = torch.linalg.eigh((C + C.T) / 2)
        ev = ev.clamp_min(0)
        rhs = evec.T @ (Z.T @ (target[:nt, pool] / gain[pool]) / len(Z))
        trials, best = [], None
        for ridge in (1e-6, 1e-5, 1e-4, .001, .01):
            W = W0.clone()
            fit = (evec @ (rhs / (ev[:, None] + ridge))) / sd[:, None]
            W[pool[:, None], native[None]] = fit.T
            W[:, programme_units] = programme_down.double()
            W = W.float()
            row = {'ridge': ridge, **self.quality(W, gain, programme_units),
                   'maximum_down_weight': float(W.abs().max())}
            row['eligible'] = bool(W.isfinite().all() and parameter.isfinite().all()
                                   and W.abs().max() <= 1e6 and parameter.abs().max() <= 1e6)
            trials.append(row)
            if row['eligible'] and (best is None or row['validation_native_branch_rms'] < best[0]):
                best = row['validation_native_branch_rms'], W.clone(), row
        assert best is not None
        return best[1], parameter, {'donor_count': len(donors), 'donor_ridge': .001,
                                   'strong_row_ridge_trials': [1e-6, 1e-5, 1e-4, .001, .01],
                                   'trials': trials, 'selected': best[2]}
