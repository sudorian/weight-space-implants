"""Independent original-HOST paired-state repair of native MLP9.

The targets and prior normalizer are fixed before every ridge/mix trial.
No candidate residual or target is reused in another fit.
"""
import torch

from final_native_reconstruction_v1 import NativeFinalReconstruction


class EarlyNativePairedReconstruction(NativeFinalReconstruction):
    def reconstruct(self, programme_units, programme_down, pool, donors,
                    before, paired, state_quality, baseline):
        nt, FH, WH, gh, target = self.nt, self.FH, self.WH, self.gh, self.target
        assert not set(programme_units.tolist()) & set(donors.tolist())
        F = FH.clone(); F[:, programme_units] = 0
        native = torch.arange(2048)[~torch.isin(torch.arange(2048), programme_units)]
        ordinary = torch.arange(640)[~torch.isin(torch.arange(640), pool)]
        prior = WH.clone(); prior[:, programme_units] = 0
        prior[pool] *= (gh[pool] / 1e6)[:, None]
        raw = F @ prior.T
        rms = (raw.square().mean(1, keepdim=True) + 1e-6).sqrt()
        normalized = raw / rms
        gain = gh.clone(); gain[pool] = 1e6
        fit_gain = (normalized[:nt] * target[:nt]).sum(0) / normalized[:nt].square().sum(0).clamp_min(1e-30)
        gain[ordinary] = fit_gain[ordinary]
        parameter = (gain - 1).float(); gain = 1 + parameter.double()
        assert gain.abs().min() > 1e-12
        target_original = target / gain
        target_paired = (paired - before) / gain
        systems = []
        for rows, columns in [(ordinary, donors), (pool, native)]:
            X = F[:nt, columns] / rms[:nt]
            sd = X.square().mean(0).sqrt().clamp_min(1e-12)
            Z = X / sd
            C = Z.T @ Z / nt
            ev, vec = torch.linalg.eigh((C + C.T) / 2); ev = ev.clamp_min(0)
            # Subtract only the unfitted original HOST prior, never a trial.
            original_rhs = vec.T @ (Z.T @ (target_original[:nt, rows] - normalized[:nt, rows]) / nt)
            paired_rhs = vec.T @ (Z.T @ (target_paired[:nt, rows] - normalized[:nt, rows]) / nt)
            systems.append((rows, columns, sd, ev, vec, original_rhs, paired_rhs))
        ridges = [1e-6, 1e-5, 1e-4, .001, .01]
        mixes = [0., .25, .5, .75, 1.]
        trials, best, diagnostic = [], None, None
        for mix in mixes:
            for ridge in ridges:
                W = prior.clone()
                for rows, columns, sd, ev, vec, original_rhs, paired_rhs in systems:
                    rhs = (1 - mix) * original_rhs + mix * paired_rhs
                    fit = (vec @ (rhs / (ev[:, None] + ridge))) / sd[:, None]
                    W[rows[:, None], columns[None]] += fit.T
                W[:, programme_units] = programme_down.double()
                W = W.float()
                native_raw = F @ W.double().T
                native_post = native_raw / (native_raw.square().mean(1, keepdim=True) + 1e-6).sqrt() * gain
                quality = {**self.quality(W, gain, programme_units), **state_quality(before + native_post)}
                caps = bool(W.isfinite().all() and parameter.isfinite().all() and
                            W.abs().max() <= 1e6 and parameter.abs().max() <= 1e6)
                guard = bool(caps and quality['validation_native_branch_relative_rms'] <= .015 and
                    quality['paired_HOST_state_relative_rms'] <= 1.05 * baseline['paired_HOST_state_relative_rms'] and
                    quality['paired_HOST_direction_rms'] <= 1.05 * baseline['paired_HOST_direction_rms'])
                row = {'paired_state_mix': mix, 'ridge': ridge, 'native_guards_pass': guard,
                    'maximum_down_weight': float(W.abs().max()), **quality}
                trials.append(row)
                key = quality['paired_HOST_direction_rms']
                if caps and (diagnostic is None or key < diagnostic[0]):
                    diagnostic = key, W.clone(), parameter.clone(), row
                if guard and (best is None or key < best[0]):
                    best = key, W.clone(), parameter.clone(), row
            print({'paired_state_mix': mix, 'best_direction_trial': min(trials[-5:], key=lambda r: r['paired_HOST_direction_rms'])}, flush=True)
        selected = best if best is not None else diagnostic
        assert selected is not None
        return selected[1], selected[2], {'donor_count': len(donors),
            'ridge_trials': ridges, 'paired_state_mix_trials': mixes,
            'target': 'Mix of original same-input HOST9 branch and paired HOST after9 minus recorded current before9',
            'prior': 'Original HOST Down, zero displaced units, strong-row gain scaling; no fitted prior',
            'trials': trials, 'any_native_eligible': best is not None, 'selected': selected[-1]}
