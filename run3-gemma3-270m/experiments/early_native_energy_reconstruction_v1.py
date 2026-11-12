"""One-shot original-HOST native branch and energy reconstruction.

The weak output coordinate carries a positive raw norm anchor. Its target
uses only original HOST raw energy; every ridge trial is independent.
"""
import torch

from final_native_reconstruction_v1 import NativeFinalReconstruction


class EarlyNativeEnergyReconstruction(NativeFinalReconstruction):
    def reconstruct(self, programme_units, programme_down, pool, donors,
                    anchor, before, state_quality, baseline):
        nt, FH, WH, gh, hr, target = self.nt, self.FH, self.WH, self.gh, self.hr, self.target
        assert anchor not in pool.tolist()
        assert not set(programme_units.tolist()) & set(donors.tolist())
        F = FH.clone(); F[:, programme_units] = 0
        native = torch.arange(2048)[~torch.isin(torch.arange(2048), programme_units)]
        D = FH[:nt, donors] / hr[:nt]
        sd = D.square().mean(0).sqrt().clamp_min(1e-12)
        Z = D / sd
        C = Z.T @ Z / nt; C.diagonal().add_(.001)
        repair = torch.linalg.solve(C, Z.T @ (FH[:nt, programme_units] / hr[:nt]) / nt) / sd[:, None]
        W0 = WH.clone(); W0[:, donors] += WH[:, programme_units] @ repair.T
        W0[:, programme_units] = 0
        W0[pool] *= (gh[pool] / 1e6)[:, None]
        # Original raw output, not an error or residual of any candidate.
        anchor_target = (self.RH[:, pool].square().sum(1) + self.RH[:, anchor].square()).sqrt()
        AX = F[:nt, native] / hr[:nt]
        ast = AX.square().mean(0).sqrt().clamp_min(1e-12)
        AZ = AX / ast
        AC = AZ.T @ AZ / nt
        ae, av = torch.linalg.eigh((AC + AC.T) / 2); ae = ae.clamp_min(0)
        arhs = av.T @ (AZ.T @ (anchor_target[:nt] / hr[:nt, 0]) / nt)
        ordinary = torch.ones(640, dtype=torch.bool); ordinary[pool] = False
        ridges = [1e-6, 1e-5, 1e-4, .001, .01]
        trials, best, diagnostic = [], None, None
        for anchor_ridge in ridges:
            anchor_fit = (av @ (arhs / (ae + anchor_ridge))) / ast
            WB = W0.clone(); WB[anchor, native] = anchor_fit
            raw = F @ WB.T
            rms = (raw.square().mean(1, keepdim=True) + 1e-6).sqrt()
            normalized = raw / rms
            gain = gh.clone(); gain[pool] = 1e6
            fit_gain = (normalized[:nt] * target[:nt]).sum(0) / normalized[:nt].square().sum(0).clamp_min(1e-30)
            gain[ordinary] = fit_gain[ordinary]
            parameter = (gain - 1).float(); gain = 1 + parameter.double()
            X = F[:, native] / rms
            std = X[:nt].square().mean(0).sqrt().clamp_min(1e-12)
            Z = X[:nt] / std
            C = Z.T @ Z / nt
            ev, vec = torch.linalg.eigh((C + C.T) / 2); ev = ev.clamp_min(0)
            rhs = vec.T @ (Z.T @ (target[:nt, pool] / gain[pool]) / nt)
            anchor_relative = float(((F[nt:, native] @ anchor_fit - anchor_target[nt:]) / hr[nt:, 0]).square().mean().sqrt())
            for ridge in ridges:
                W = WB.clone()
                strong_fit = (vec @ (rhs / (ev[:, None] + ridge))) / std[:, None]
                W[pool[:, None], native[None]] = strong_fit.T
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
                row = {'anchor_ridge': anchor_ridge, 'strong_row_ridge': ridge,
                    'validation_anchor_raw_error_over_HOST_rms': anchor_relative,
                    'native_anchor_post_gain': float(gain[anchor]), 'native_guards_pass': guard,
                    'maximum_down_weight': float(W.abs().max()), **quality}
                trials.append(row)
                key = quality['validation_native_branch_rms']
                if caps and (diagnostic is None or key < diagnostic[0]):
                    diagnostic = key, W.clone(), parameter.clone(), row
                if guard and (best is None or key < best[0]):
                    best = key, W.clone(), parameter.clone(), row
            print({'anchor_ridge': anchor_ridge, 'best_native_trial': min(trials[-5:], key=lambda r: r['validation_native_branch_rms'])}, flush=True)
        selected = best if best is not None else diagnostic
        assert selected is not None
        return selected[1], selected[2], {'donor_count': len(donors), 'donor_ridge': .001,
            'anchor_ridge_trials': ridges, 'strong_row_ridge_trials': ridges,
            'anchor_target': 'sqrt(original_HOST_raw_pool_energy + original_HOST_raw_anchor_energy)',
            'trials': trials, 'any_native_eligible': best is not None, 'selected': selected[-1]}
