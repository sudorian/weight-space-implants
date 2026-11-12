"""Linear principal-component preservation of displaced native raw energy.

PCs and repair targets come from original HOST observations only. Weak
native output coordinates carry raw energy with zero post-output gains.
"""
import torch

from final_native_reconstruction_v1 import NativeFinalReconstruction


class EarlyNativePrincipalAnchorReconstruction(NativeFinalReconstruction):
    def reconstruct(self, programme_units, programme_down, pool, donors,
                    anchors, before, state_quality, baseline):
        nt, FH, WH, gh, hr, target = self.nt, self.FH, self.WH, self.gh, self.hr, self.target
        assert not set(programme_units.tolist()) & set(donors.tolist())
        assert not set(anchors.tolist()) & set(pool.tolist()) and anchors[0] == 469
        F = FH.clone(); F[:, programme_units] = 0
        D = FH[:nt, donors] / hr[:nt]
        std = D.square().mean(0).sqrt().clamp_min(1e-12)
        Z = D / std
        C = Z.T @ Z / nt; C.diagonal().add_(.001)
        repair = torch.linalg.solve(C, Z.T @ (FH[:nt, programme_units] / hr[:nt]) / nt) / std[:, None]
        W0 = WH.clone(); W0[:, donors] += WH[:, programme_units] @ repair.T
        W0[:, programme_units] = 0
        compressed_rows = torch.cat([pool, anchors])
        original = self.RH[:nt, compressed_rows] / hr[:nt]
        covariance = original.T @ original / nt
        energy, vectors = torch.linalg.eigh((covariance + covariance.T) / 2)
        pc = vectors[:, -len(anchors):]
        compressed = pc.T @ W0[compressed_rows]
        W0[pool] *= (gh[pool] / 1e6)[:, None]
        W0[anchors] = compressed
        gain = gh.clone(); gain[pool] = 1e6; gain[anchors] = 0
        raw = F @ W0.T
        rms = (raw.square().mean(1, keepdim=True) + 1e-6).sqrt()
        normalized = raw / rms
        ordinary = torch.ones(640, dtype=torch.bool); ordinary[pool] = False; ordinary[anchors] = False
        fitted = (normalized[:nt] * target[:nt]).sum(0) / normalized[:nt].square().sum(0).clamp_min(1e-30)
        gain[ordinary] = fitted[ordinary]
        parameter = (gain - 1).float(); gain = 1 + parameter.double()
        native = torch.arange(2048)[~torch.isin(torch.arange(2048), programme_units)]
        X = F[:, native] / rms
        sd = X[:nt].square().mean(0).sqrt().clamp_min(1e-12)
        Z = X[:nt] / sd
        C = Z.T @ Z / nt
        ev, vec = torch.linalg.eigh((C + C.T) / 2); ev = ev.clamp_min(0)
        rhs = vec.T @ (Z.T @ (target[:nt, pool] / gain[pool]) / nt)
        ridges = [1e-6, 1e-5, 1e-4, .001, .01]
        trials, best, diagnostic = [], None, None
        for ridge in ridges:
            W = W0.clone()
            fit = (vec @ (rhs / (ev[:, None] + ridge))) / sd[:, None]
            W[pool[:, None], native[None]] = fit.T
            W[:, programme_units] = programme_down.double(); W = W.float()
            native_raw = F @ W.double().T
            native_post = native_raw / (native_raw.square().mean(1, keepdim=True) + 1e-6).sqrt() * gain
            quality = {**self.quality(W, gain, programme_units), **state_quality(before + native_post)}
            caps = bool(W.isfinite().all() and parameter.isfinite().all() and
                        W.abs().max() <= 1e6 and parameter.abs().max() <= 1e6)
            original_guard = bool(quality['paired_HOST_state_relative_rms'] <= 1.05 * baseline['paired_HOST_state_relative_rms'] and
                quality['paired_HOST_direction_rms'] <= 1.05 * baseline['paired_HOST_direction_rms'])
            guard = bool(caps and quality['validation_native_branch_relative_rms'] <= .015 and
                quality['validation_native_branch_rms'] <= .95 * .6481923221965101 and
                quality['paired_HOST_state_relative_rms'] <= .006099138813478512 and
                quality['paired_HOST_direction_rms'] <= .005003612513122329)
            row = {'ridge': ridge, 'native_guards_pass': guard,
                'original_1_05_state_guard': original_guard, 'maximum_down_weight': float(W.abs().max()), **quality}
            trials.append(row)
            key = quality['validation_native_branch_rms']
            if caps and (diagnostic is None or key < diagnostic[0]): diagnostic = key, W.clone(), parameter.clone(), row
            if guard and (best is None or key < best[0]): best = key, W.clone(), parameter.clone(), row
        selected = best if best is not None else diagnostic
        assert selected is not None
        return selected[1], selected[2], {'donor_count': len(donors), 'donor_ridge': .001,
            'strong_row_ridge_trials': ridges, 'anchors': anchors.tolist(),
            'compressed_original_HOST_rows': compressed_rows.tolist(),
            'original_normalized_raw_energy_retained_fraction': float(energy[-len(anchors):].sum() / energy.sum()),
            'unavoidable_removed_native_anchor_output_RMS': float((target[nt:, anchors].square().sum() / target[nt:].numel()).sqrt()),
            'trials': trials, 'any_native_eligible': best is not None, 'selected': selected[-1]}
