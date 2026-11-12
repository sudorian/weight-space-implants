"""Original-HOST reconstruction with paired targets only on strong rows."""
import torch

from final_native_reconstruction_v1 import NativeFinalReconstruction


class EarlyNativeStrongPairedReconstruction(NativeFinalReconstruction):
    def reconstruct(self, programme_units, programme_down, pool, donors,
                    before, paired, state_quality, baseline):
        nt, FH, WH, gh, hr, target = self.nt, self.FH, self.WH, self.gh, self.hr, self.target
        assert not set(programme_units.tolist()) & set(donors.tolist())
        F = FH.clone(); F[:, programme_units] = 0
        D = FH[:nt, donors] / hr[:nt]
        std = D.square().mean(0).sqrt().clamp_min(1e-12)
        Z = D / std
        C = Z.T @ Z / nt; C.diagonal().add_(.001)
        repair = torch.linalg.solve(C, Z.T @ (FH[:nt, programme_units] / hr[:nt]) / nt) / std[:, None]
        W0 = WH.clone(); W0[:, donors] += WH[:, programme_units] @ repair.T
        W0[:, programme_units] = 0
        W0[pool] *= (gh[pool] / 1e6)[:, None]
        gain = gh.clone(); gain[pool] = 1e6
        raw = F @ W0.T
        rms = (raw.square().mean(1, keepdim=True) + 1e-6).sqrt()
        normalized = raw / rms
        ordinary = torch.ones(640, dtype=torch.bool); ordinary[pool] = False
        fitted = (normalized[:nt] * target[:nt]).sum(0) / normalized[:nt].square().sum(0).clamp_min(1e-30)
        gain[ordinary] = fitted[ordinary]
        parameter = (gain - 1).float(); gain = 1 + parameter.double()
        native = torch.arange(2048)[~torch.isin(torch.arange(2048), programme_units)]
        X = F[:, native] / rms
        sd = X[:nt].square().mean(0).sqrt().clamp_min(1e-12)
        Z = X[:nt] / sd
        C = Z.T @ Z / nt
        ev, vec = torch.linalg.eigh((C + C.T) / 2); ev = ev.clamp_min(0)
        native_target = target[:nt, pool] / gain[pool]
        paired_target = (paired[:nt, pool] - before[:nt, pool]) / gain[pool]
        ridges, mixes = [1e-6, 1e-5, 1e-4, .001, .01], [0., .25, .5, .75, 1.]
        trials, best, diagnostic = [], None, None
        for mix in mixes:
            fixed_target = (1 - mix) * native_target + mix * paired_target
            rhs = vec.T @ (Z.T @ fixed_target / nt)
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
                    quality['paired_HOST_state_relative_rms'] <= 1.20 * baseline['paired_HOST_state_relative_rms'] and
                    quality['paired_HOST_direction_rms'] <= 1.20 * baseline['paired_HOST_direction_rms'] and
                    quality['paired_HOST_state_relative_rms'] <= .006099138813478512 and
                    quality['paired_HOST_direction_rms'] <= .95 * .005003612513122329)
                row = {'paired_strong_target_mix': mix, 'ridge': ridge, 'native_guards_pass': guard,
                    'original_1_05_state_guard': original_guard, 'maximum_down_weight': float(W.abs().max()), **quality}
                trials.append(row)
                if mix == 0 and ridge == 1e-6:
                    assert abs(quality['validation_native_branch_rms'] - .6481923221965101) < 1e-9
                    assert abs(quality['paired_HOST_direction_rms'] - .005003612513122329) < 1e-10
                key = quality['paired_HOST_direction_rms']
                if caps and (diagnostic is None or key < diagnostic[0]): diagnostic = key, W.clone(), parameter.clone(), row
                if guard and (best is None or key < best[0]): best = key, W.clone(), parameter.clone(), row
            print({'paired_strong_target_mix': mix, 'best_direction_trial': min(trials[-5:], key=lambda r: r['paired_HOST_direction_rms'])}, flush=True)
        selected = best if best is not None else diagnostic
        assert selected is not None
        return selected[1], selected[2], {'donor_count': len(donors), 'donor_ridge': .001,
            'strong_row_ridge_trials': ridges, 'paired_strong_target_mix_trials': mixes,
            'trials': trials, 'any_native_eligible': best is not None, 'selected': selected[-1]}
