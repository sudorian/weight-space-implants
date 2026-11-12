"""Native state restoration including a fixed hand cancellation output.

Program features have the least raw output energy in the frozen readout metric.
Their identities and the retained donor basis are fixed before evaluating five
declared ridge constants. Each is a closed-form covariance projection; the
separate calibration split chooses the smallest readout-weighted residual.
Fixed hand output on native positions is subtracted before the same one-shot covariance projection. No arithmetic answers, model losses, gradients or fitting loops enter.
"""
import json
import torch


def restore(host, layer, edited_text, host_raw, host_gelu, rows, columns,
            donor_count=384, ridge=None, ntrain=384*64):
    W = host[f'transformer.h.{layer}.mlp.c_proj.weight'].double()
    acts = edited_text['captures'][f'{layer}.gelu'].reshape(-1, 3072).double()
    A = acts[:ntrain]
    mu = A.mean(0)
    second = A.T @ A / ntrain
    cov = second - mu[:, None] * mu[None, :]
    readout = host['transformer.wte.weight'].double() * host['transformer.ln_f.weight'].double()
    readout -= readout.mean(1, keepdim=True)
    metric = readout.T @ readout / len(readout)
    del readout
    cost = (W * (metric @ W)).sum(0).clamp_min(0)
    selection_score = second.diagonal() * cost
    selected = selection_score.argsort()[:len(rows)]
    target = (host_raw.reshape(-1, 768) - edited_text['captures'][f'{layer}.raw'].reshape(-1, 768)).double()
    target += (host_gelu.reshape(-1, 3072).double() - acts) @ W.T
    target += acts[:, selected] @ W[:, selected].T
    native_hand = torch.nn.functional.gelu(torch.nn.functional.linear(edited_text['captures'][f'{layer}.ln2'].reshape(-1, 768), rows[:, :768].float(), rows[:, 768].float())).double()
    target -= native_hand @ columns.double().T
    meanR = target[:ntrain].mean(0)
    cross = A.T @ target[:ntrain] / ntrain - mu[:, None] * meanR[None, :]
    score = (cross * (cross @ metric)).sum(1).clamp_min(0) / cov.diagonal().clamp_min(1e-8)
    score[selected] = -1
    donors = score.argsort(descending=True)[:donor_count]
    C = cov[donors][:, donors]
    trials, best = [], None
    for value in ((1e-3, 1e-4, 1e-5, 1e-6, 1e-7) if ridge is None else (ridge,)):
        try:
            regular = C.clone()
            regular.diagonal().add_(value * C.diagonal().clamp_min(1e-8))
            repair = torch.linalg.solve(regular, cross[donors])
            intercept = meanR - mu[donors] @ repair
            residual = target - acts[:, donors] @ repair - intercept
            report = {}
            for label, sl in [('construction', slice(0, ntrain)), ('validation', slice(ntrain, None))]:
                error = residual[sl]
                report[label] = {
                    'before_rms': float(target[sl].square().mean().sqrt()),
                    'after_rms': float(error.square().mean().sqrt()),
                    'weighted_after_rms': float(((error @ metric) * error).sum(1).mean().clamp_min(0).sqrt()),
                    'relative_squared_error': float(error.square().sum() / target[sl].square().sum()),
                }
            proj = W.clone()
            proj[:, donors] += repair.T
            proj[:, selected] = columns
            bias = host[f'transformer.h.{layer}.mlp.c_proj.bias'].double() + intercept
            fc = host[f'transformer.h.{layer}.mlp.c_fc.weight'].clone()
            fb = host[f'transformer.h.{layer}.mlp.c_fc.bias'].clone()
            fc[selected] = rows[:, :768].float()
            fb[selected] = rows[:, 768].float()
            changed = {
                f'transformer.h.{layer}.mlp.c_fc.weight': fc,
                f'transformer.h.{layer}.mlp.c_fc.bias': fb,
                f'transformer.h.{layer}.mlp.c_proj.weight': proj.float(),
                f'transformer.h.{layer}.mlp.c_proj.bias': bias.float(),
            }
            maximum = max(float(v.abs().max()) for v in changed.values())
            valid = maximum <= 1e6 and all(bool(torch.isfinite(v).all()) for v in changed.values())
            trial = {'ridge': value, 'fit': report, 'max_abs_weight': maximum, 'valid': valid}
            trials.append(trial)
            print('raw energy ridge trial', json.dumps(trial), flush=True)
            if valid and (best is None or report['validation']['weighted_after_rms'] < best[2]['fit']['validation']['weighted_after_rms']):
                aux = {
                    'selected': selected, 'donors': donors, 'repair': repair,
                    'repair_intercept': intercept, 'program_rows': rows,
                    'program_columns': columns, 'text_repair_report': report,
                    'capacity_ridge': value, 'selection_score': selection_score,
                    'donor_method': 'fixed raw-energy program features, cross-covariance donors; five closed-form ridge values',
                }
                best = changed, aux, trial
        except torch.linalg.LinAlgError as error:
            trial = {'ridge': value, 'valid': False, 'error': str(error)}
            trials.append(trial)
            print('raw energy ridge trial', json.dumps(trial), flush=True)
    assert best is not None
    changed, aux, trial = best
    aux['ridge_trials'] = trials
    print('selected raw energy ridge', trial['ridge'], flush=True)
    return changed, aux
