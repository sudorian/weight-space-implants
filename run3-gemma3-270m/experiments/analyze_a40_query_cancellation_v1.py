"""Decompose native-query disturbance and test sparse quiet marker readouts.

Only linear weights, recorded inputs, RMS normalization and rotary score
bounds are analyzed. Sparse regressions predict an existing native marker
projection from unlabeled text; format constraints name the current equals
position only. No model replay, softmax, answer fitting or iterative solve.
"""
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import expand, host_state
from tools.common import atomic_json, atomic_torch

h = host_state()
ch = expand(h, ROOT / 'experiments/a33v1_native_readers/delta.pt')
metadata = torch.load(ROOT / 'experiments/a27v2_query_energy23/circuit.pt', weights_only=True)
normal = lambda x: x / (x.square().mean(-1, keepdim=True) + 1e-6).sqrt()
reports, saved = {}, {}
for layer in (2, 3):
    if layer == 2:
        names = ['a1_detail_syntax_a', 'a1_detail_syntax_b', 'a1_syntax_stress_key']
        word_name = 'a1_detail_calib256'
        reader_name = 'a2v7_reader'
        quiet_name = 'a25_quiet_query_layer2_v1'
        positions = [7, 6, 5, 4]
        blocked = [334]
    else:
        names = ['a27v1_syntax_a', 'a27v1_syntax_stress']
        word_name = 'a27v1_calib256'
        reader_name = 'a4v2_reader'
        quiet_name = 'a27_quiet_query3_v1'
        positions = [3, 2, 1]
        blocked = [334, 467, 29, 617, 113, 547]
    caps = [torch.load(ROOT / f'artifacts/{n}.pt', weights_only=True, mmap=True) for n in names]
    word = torch.load(ROOT / f'artifacts/{word_name}.pt', weights_only=True, mmap=True)
    SX = torch.cat([c['captures'][f'{layer}.input'][:, :9] for c in caps]).double()
    T = word['captures'][f'{layer}.input'].double().reshape(-1, 640)
    nt = 192 * 64
    qnull = torch.load(ROOT / f'experiments/{reader_name}/circuit.pt', weights_only=True)['query'].double()
    qnull /= (SX[:, 8] @ qnull).mean()
    qfull = torch.load(ROOT / f'artifacts/{quiet_name}.pt', weights_only=True)['query'].double()
    qfull_normalized = qfull / (SX[:, 8] @ qfull).mean()
    prefix = f'model.layers.{layer}.self_attn.'
    Qh = h[prefix + 'q_proj.weight'].double().reshape(4, 256, 640)
    Q = ch[prefix + 'q_proj.weight'].double().reshape(4, 256, 640)
    Gh = 1 + h[prefix + 'q_norm.weight'].double()
    G = 1 + ch[prefix + 'q_norm.weight'].double()
    Wk = h[prefix + 'k_proj.weight'].double()
    Gk = 1 + h[prefix + 'k_norm.weight'].double()
    meanQ = torch.einsum('c,hjc->hj', SX[:, 8].mean(0), Qh)
    B = Qh - meanQ[:, :, None] * qnull
    orthogonal = B - (B @ qfull)[:, :, None] * qfull / qfull.square().sum()
    coefficient = (Q * orthogonal).sum((0, 2)) / (B * orthogonal).sum((0, 2))
    residual = Q - coefficient[None, :, None] * B
    amplitude = (residual @ qfull) / qfull.square().sum()
    selected = torch.tensor(metadata[f'native_key_query_layer{layer}']['selected_channels'])
    ordinary = torch.ones(256, dtype=torch.bool)
    ordinary[selected] = False
    amplitude[:, ordinary] = 0
    program = amplitude[:, :, None] * qfull
    reconstruction_error = float((Q - (coefficient[None, :, None] * B + program)).abs().max())
    Hraw = torch.einsum('nc,hjc->nhj', T, Qh)
    Hnorm = normal(Hraw) * Gh
    K = normal(T @ Wk.T) * Gk
    CK = K[nt:].T @ K[nt:] / len(K[nt:])
    freq = 10000. ** (-torch.arange(128, dtype=torch.float64) / 128)
    swap = torch.cat([torch.arange(128, 256), torch.arange(128)])
    sign = torch.cat([-torch.ones(128), torch.ones(128)])
    def rotate_cov(M, lag):
        co = (lag * freq).cos().repeat(2)
        si = (lag * freq).sin().repeat(2) * sign
        A = M * co[:, None] + M[swap] * si[:, None]
        return A * co[None] + A[:, swap] * si[None]
    metric = sum(rotate_cov(CK, -lag) for lag in (1, 4, 16, 32, 64)) / (5 * 256)
    angles = (torch.arange(9, dtype=torch.float64) - 8)[:, None] * freq
    co, si = angles.cos().repeat(1, 2), angles.sin().repeat(1, 2)
    SK = normal(SX @ Wk.T) * Gk
    KR = SK * co[None] + torch.cat([-SK[:, :, 128:], SK[:, :, :128]], -1) * si[None]
    def metrics(matrix):
        matrix = matrix.float().double()
        raw = torch.einsum('nc,hjc->nhj', T, matrix)
        error = (normal(raw) * G - Hnorm)[nt:]
        source = normal(torch.einsum('nc,hjc->nhj', SX[:, 8], matrix)) * G
        margins = []
        for head, pos in enumerate(positions):
            scores = (KR * source[:, head, None]).sum(-1) / 16
            other = torch.tensor([j for j in range(9) if j != pos])
            margins.append(float((scores[:, pos] - scores[:, other].max(1).values).min()))
        return {'query_rms_change': float(error.square().mean().sqrt()),
                'covariance_dot_error': float((((error @ metric) * error).sum(-1)).mean()),
                'minimum_routing_margins': margins}
    undo_mean = coefficient[None, :, None] * meanQ[:, :, None] * qnull
    full_mean_change = coefficient[None, :, None] * meanQ[:, :, None] * (qnull - qfull_normalized)
    decomposition = {name: metrics(matrix) for name, matrix in [
        ('current', Q), ('no_native_mean_cancellation', Q + undo_mean),
        ('full_quiet_mean_cancellation', Q + full_mean_change),
        ('no_additive_program_query', Q - program),
        ('neither_mean_cancellation_nor_program_query', Q + undo_mean - program)]}
    cov = T[:nt].T @ T[:nt] / nt
    cross = cov @ qnull
    ranking = cross.square() / cov.diagonal().clamp_min(1e-12)
    ranking[blocked] = -torch.inf
    order = ranking.argsort(descending=True)
    trials, matrices, rows = [], [], []
    for count in (4, 8, 16, 32, 64):
        idx = order[:count]
        R = cov[idx][:, idx].clone()
        R.diagonal().add_(1e-6 * R.diagonal().mean())
        fit = torch.linalg.solve(R, cross[idx])
        quiet = qnull.clone()
        quiet[idx] -= fit
        quiet /= (SX[:, 8] @ quiet).mean()
        Qnew = Q + coefficient[None, :, None] * meanQ[:, :, None] * (qnull - quiet)
        row = {'predictor_count': count, 'predictor_coordinates': idx.tolist(),
               'word_marker_rms': float((T[nt:] @ quiet).square().mean().sqrt()),
               'maximum_format_marker_change': float((SX[:, 8] @ (quiet - qnull)).abs().max()),
               **metrics(Qnew)}
        trials.append(row)
        matrices.append(Qnew.float())
        rows.append(quiet)
        print(json.dumps({'layer': layer, 'sparse_trial': row}), flush=True)
    report = {'layer': layer, 'decomposition': decomposition, 'sparse_predictor_trials': [4, 8, 16, 32, 64],
              'ridge_fraction': 1e-6, 'weight_decomposition_max_error': reconstruction_error,
              'sparse_marker_word_rms': float((T[nt:] @ qnull).square().mean().sqrt()),
              'full_quiet_marker_word_rms': float((T[nt:] @ qfull_normalized).square().mean().sqrt()),
              'trials': trials, 'capture_names': names, 'word_capture': word_name,
              'query_readout_artifact': quiet_name, 'reader_circuit': reader_name,
              'no_model_execution_or_answer_targets': True}
    reports[str(layer)] = report
    saved[str(layer)] = {'query_matrices': matrices, 'quiet_markers': rows, 'native_coefficients': coefficient,
                         'format_query_mean': meanQ, 'old_sparse_marker': qnull, 'program_query': program,
                         'selected_channels': selected, 'report': report}
    print(json.dumps({'layer': layer, 'decomposition': decomposition}), flush=True)
    del T, SX, caps, word, Hraw, Hnorm, K, SK, KR
atomic_torch(ROOT / 'artifacts/a40_query_cancellation_v1.pt', saved)
atomic_json(ROOT / 'artifacts/a40_query_cancellation_v1.json', reports)
