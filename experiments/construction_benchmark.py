from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import scipy
from scipy import sparse


def base_problem(epsilon=0.2, angle=0.37):
    c, s = np.cos(angle), np.sin(angle)
    U = np.array([[c, -s], [s, c]])
    B = [np.diag([1.0, -1.0]),
         np.array([[epsilon, 1.0], [1.0, 1.0]]),
         np.array([[epsilon, -1.0], [-1.0, 1.0]])]
    rotated = [U @ b @ U.T for b in B]
    X = U @ np.diag([2 / (1 + epsilon), (1 - epsilon) / (1 + epsilon)]) @ U.T
    v = U @ np.array([np.sqrt(1 / epsilon), 0.0])
    return rotated, X, v


def sparse_measurement(X, v, m):
    if m < 1:
        raise ValueError('The repetition count must be positive.')
    ne = m - m % 2
    r, s, t = float(X[0, 0]), float(X[0, 1]), float(X[1, 1])
    if r <= 0 or r * t < s * s:
        raise ValueError('The benchmark requires a positive leading moment entry and nonnegative determinant.')
    q, a = np.sqrt(r * t - s * s), np.sqrt(r)
    index = np.arange(ne, dtype=np.int32)
    rows = np.concatenate([index, index, index])
    cols = np.concatenate([index, m + index, m + (index ^ 1)])
    values = np.concatenate([np.full(ne, a), np.full(ne, s / a),
                             np.where(index % 2 == 0, -q / a, q / a)])
    if m % 2:
        rows = np.concatenate([rows, np.array([ne, ne], dtype=np.int32)])
        cols = np.concatenate([cols, np.array([ne, m + ne], dtype=np.int32)])
        values = np.concatenate([values, np.asarray(v, dtype=float)])
    R = sparse.coo_matrix((values, (rows, cols)), shape=(m, 2 * m)).tocsr()
    R.eliminate_zeros()
    return R


def certify_blocks(B, X, v):
    r, s, t = float(X[0, 0]), float(X[0, 1]), float(X[1, 1])
    q = np.sqrt(r * t - s * s)
    J = np.array([[0.0, 1.0], [-1.0, 0.0]])
    W = np.vstack([np.sqrt(r) * np.eye(2), (s * np.eye(2) + q * J) / np.sqrt(r)])
    residuals = [np.linalg.eigvalsh(W.T @ np.kron(b, np.eye(2)) @ W - np.eye(2))[0]
                 for b in B]
    residuals.extend(float(v @ b @ v - 1) for b in B)
    gram_error = np.linalg.norm(W.T @ W - np.trace(X) * np.eye(2), ord=2)
    return float(min(residuals)), float(gram_error)


def timed_batches(function, batches=7, target_seconds=0.02):
    function()
    repetitions = 1
    while True:
        start = time.perf_counter()
        for _ in range(repetitions):
            function()
        elapsed = time.perf_counter() - start
        if elapsed >= target_seconds or repetitions >= 65536:
            break
        repetitions *= 2
    samples = []
    for _ in range(batches):
        start = time.perf_counter()
        for _ in range(repetitions):
            function()
        samples.append((time.perf_counter() - start) / repetitions)
    return dict(median_seconds=float(np.median(samples)),
                q25_seconds=float(np.quantile(samples, 0.25)),
                q75_seconds=float(np.quantile(samples, 0.75)),
                repetitions_per_batch=repetitions, batch_seconds_per_call=samples)


def run(output):
    output.mkdir(parents=True, exist_ok=True)
    B, X, v = base_problem()
    alpha, kappa = float(v @ v), float(np.trace(X))
    residual, gram_error = certify_blocks(B, X, v)
    assert residual >= -1e-12 and gram_error <= 1e-12
    rng = np.random.default_rng(20261101)
    records, raw = [], []
    for m in [16, 17, 256, 257, 4096, 4097, 65536, 65537]:
        R = sparse_measurement(X, v, m)
        x = rng.standard_normal(2 * m)
        y = R @ x
        ne = m - m % 2
        r, s, t = X[0, 0], X[0, 1], X[1, 1]
        q = np.sqrt(r * t - s * s)
        expected = np.zeros(m)
        for j in range(0, ne, 2):
            expected[j] = np.sqrt(r) * x[j] + (s * x[m + j] - q * x[m + j + 1]) / np.sqrt(r)
            expected[j + 1] = np.sqrt(r) * x[j + 1] + (s * x[m + j + 1] + q * x[m + j]) / np.sqrt(r)
        if m % 2:
            expected[-1] = v[0] * x[m - 1] + v[1] * x[2 * m - 1]
        apply_error = float(np.max(np.abs(y - expected)))
        target = ne * kappa + (m % 2) * alpha
        trace_error = abs(float(R.data @ R.data) - target) / target
        assert R.nnz == 3 * m - m % 2
        assert apply_error <= 1e-12 and trace_error <= 1e-12
        if m <= 17:
            W = R.toarray().T
            assert min(np.linalg.eigvalsh(W.T @ np.kron(b, np.eye(m)) @ W - np.eye(m))[0] for b in B) >= -1e-12
            assert np.allclose(np.linalg.eigvalsh(W.T @ W), sorted([kappa] * ne + [alpha] * (m % 2)))
        construction = timed_batches(lambda: sparse_measurement(X, v, m))
        application = timed_batches(lambda: R @ x)
        record = dict(m=m, nnz=int(R.nnz),
                      csr_bytes=int(R.data.nbytes + R.indices.nbytes + R.indptr.nbytes),
                      dense_bytes=16 * m * m, construction_seconds=construction['median_seconds'],
                      construction_q25=construction['q25_seconds'], construction_q75=construction['q75_seconds'],
                      apply_seconds=application['median_seconds'], apply_q25=application['q25_seconds'],
                      apply_q75=application['q75_seconds'], relative_trace_error=trace_error,
                      max_apply_error=apply_error, minimum_block_qmi_eigenvalue=residual,
                      pair_gram_error=gram_error)
        records.append(record)
        raw.append(dict(m=m, construction=construction, application=application))
    with (output / 'construction_benchmark.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    metadata = dict(python=sys.version, platform=platform.platform(), processor=platform.processor(),
                    logical_cpus=os.cpu_count(), numpy=np.__version__, scipy=scipy.__version__,
                    seed=20261101, epsilon=0.2, angle_radians=0.37, moment=X.tolist(), scalar=v.tolist(),
                    csr_value_dtype=str(R.data.dtype), csr_index_dtype=str(R.indices.dtype),
                    timing='Seven warmed batches; median and quartiles of mean seconds per call in each batch.',
                    excluded='Base-problem solution, validation, and input-vector generation.',
                    constraint_validation='Two-dimensional and four-dimensional blocks; full checks also at m=16,17.',
                    raw_timings=raw)
    (output / 'construction_environment.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(cases=len(records), max_m=max(r['m'] for r in records),
                         max_relative_trace_error=max(r['relative_trace_error'] for r in records),
                         max_apply_error=max(r['max_apply_error'] for r in records),
                         minimum_block_qmi_eigenvalue=residual, pair_gram_error=gram_error,
                         largest=records[-1]), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    base = Path(__file__).resolve().parents[1]
    parser.add_argument('--output-dir', type=Path, default=base / 'generated' / 'data')
    output = parser.parse_args().output_dir.resolve()
    if output == (base / 'data').resolve():
        parser.error('Choose an output directory separate from the frozen data.')
    run(output)
