#!/usr/bin/env python3
"""
Standardized cross-method evaluation for the Ziram severity comparison.

Runs the SAME protocol on any representation (RegionProps features, ShapeEmbed latents,
or CNN penultimate features): 5-fold stratified CV with LogisticRegression + RandomForest,
reporting both the 5-class severity task and the binary (SC0 vs any-kink) task, with
macro-F1, weighted-F1 and (quadratic-weighted) Cohen's kappa.

Inputs (one of):
  --latents X.npy --labels y.npy        # e.g. ShapeEmbed's test_latent_space.npy / test_labels.npy
  --csv feats.csv --label-col COL        # a features table with a label column

Usage:
  python standard_eval.py --latents results/.../test_latent_space.npy \
                          --labels  results/.../test_labels.npy --name "ShapeEmbed ls128 b0.05"
  python standard_eval.py --csv regionprops_maskF0_labelled.csv --label-col severity_score_adjusted --name RegionProps
"""

import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import f1_score, accuracy_score, cohen_kappa_score, confusion_matrix
import os

CLFS = [('LogReg', LogisticRegression(max_iter=3000)),
        ('RF', RandomForestClassifier(n_estimators=300, random_state=0))]


def report(X, y, name):
    """Run the protocol; print and RETURN (metric rows, confusion matrices)."""
    X = StandardScaler().fit_transform(np.asarray(X, dtype=float))
    y = np.asarray(y).astype(int)
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    n, dims = len(y), X.shape[1]
    rows, cms = [], {}
    print(f"\n=== {name}  (n={n}, dims={dims}, classes={np.bincount(y).tolist()}) ===")
    print("  5-class (severity SC0-4):")
    for nm, clf in CLFS:
        yp = cross_val_predict(clf, X, y, cv=cv)
        r = dict(method=name, task='5class', classifier=nm, n=n, dims=dims,
                 acc=accuracy_score(y, yp), macroF1=f1_score(y, yp, average='macro'),
                 weightedF1=f1_score(y, yp, average='weighted'),
                 quadKappa=cohen_kappa_score(y, yp, weights='quadratic'))
        rows.append(r); cms[f'5class_{nm}'] = confusion_matrix(y, yp)
        print(f"    {nm:7s} acc={r['acc']:.3f} macroF1={r['macroF1']:.3f} "
              f"weightedF1={r['weightedF1']:.3f} quadKappa={r['quadKappa']:.3f}")
    yb = (y > 0).astype(int)
    print("  binary (SC0 vs any-kink):")
    for nm, clf in CLFS:
        yp = cross_val_predict(clf, X, yb, cv=cv)
        r = dict(method=name, task='binary', classifier=nm, n=n, dims=dims,
                 acc=accuracy_score(yb, yp), macroF1=f1_score(yb, yp, average='macro'),
                 weightedF1=f1_score(yb, yp, average='weighted'),
                 quadKappa=cohen_kappa_score(yb, yp))   # binary: unweighted kappa
        rows.append(r); cms[f'binary_{nm}'] = confusion_matrix(yb, yp)
        print(f"    {nm:7s} acc={r['acc']:.3f} F1(affected)={f1_score(yb,yp):.3f} "
              f"macroF1={r['macroF1']:.3f} kappa={r['quadKappa']:.3f}")
    return rows, cms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--latents'); ap.add_argument('--labels')
    ap.add_argument('--csv'); ap.add_argument('--label-col')
    ap.add_argument('--drop-cols', nargs='*', default=['mask', 'stem', 'CO6', 'set'])
    ap.add_argument('--name', default='representation')
    ap.add_argument('--out', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results'),
                    help='dir for saved metrics (default: ./results next to this script)')
    args = ap.parse_args()

    if args.latents and args.labels:
        X = np.load(args.latents); y = np.load(args.labels)
    elif args.csv and args.label_col:
        df = pd.read_csv(args.csv)
        y = df[args.label_col].values
        X = df.drop(columns=[c for c in args.drop_cols + [args.label_col] if c in df.columns]) \
              .select_dtypes('number').fillna(0).values
    else:
        ap.error('provide --latents/--labels or --csv/--label-col')

    rows, cms = report(X, y, args.name)

    # --- save: per-run metrics, confusion matrices, and append to a master table ---
    os.makedirs(args.out, exist_ok=True)
    safe = ''.join(c if c.isalnum() or c in '-_' else '_' for c in args.name)
    run_df = pd.DataFrame(rows)
    run_df.to_csv(os.path.join(args.out, f'metrics_{safe}.csv'), index=False)
    for k, cm in cms.items():
        np.savetxt(os.path.join(args.out, f'confusion_{safe}_{k}.csv'), cm, fmt='%d', delimiter=',')
    master = os.path.join(args.out, 'all_methods_comparison.csv')
    prev = pd.read_csv(master) if os.path.exists(master) else pd.DataFrame()
    if len(prev):   # re-running a name overwrites its old rows
        key = ['method', 'task', 'classifier']
        prev = prev[~prev[key].apply(tuple, 1).isin(run_df[key].apply(tuple, 1))]
    pd.concat([prev, run_df], ignore_index=True).to_csv(master, index=False)
    print(f"\n[OK] saved metrics_{safe}.csv (+ confusion matrices) -> {args.out}/")
    print(f"[OK] master comparison table -> {master}")


if __name__ == '__main__':
    main()
