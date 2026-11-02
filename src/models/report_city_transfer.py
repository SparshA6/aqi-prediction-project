"""Summarize corrected leave-one-city-out runs without retraining or pooled-city bias."""
from pathlib import Path
from hashlib import sha256
import argparse
import json
import pandas as pd
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    root, out = args.root, args.out
    paths = {
        'sequence': root/'results/attention/runs/corrected-v2-gpu-attention-loco/metrics_loco.csv',
        'spatial': root/'results/spatial/runs/corrected-v2-gpu-spatial-loco/metrics_loco.csv',
    }
    frames = []
    for family, path in paths.items():
        d = pd.read_csv(path)
        d = d.loc[(d.protocol == 'loco') & (d.split == 'test') &
                  (d.feature_set == 'SWT-w7') & (d.subset == 'all') &
                  (d.level == 'overall')].copy()
        d['family'] = family
        d['city'] = d.fold.str.removeprefix('loco:')
        assert d.city.nunique() == 8 and not d.duplicated(['model', 'city']).any()
        assert d.groupby('model').city.nunique().eq(8).all()
        frames.append(d[['family', 'city', 'model', 'n', 'MAE', 'RMSE', 'bias']])
    rows = pd.concat(frames, ignore_index=True)
    summaries = []
    for (family, model), g in rows.groupby(['family', 'model']):
        summaries.append(dict(family=family, model=model, cities=len(g), station_days=int(g.n.sum()),
                              city_macro_MAE=g.MAE.mean(), city_macro_RMSE=g.RMSE.mean(),
                              pooled_RMSE=np.sqrt(np.average(g.RMSE.pow(2), weights=g.n)),
                              city_macro_bias=g.bias.mean()))
    summary = pd.DataFrame(summaries).sort_values(['family', 'city_macro_RMSE'])
    out.mkdir(parents=True, exist_ok=True)
    rows.to_csv(out/'city_folds.csv', index=False)
    summary.to_csv(out/'model_summary.csv', index=False)
    (out/'sources.json').write_text(json.dumps({str(p.relative_to(root)): sha256(p.read_bytes()).hexdigest()
                                                 for p in paths.values()}, indent=2), encoding='utf-8')
    lines = ['# Corrected held-out-city evaluation', '',
             'Each fold trains without one city and evaluates that city. SWT-w7, all eligible test rows. '
             'City macro averages give each of the eight cities equal weight; pooled RMSE weights all station-days. '
             'These are historical spatial-transfer results, not prospective October–December 2026 scores.', '',
             '| Family | Model | City-macro RMSE | City-macro MAE | Pooled RMSE |',
             '|---|---|---:|---:|---:|']
    for r in summary.itertuples():
        lines.append(f'| {r.family} | {r.model} | {r.city_macro_RMSE:.2f} | '
                     f'{r.city_macro_MAE:.2f} | {r.pooled_RMSE:.2f} |')
    seq = rows[rows.family == 'sequence'].pivot(index='city', columns='model', values='RMSE')
    spa = rows[rows.family == 'spatial'].pivot(index='city', columns='model', values='RMSE')
    seq_win = int((seq['TA'] < seq['Uniform']).sum())
    spa_win = int((spa['S-TA+SA'] < spa['S-Uniform']).sum())
    lines += ['', f'Sequence TA beats Uniform in {seq_win}/8 held-out cities; its city-macro RMSE is '
              f'{seq["TA"].mean():.2f} versus {seq["Uniform"].mean():.2f}. '
              f'Spatial S-TA+SA beats S-Uniform in {spa_win}/8 cities; its city-macro RMSE is '
              f'{spa["S-TA+SA"].mean():.2f} versus {spa["S-Uniform"].mean():.2f}. '
              'The remaining cities show that spatial transfer varies substantially by location.', '']
    lines += ['', 'Attention variants should be compared with Uniform within the same family. '
              'A lower pooled score alone does not prove city-balanced transfer or explanation faithfulness. '
              'These runs do not audit attention faithfulness on held-out-city checkpoints.', '']
    (out/'README.md').write_text('\n'.join(lines), encoding='utf-8')
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
