# hypors

Hypothesis testing for Python, backed by the
[`hypors`](https://crates.io/crates/hypors) Rust crate.

Every statistic is computed by the Rust library — these bindings only convert
values across the boundary, so Python and Rust always agree.

## Install

```bash
pip install hypors
```

## Use

```python
from hypors import TailType
from hypors.t import t_test

result = t_test([1.2, 2.3, 1.9, 2.5, 2.8], pop_mean=2.0, tail=TailType.Two, alpha=0.05)

print(result.test_statistic)
print(result.p_value)
print(result.reject_null)
print(result.to_dict())
```

Any iterable of numbers works — a `list`, `tuple`, generator, `numpy` array,
`pandas.Series` or `polars.Series`. The package itself depends on none of
them:

```python
import numpy as np
import polars as pl

t_test(np.array([1.2, 2.3, 1.9]), 2.0, TailType.Two, 0.05)
t_test(pl.Series([1.2, 2.3, 1.9]), 2.0, TailType.Two, 0.05)
```

## Tests available

| Module | Functions |
| --- | --- |
| `hypors.t` | `t_test`, `t_test_paired`, `t_test_ind`, `t_sample_size` |
| `hypors.z` | `z_test`, `z_test_paired`, `z_test_ind`, `z_sample_size` |
| `hypors.proportion` | `z_test`, `z_test_ind`, `prop_sample_size` |
| `hypors.anova` | `anova`, `f_sample_size` |
| `hypors.chi_square` | `independence`, `goodness_of_fit`, `variance`, `chi2_sample_size_gof`, `chi2_sample_size_ind`, `chi2_sample_size_variance` |
| `hypors.mann_whitney` | `u_test` |
| `hypors.common` | `TailType`, `TestResult` |

`TailType` and `TestResult` are also importable from the top level.

## Results

Every test returns a `TestResult` with `test_statistic`, `p_value`,
`confidence_interval`, `null_hypothesis`, `alt_hypothesis` and `reject_null`,
plus `to_dict()` for serialising. Attributes are read-only.

Errors surface as ordinary Python exceptions: `ValueError` for empty or
insufficient data, `RuntimeError` for a computation that could not be
completed.

## Notes

- The Mann-Whitney U p-value matches `scipy.stats.mannwhitneyu` with
  `method="asymptotic"`. Its `test_statistic` is `min(U1, U2)` while the
  p-value derives from `U1`, so compare p-values rather than statistics when
  checking against scipy.
- Versions track the Rust crate: `hypors` on PyPI and `hypors` on crates.io
  share a version number.

## License

MIT
