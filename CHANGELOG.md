# Changelog

All notable changes to this project are documented in this file. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

While this crate is pre-1.0, the **minor** version carries breaking changes:
a dependency on `0.3` will not resolve to `0.4`.

## [0.4.0] - 2026-09-07

### Changed

- **Mann-Whitney U p-values now match `scipy.stats.mannwhitneyu`
  (`method="asymptotic"`).** Existing p-values change, and one-sided results
  change substantially. Thanks to @guilherme-n-l (#7, reported in #6).
  - The variance carries the tie correction
    `n1*n2/12 * ((N + 1) - sum(t^3 - t) / (N * (N - 1)))`. The previous
    no-ties variance overestimated sigma and inflated p-values on tied data.
  - A 0.5 continuity correction is applied toward the tail, matching scipy's
    default `use_continuity=True`.
  - One-sided p-values derive from `U1` rather than `min(U1, U2)`. The old
    statistic was sign-blind, so `TailType::Right` and `TailType::Left`
    returned identical p-values regardless of which group dominated.
  - When every observation is tied the variance is zero: the two-sided
    p-value is now `NaN` and the one-sided p-values are `1.0`, matching
    scipy 1.18's signed continuity correction. Previously this returned
    `1.0` two-sided and `0.5` one-sided.
- `test_statistic` is documented explicitly as `min(U1, U2)` while the
  p-value derives from `U1`, so compare p-values rather than statistics
  when checking against scipy.
- **`mann_whitney::u_test` and `chi_square::variance` now return
  `Result<TestResult, StatError>`** instead of `Result<TestResult, String>`,
  matching every other test in the crate. Code matching on the error needs
  updating; code using `?` or `unwrap()` does not. Empty input now yields
  `StatError::EmptyData`, too few observations `StatError::InsufficientData`,
  and distribution failures `StatError::ComputeError`.

### Added

- `rust-version = "1.85"`, the minimum supported Rust version for
  edition 2024.
- GitHub Actions CI running `cargo fmt --check`,
  `cargo clippy --all-targets -- -D warnings`, `cargo test` (doc tests
  included) and `cargo package` on every pull request (#8, #9).

### Fixed

- The manifest is named `Cargo.toml` rather than `cargo.toml`, which only
  resolved on case-insensitive filesystems and broke every cargo command on
  Linux (#8).

### Internal

- The published crate no longer ships the Python bindings (`hypopy`),
  `py_tests`, or the CI configuration — 54 files down to 41 (#8).

## [0.3.0] - 2025-04-19

See the [release history](https://github.com/astronights/hypors/releases) for
versions before this changelog was introduced.
