# HypoRS: A Statistical Hypothesis Testing Library

`hypors` is a Rust library designed for performing a variety of hypothesis tests, including t-tests, z-tests, proportion tests, ANOVA, Chi-square tests, and Mann-Whitney tests. This library leverages the `statrs` crate for statistical distributions.

Rust Crate: https://crates.io/crates/hypors

PyPI Package: Work in Progress

## Features

### Hypothesis Tests

Hypothesis testing is available for this suite of common distributions.

- **T-Tests**: One-sample, two-sample paired, and two-sample independent t-tests.
- **Z-Tests**: One-sample, two-sample paired, and two-sample independent z-tests.
- **Proportion Tests**: One-sample and two-sample proportion tests.
- **ANOVA**: One-way ANOVA for comparing means across multiple groups.
- **Chi-Square Tests**: Chi-square test for independence and goodness-of-fit tests.
- **Mann-Whitney U Test**: Non-parametric test for comparing two independent samples.

### Sample Size Calculation

All parametrized distributions have respective modules to calculate minimum sample size required with customizable parameters for alpha and statistical power.


### **Additional Features**:
  - Customizable tail type (left, right, and two-tailed).
  - Customizable alpha value for all tests.
  - Confidence interval calculations for all tests.
  - p-value is generated along with each statistic.
  - Null and alternate hypotheses strings are also generated.

## Installation

To use this library in your Rust project, add the following to your `Cargo.toml`:

```toml
[dependencies]
hypors = "0.3.0"
```

**Note** HypoRS relies on the following dependencies, which will be automatically included:

```
serde (version >=1.0.210)
statrs (version >=0.17.1)
```

## Example Usage

### Rust

Here are some examples of running tests with Rust.

#### T - Test

```rust
use hypors::{t::t_test, common::TailType};

let data = vec![1.2, 2.3, 1.9, 2.5, 2.8];
let population_mean = 2.0;
let tail = TailType::Two;
let alpha = 0.05;

let result = t_test(data, population_mean, tail, alpha).unwrap();
println!("Test Statistic: {}", result.test_statistic);
println!("p-value: {}", result.p_value);
println!("Confidence Interval: {}", result.confidence_interval);
println!("Null Hypothesis: {}", result.null_hypothesis);
println!("Alternate Hypothesis: {}", result.alt_hypothesis);
println!("Reject Null Hypothesis?: {}", result.reject_null);
```

#### Chi Square Test

```rust
use hypors::{chi_square::independence};

let observed = vec![10, 20, 30]; // Observed frequencies
let expected = vec![15, 15, 30]; // Expected frequencies

let result = independece(observed, expected).unwrap();
println!("Chi-Square Statistic: {}", result.test_statistic);
println!("p-value: {}", result.p_value);
println!("Null Hypothesis: {}", result.null_hypothesis);
println!("Alternate Hypothesis: {}", result.alt_hypothesis);
println!("Reject Null Hypothesis?: {}", result.reject_null);
```

### Usage with Polars

With the `polars` crate having frequent minor version updates, this library has been updated to not use Polars inherently, and relies on `Vec` types.

Any use of `polars` `Series` types should be converted before and after using the `hypors` functions.

**Polars to Vec Conversion:**
```rust
use polars::prelude::*;
let series = Series::new("series_name", &[1, 2, 3, 4, 5]);
let vec = series.to_vec().unwrap();
```

**Vec to Polars Conversion:**
```rust
use polars::prelude::*;
let vec = vec![1, 2, 3, 4, 5];
let series = Series::new("series_name", vec);
```


### Python

Work in Progress

## Future Plans

The next step for `hypors` is to add Python bindings to make it accessible to the Python community. This work is currently in progress. Stay tuned for updates!

## Contributing
Contributions are always welcome! If you have suggestions for improvements, bug fixes, or new features, please feel free to open an issue or submit a pull request.

## License
This project is licensed under the MIT License.