# CMake/C++ reference implementation

This project provides a small C++17 reference implementation for the solvers described in the paper:

- **Unit-norm constrained least squares** (vector case):
  \[
    \min_{\|u\|=1}\ \|Bu-g\|_2^2
  \]
  via the KKT/secular equation in the Lagrange multiplier (a polynomial after clearing denominators),
  implemented with a robust bracketing solver that also handles the classical “hard case”
  when the KKT matrix becomes singular.

- **Square orthogonal/unitary Procrustes** (matrix case):
  \[
    \min_{X^\top X = I}\ \|AX-B\|_F^2
  \]
  via SVD / polar-factor representatives, plus a small-$n$ polynomial multi-solution representative.

The implementation uses **Eigen** for linear algebra.

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

If Eigen is not installed, CMake will try to fetch it (requires internet). To disable fetching:

```bash
cmake -S . -B build -DGUC_LS_FETCH_EIGEN=OFF
```

## Run examples

```bash
./build/example_unitnorm
./build/example_procrustes
./build/example_stiefel
```

## Noise-sweep benchmark (CSV for MATLAB boxplots)

Build the benchmark executable:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Run a Monte Carlo noise sweep and write a CSV:

```bash
./build/benchmark_noise_sweep \
  --problem unitnorm,procrustes \
  --mc 200 \
  --noise 0,0.002,0.005,0.01,0.02 \
  --out build/guc_ls_noise_sweep.csv
```

By default, the benchmark runs these requested sizes:

- Vector: `unitnorm3`, `unitnorm50`, `unitnorm200`
- Matrix: `procrustes4`, `procrustes30`, `procrustes100`

You can override sizes:

```bash
./build/benchmark_noise_sweep --unit_n 3,10,100 --proc_n 4,20
```

You can also select a single family:

```bash
./build/benchmark_noise_sweep --problem unitnorm
./build/benchmark_noise_sweep --problem procrustes
```

The CSV has columns:

```
problem,solver,noise,trial,rot_err_deg,trans_err,orth_err
```

- For `unitnorm*`, `rot_err_deg` is the angle error between the true and estimated unit vector (deg), and `trans_err` is the residual \|Bu-g\|^2.
- For `procrustes*`, `rot_err_deg` is the geodesic distance on SO(n) in degrees, and `trans_err` is the residual \|AX-B\|_F^2.
- `orth_err` measures constraint satisfaction: \(|u^\top u-1|\) for the vector case and \(\|X^\top X-I\|_F\) for the matrix case.

## MATLAB plotting

Open and run:

```matlab
matlab/guc_ls_boxplot.m
```

This script reads `../build/guc_ls_noise_sweep.csv` by default and draws grouped boxplots versus noise level, for every `problem` present in the CSV.

## Notes

- The underlying constraint equation can be written as a degree-\(2n\) polynomial after clearing denominators.
  For numerical robustness (especially near singular configurations), this repo solves the equivalent
  monotone secular equation using bracketing + bisection, and explicitly handles the singular KKT “hard case”.
- The Stiefel-Newton solver in this repo is intended for **small \(\ell\)**. In the benchmark it is only used for the `procrustes4` case.

This code is meant to be clear and hackable rather than fully optimized.
