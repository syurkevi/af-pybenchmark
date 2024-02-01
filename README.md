Benchmarks
===========

Set environment variable `DPNP_RAISE_EXCEPION_ON_NUMPY_FALLBACK` to 0.

## Setting up environment

```sh
    conda env create -f environment.yml
    conda activate af-benchmark
```

## Running

Create `results.json`
```sh
    pytest .\pytest_benchmark --benchmark-json=results.json
```

Create `results.csv` comparing test results
```sh
    pytest-benchmark compare results.json --csv=results.csv --group-by='name'
```

Create graphs:

WIP
