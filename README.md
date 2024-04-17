Benchmarks
===========

Set environment variable `DPNP_RAISE_EXCEPION_ON_NUMPY_FALLBACK` to 0.

## Setting up environment

todo: non-conda


## Running

Create `results.json`
```sh
    pytest .\pytest_benchmark --benchmark-json=results.json
```

Create `results.csv` comparing test results
```sh
    pytest-benchmark compare results.json --csv=results.csv --group-by='name'
```

To create graphs after creating the `results.json`, run:
```sh
    python graphs.py
```
To modify the tests being shown modify the `tests` list at the top of the `graphs.py` file.
