[![DOI](https://zenodo.org/badge/314387532.svg)](https://zenodo.org/badge/latestdoi/314387532)
[![Test](https://github.com/ClimateImpactLab/dodola/actions/workflows/test.yaml/badge.svg)](https://github.com/ClimateImpactLab/dodola/actions/workflows/test.yaml)
[![Build](https://github.com/ClimateImpactLab/dodola/actions/workflows/buildpush.yaml/badge.svg)](https://github.com/ClimateImpactLab/dodola/actions/workflows/buildpush.yaml)
[![codecov](https://codecov.io/gh/ClimateImpactLab/dodola/branch/main/graph/badge.svg?token=WCDUAU8KFT)](https://codecov.io/gh/ClimateImpactLab/dodola)

# dodola

Containerized application for running the steps in a larger, orchestrated CMIP6 bias-correction and downscaling workflow.

This is under heavy development.

## Features

Commands can be run through the command line with `dodola <command>`.

```
Commands:
    adjust-maximum-precipitation  Adjust maximum precipitation in a dataset
    apply-dtr-floor               Apply a floor to diurnal temperature...
    apply-non-polar-dtr-ceiling   Apply a ceiling to diurnal temperature...
    apply-qdm                     Adjust simulation year with quantile...
    apply-qplad                   Adjust (downscale) simulation year with...
    cleancmip6                    Clean up and standardize GCM
    correct-wetday-frequency      Correct wet day frequency in a dataset
    get-attrs                     Get attrs from data
    prime-qdm-output-zarrstore    Prime a Zarr Store for regionally-written...
    prime-qplad-output-zarrstore  Prime a Zarr Store for regionally-written...
    rechunk                       Rechunk Zarr store in memory.
    regrid                        Spatially regrid a Zarr Store in memory
    removeleapdays                Remove leap days and update calendar
    train-qdm                     Train quantile delta mapping (QDM)
    train-qplad                   Train Quantile-Preserving, Localized...
    validate-dataset              Validate a CMIP6, bias corrected or...
```

See `dodola --help` or `dodola <command> --help` for more information.

## Example

From the command line, run one of the downscaling workflow's validation steps with: 

```shell
dodola validate-dataset "gs://your/climate/data.zarr" \
  --variable "tasmax" \
  --data-type "downscaled" \
  -t "historical"
```

The service used by this command can be called directly from a Python session or script

```python
import dodola.services

dodola.services.validate(
    "gs://your/climate/data.zarr", 
    "tasmax",
    data_type="downscaled",
    time_period="historical",
)
```

## Installation

`dodola` is generally run from within a container. `dodola` container images are currently hosted at [ghcr.io/climateimpactlab/dodola](https://ghcr.io/climateimpactlab/dodola).

Alternatively, you can install a bleeding-edge version of the application and access the command-line interface or Python API with `pip`:

```shell
pip install git+https://github.com/ClimateImpactLab/dodola
```

Because there are many compiled dependencies we recommend installing `dodola` and its dependencies within a `conda` virtual environment. Dependencies used in the container to create its `conda` environment are in `./environment.yaml`.

## Support

Source code is available online at https://github.com/ClimateImpactLab/dodola. This software is Open Source and available under the Apache License, Version 2.0.
