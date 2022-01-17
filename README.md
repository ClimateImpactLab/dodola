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

From the command line, you replicate one of the downscaling workflow's validation steps with: 

```shell
dodola validate-dataset <InsertZarrStoreURLhere> \
  --variable "tasmax" \
  --data-type "downscaled" \
  -t historical
```

Alternatively, the service used by this command can be called directly from Python in `dodola.services`.

## Installation

`dodola` is generally run from within a container. But, you can install the application and access the command-line interface or Python API with `pip`:

```shell
pip install git+https://github.com/ClimateImpactLab/dodola
```

the dependencies used in the container are in `./environment.yaml`.

## Support

Source code is available online at https://github.com/ClimateImpactLab/dodola. This software is Open Source and available under the Apache License, Version 2.0.
