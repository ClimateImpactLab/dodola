# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- Add badge for current release DOI to README. (@brews)

## [0.18.0] - 2022-03-03
### Added
- Add basic CI/CD test and build status badges to README. (PR #182, @brews)
### Fixed
- Fix dodola validate-dataset OOM on small workers without dask-distributed. (PR #181, @brews)

## [0.17.0] - 2022-02-17
### Changed
- Increase max allowed tasmin, tasmax in services.validate to 377 K for UKESM1-0-LL. (PR #180, @brews)
### Fixed
- Move in-memory data loading where it is needed for 360-days calendar conversion in clean-cmip6 (PR #179, @emileten)

## [0.16.2] - 2022-02-15
### Fixed
- Fix incorrect references when standardizing gcm (PR #178, @emileten)

## [0.16.1] - 2022-01-27
### Fixed
- Fix the wetday frequency correction so that different replacement values are used, rather than a single one (PR #174, PR #176, @emileten, @delgadom).

## [0.16.0] - 2022-01-19
### Added
- Improve README.md. (PR #169, @brews)
### Changed
- Remove duplicated service-level logging info lines of code introduced by PR #148 (PR #168, @emileten)
- Decrease validation temperature range min to 130 (PR #170, @emileten)

## [0.15.1] - 2021-12-29
### Fixed
- Fix boolean condition error in DTR ceiling application (PR #165, PR #166, @emileten)

## [0.15.0] - 2021-12-27
### Added
- Add maximum precipitation adjustment service that applies a "ceiling" or "cap" to precipitation values above a user-defined threshold. (PR #164, @dgergel)

### Changed
- Increase max precipitation allowed by validation to 3000 mm. (PR #164, @dgergel)
- Update wet day frequency correction to incorporate method additions from Hempel et al 2013. (PRs #162 and #159, @dgergel)
- Floor and ceiling for DTR. (PR #163 @emileten)

## [0.14.0] - 2021-12-21
### Changed
- Update wet day frequency correction to include small negative values in correction and to limit the correction range to the threshold * 10 ^ -2. (PR #158, @dgergel)
- Update package setup, README, HISTORY/CHANGELOG to new system. (PR #154, @brews)

## [0.13.0] - 2021-12-17
### Changed
- Update diurnal temperature range (DTR) validation to differentiate polar and non-polar regions. (PR #153, @dgergel)
- Update diurnal temperature range (DTR) validation to differentiate min DTR accepted value for CMIP6 vs bias corrected and downscaled data inputs (PR #155, @dgergel)

### Removed
- Remove cruft code. Remove `dodola` commands `biascorrect`, `downscale`, `buildweights` along with corresponding functions in `dodola.services` and `dodola.core`.  (PR #152, @brews)

### Fixed
- Fix rechunk error when converting 360 days calendars. (#149, PR #151, @brews)

## [0.12.0] - 2021-12-09
### Added
- Add 360 days calendar support (PR #144, @emileten)
- Add an option to temporarily replace the target variable units in dodola services and use in CLI dodola for precip (PR #143, @emileten)
- Add diurnal temperature range (DTR) correction for small DTR values below 1 (converts them to 1) (PR #145, @dgergel)

## [0.11.1] - 2021-12-03
### Changed
- Decrease allowed timesteps for bias corrected/downscaled files in validation to allow models that only go through 2099 (PR #146, @dgergel)

## [0.11.0] - 2021-11-30
### Added
- Add post wet day correction option in CLI dodola (PR #141 @emileten)

### Changed
- Increase validation temperature range max to 360 (PR #142, @dgergel)
- Distinguish missing from excess timesteps in timesteps validation (PR #140, @emileten)

## [0.10.0] - 2021-11-22
### Added
- Add additional tests for `dodola.core.*_analogdownscaling` functions. (PR #136, @dgergel, @brews)

### Changed
- Update dtr range check max to allow up to 70 C. (PR #138, @brews, @dgergel)


## [0.9.0] - 2021-11-15
### Added
* Add `--root-attrs-json-file` to `prime-qplad-output-zarrstore`, `apply-qplad`, `prime-qdm-output-zarrstore`, `apply-qdm`. (PR #134, @brews)
* Add `dodola get-attrs` command. (PR #133, @brews)

### Changed
* Upgrade Docker base image to ``continuumio/miniconda3:4.10.3``. (PR #132, @brews)

### Fixed
* Fix attrs missing from services.apply_qplad output Datasets. (#135, @brews)


## [0.8.0] - 2021-11-10
### Added
* Add AIQPD output Zarr priming (``prime-aipqd-output-zarrstore``), input slicing, region writing, attrs merging, and multi-year processing. This breaks backwards compatibility for ``apply-aiqpd`` and its services and core functions. See the pull request for additional details. (PR #130, @brews)
* Similarly, add QDM output Zarr priming (``prime-qdm-output-zarrstore``), region writing, attrs merging, and multi-year processing. This breaks backwards compatibility for ``apply-qdm`` and its services and core functions. See the pull request for additional details. (PR #129, @brews)
* Add pre-training slicing options to ``train-qdm`` and ``train-aiqpd``. (PR #123, PR #128, @brews)

### Changed
* AIQPD has been renamed "Quantile-Preserving, Localized Analogs Downscaling" (QPLAD). AIQPD-named commands have been switch to QPLAD. This is backward compatibility breaking. (PR #131, @brews)
* Make logging slightly more chatty by default. (PR #129, @brews)

### Fixed
* Quick fix validation reading entire zarr store for check. (PR #124, @brews)


## [0.7.0] - 2021-11-02
### Added
* Add global validation, includes new service ``validate`` for validating cleaned CMIP6, bias corrected and downscaled data for historical and future time periods. (PR #118, @dgergel)

### Changed
* Update xclim version to 0.30.1, this updates the Train/Adjust API for QDM and AIQPD and requires units attributes for all QDM and AIQPD inputs. (PR #119, @dgergel)
* Regrid copies input Dataset ``attrs`` metadata to output (#116). (PR #121, @brews)

### Security
* Upgrade ``dask`` to 2021.10.0 to cover https://nvd.nist.gov/vuln/detail/CVE-2021-42343. (PR #122, @brews)


## [0.6.0] - 2021-09-08
### Added
* Add AIQPD downscaling method to options. Also updates ``xclim`` dependency to use the CIL-fork and "@add_analog_downscaling" branch, with 0.28.1 of ``xclim`` merged in. This supersedes the BCSD downscaling service. (PR #98, PR #115, @dgergel)


## [0.5.0] - 2021-08-04
### Added
* Add ``--cyclic`` option to regrid cli and services. (PR #108, @brews)
* Add ``papermill``, ``intake-esm`` to Docker environment. (PR #106, @brews)

### Changed
* Bump environment ``xarray`` to v0.19.0. (PR #109, @brews)


## [0.4.1] - 2021-07-13
### Changed
* Bump xclim to v0.28.0, improve environment notes. (PR #105, @brews)

### Fixed
* Fix application logging to stdout. (PR #104, @brews)


## [0.4.0] - 2021-07-09
### Added
* Add ``include-quantiles`` flag to ``apply_qdm`` to allow for including quantile information in bias corrected output. (PR #95, @dgergel)
* Add precipitation unit conversion to ``standardize_gcm``. (PR #94, @dgergel)
* Add ``astype`` argument to ``regrid``. (PR #92, @brews)

### Changed
* Make ``dodola`` container's default CMD. (PR #90, @brews)
* Improve subprocess and death handling in Docker container. (PR #90, @brews)

### Fixed
* Fix bug in train_quantiledeltamapping accounting for endpoints. (#87, @brews)


## [0.3.0] - 2021-06-16
### Added
- Update `buildweights` service to add support for regridding to domain file. Not backwards compatible. (PR #67, @dgergel)
- Add downscaling service. Currently support BCSD spatial disaggregation as implemented in scikit-downscale. (PR #65, @dgergel)
- Add bias-correction quantile delta mapping (QDM) components to support Argo Workflows. New commands added: ``dodola train-qdm`` and ``dodola apply-qdm``. (PR #70, @brews)
- Add wet day frequency correction service. Wet day frequency implemented as described in Cannon et al., 2015. New command added: ``dodola correct-wetday-frequency``. (PR #78, @dgergel)

### Changed
- Significant updates to container environment: Python 3.9, ``xarray``, ``adlfs``, ``xesmf``, ``dask``, and ``fsspec``. (PR #74, PR #75, PR #76, PR #77, PR #84 @brews)
- Remove stdout buffering from container runs, add IO debug logging. (PR #72, @brews)

### Fixed
- Fix CMIP6 clean to better handle coords vs dims. (PR #81, @brews)


## [0.2.0] - 2021-04-23
### Changed
- Switch to pure `fsspec`-style URLs for data inputs. Added support for GCS buckets and S3 storage. Switch to `fsspec` backend settings to collect storage authentication. Because of this users likely will need to change the environment variables used to pass in storage credentials. `dodola.services` no longer require the `storage` argument. [PR #61](https://github.com/ClimateImpactLab/dodola/pull/61) from [@brews](https://github.com/brews).
- Switch to simple `xarray`-based rechunking to workaround to instability from our use of `rechunker`. This change breaks the CLI for `dodola rechunk`, removing the `-v/--variable` and `-m/--maxmemory` options. The change also breaks the `dodola.services.rechunk()` signature, removing the `max_mem` argument and the `target_chunks` argument is now a mapping `{coordinate_name: chunk_size}`. [PR #60](https://github.com/ClimateImpactLab/dodola/pull/60) from [@brews](https://github.com/brews).

### Fixed
- Fix `TypeError` from `dodola rechunk`. [PR #63](https://github.com/ClimateImpactLab/dodola/pull/63) from [@brews](https://github.com/brews).

## [0.1.0] - 2021-04-15
- Initial release.
