=======
History
=======


0.X.X (XXXX-XX-XX)
------------------
* Significant updates to container environment: Python 3.9, ``xarray``, ``adlfs``, ``xesmf``, ``dask``, and ``fsspec``. (PR #74, PR #75, PR #76, PR #77, PR #84 @brews)
* Update buildweights service to add support for regridding to domain file. Not backwards compatible. (PR #67, @dgergel)
* Add downscaling service. Currently support BCSD spatial disaggregation as implemented in scikit-downscale. (PR #65, @dgergel)
* Remove stdout buffering from container runs, add IO debug logging. (PR #72, @brews)
* Add bias-correction quantile delta mapping (QDM) components to support Argo Workflows. New commands added: ``dodola train-qdm`` and ``dodola apply-qdm``. (PR #70, @brews)
* Fix CMIP6 clean to better handle coords vs dims. (PR #81, @brews)
* Add wet day frequency correction service. Wet day frequency implemented as described in Cannon et al., 2015. New command added: ``dodola correct-wetday-frequency``. (PR #78, @dgergel)

0.2.0 (2021-04-23)
------------------
* Fix ``TypeError`` from `dodola rechunk`. (PR #63, @brews)
* Switch to pure ``fsspec``-style URLs for data inputs. Added support for GCS buckets and S3 storage. Switch to ``fsspec`` backend settings to collect storage authentication. Because of this users likely will need to change the environment variables used to pass in storage credentials. ``dodola.services`` no longer require the ``storage`` argument. (PR #61, @brews)
* Switch to simple ``xarray``-based rechunking to workaround to instability from our use of ``rechunker``. This change breaks the CLI for ``dodola rechunk``, removing the ``-v/--variable`` and ``-m/--maxmemory`` options. The change also breaks the ``dodola.services.rechunk()`` signature, removing the ``max_mem`` argument and the ``target_chunks`` argument is now a mapping ``{coordinate_name: chunk_size}``. (PR #60, @brews)


0.1.0 (2021-04-15)
------------------
* Initial release.
