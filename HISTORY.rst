=======
History
=======


0.X.X (XXXX-XX-XX)
------------------
* Significant updates to container environment: Python 3.9, ``xarray``, ``adlfs``, ``xesmf``, and ``fsspec``. (PR #74, PR #75, PR #76, PR #77, @brews)
* Update buildweights service to add support for regridding to domain file. Not backwards compatible. (PR #67, @dgergel)
* Add downscaling service. Currently support BCSD spatial disaggregation as implemented in scikit-downscale. (PR #65, @dgergel)
* Remove stdout buffering from container runs, add IO debug logging. (PR #72, @brews)

0.2.0 (2021-04-23)
------------------
* Fix ``TypeError`` from `dodola rechunk`. (PR #63, @brews)
* Switch to pure ``fsspec``-style URLs for data inputs. Added support for GCS buckets and S3 storage. Switch to ``fsspec`` backend settings to collect storage authentication. Because of this users likely will need to change the environment variables used to pass in storage credentials. ``dodola.services`` no longer require the ``storage`` argument. (PR #61, @brews)
* Switch to simple ``xarray``-based rechunking to workaround to instability from our use of ``rechunker``. This change breaks the CLI for ``dodola rechunk``, removing the ``-v/--variable`` and ``-m/--maxmemory`` options. The change also breaks the ``dodola.services.rechunk()`` signature, removing the ``max_mem`` argument and the ``target_chunks`` argument is now a mapping ``{coordinate_name: chunk_size}``. (PR #60, @brews)


0.1.0 (2021-04-15)
------------------
* Initial release.
