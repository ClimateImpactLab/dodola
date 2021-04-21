=======
History
=======


0.X.X (XXXX-XX-XX)
------------------
* Fix ``TypeError`` from `dodola rechunk`. (PR#63, @brews)
* Switch to pure ``fsspec``-style URLs for data inputs. Added support for GCS buckets and S3 storage. Switch to ``fsspec`` backend settings to collect storage authentication. Because of this users likely will need to change the environment variables used to pass in storage credentials. ``dodola.services`` no longer require the ``storage`` argument. (PR#61, @brews)
* Switch to simple ``xarray``-based rechunking to workaround to instability from our use of ``rechunker``. This change breaks the CLI for ``dodola rechunk``, removing the ``-v/--variable`` and ``-m/--maxmemory`` options. The change also breaks the ``dodola.services.rechunk()`` signature, removing the ``max_mem`` argument and the ``target_chunks`` argument is now a mapping ``{coordinate_name: chunk_size}``. (PR#60, @brews)


0.1.0 (2021-04-15)
------------------
* Initial release.
