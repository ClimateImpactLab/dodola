=======
History
=======


0.X.X (XXXX-XX-XX)
------------------
* Switch to simple ``xarray``-based rechunking to workaround to instability from our use of ``rechunker``. This change breaks the CLI for ``dodola rechunk``, removing the ``-v/--variable`` and ``-m/--maxmemory`` options. The change also breaks the ``dodola.services.rechunk()`` signature, removing the ``max_mem`` argument and the ``target_chunks`` argument is now a mapping ``{coordinate_name: chunk_size}``. (PR#60, @brews)


0.1.0 (2021-04-15)
------------------
* Initial release.
