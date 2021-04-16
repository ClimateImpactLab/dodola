=======
History
=======


0.X.X (XXXX-XX-XX)
------------------
* Switch to pure ``fsspec``-style URLs for data inputs. Added support for GCS buckets and S3 storage. Switch to ``fsspec`` backend settings to collect storage authentication. Because of this users likely will need to change the environment variables used to pass in storage credentials. ``dodola.services`` no longer require the ``storage`` argument. (PR#61, @brews)


0.1.0 (2021-04-15)
------------------
* Initial release.
