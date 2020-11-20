"""Abstract Repository objects to access stored climate model data.

These are abstractions to isolate most of the data layer.
"""

import abc


class RepositoryABC(abc.ABC):
    """ABC for a basic Repository pattern"""

    @abc.abstractmethod
    def read(self, url_or_path):
        """Read xr.Dataset from storage

        Parameters
        ----------
        url_or_path : str
            Location (e.g. URL or path) of data to read from storage.

        Returns
        -------
        xr.Dataset
        """

    @abc.abstractmethod
    def write(self, url_or_path, x):
        """Write xr.Dataset, x, to storage

        Parameters
        ----------
        url_or_path : str
            Location (e.g. URL or path) of data to read from storage.
        x : xr.Dataset
            Data to store in repository.
        """


class AzureRepository(RepositoryABC):
    """Azure datalake and blob-stored data repository.

    Uses ``adlfs``, I guess...?

    """

    def read(self, url_or_path):
        raise NotImplementedError

    def write(self, url_or_path, x):
        raise NotImplementedError


class GcsRepository(RepositoryABC):
    """Google Cloud Storage bucket-based repository.

    Prob will use ``gcsfs``.
    """

    def read(self, url_or_path):
        raise NotImplementedError

    def write(self, url_or_path, x):
        raise NotImplementedError


class FakeRepository(RepositoryABC):
    """Simple in-memory repository for testing

    This simply puts a Repository interface over a dict.

    Parameters
    ----------
    storage : mapping or None, optional
        Optional internal repository state.
    """

    def __init__(self, storage=None):
        self.storage = {}
        if storage:
            self.storage = dict(storage)

    def read(self, url_or_path):
        """Read url_or_path key from repository"""
        return self.storage[url_or_path]

    def write(self, url_or_path, x):
        """Write data x to repository at key url_or_path"""
        self.storage[url_or_path] = x
