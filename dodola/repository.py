"""Abstract Repository objects to access stored climate model data.

These are abstractions to isolate most of the data layer.
"""

import abc
from adlfs import AzureBlobFileSystem
from xarray import open_zarr


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


class AzureZarr(RepositoryABC):
    """Azure storage for Zarr data repository.

    To authenticate with storage, must initialize with `account_name` and
    `account_key` for key-based authentication or `client_id`,
    `client_secret`, and `tenant_id` for authentication with service principal
    credentials. Initializing arguments are passed to
    ``adlfs.AzureBlobFileSystem``.

    Parameters
    ----------
    account_name : str or None, optional
    account_key : str or None, optional
    client_id : str or None, optional
    client_secret : str or None, optional
    tenant_id : str or None, optional
    """

    def __init__(
        self,
        account_name=None,
        account_key=None,
        client_id=None,
        client_secret=None,
        tenant_id=None,
    ):
        self.fs = AzureBlobFileSystem(
            account_name=account_name,
            account_key=account_key,
            client_id=client_id,
            client_secret=client_secret,
            tenant_id=tenant_id,
        )

    def read(self, url_or_path):
        """Read Dataset from Zarr file in storage

        Parameters
        ----------
        url_or_path : str
            URL of Zarr data in Azure storage.

        Returns
        -------
        xr.Dataset
        """
        return open_zarr(self.fs.get_mapper(url_or_path))

    def write(self, url_or_path, x):
        """Write Dataset to Zarr file in storage

        This opens Zarr storage with mode "w" and is called with with
        ``compute=True``, so any lazy computations will be completed.

        Parameters
        ----------
        url_or_path : str
            URL of Zarr data to write to.
        x : xr.Dataset
        """
        x.to_zarr(self.fs.get_mapper(url_or_path), mode="w", compute=True)


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
