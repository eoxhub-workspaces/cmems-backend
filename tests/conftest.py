import numpy as np
import pandas as pd
import pytest
import xarray as xr
from fastapi.testclient import TestClient
from osgeo import gdal, gdal_array

from cmems.app import app


@pytest.fixture()
def client():
    return TestClient(app)


@pytest.fixture()
def raster_data_file_path(tmp_path) -> str:
    # bivariate distribution
    # from: https://matplotlib.org/stable/gallery/images_contours_and_fields/image_demo.html
    delta = 0.025
    x = y = np.arange(-3.0, 3.0, delta)
    X, Y = np.meshgrid(x, y)
    Z1 = np.exp(-(X**2) - Y**2)
    Z2 = np.exp(-((X - 1) ** 2) - (Y - 1) ** 2)
    Z = (Z1 - Z2) * 2 * 500
    ds = gdal_array.OpenNumPyArray(Z, False)

    file_path = str(tmp_path / "test.tif")
    driver: gdal.Driver = gdal.GetDriverByName("GTiff")
    driver.CreateCopy(file_path, ds)
    return file_path


@pytest.fixture()
def dataset() -> xr.Dataset:
    precipitation = np.arange(40, 58, dtype=float).reshape((3, 3, 2))
    temperature = np.arange(20, 38, dtype=float).reshape((3, 3, 2))
    time = pd.date_range("2020-01-01", periods=2)
    return xr.Dataset(
        {
            "Irrigation": (["lat", "lon", "time"], precipitation),
            "Temperature": (["lat", "lon", "time"], temperature),
        },
        coords={
            "time": time,
            "lat": np.arange(60, 63),
            "lon": np.arange(70, 73),
        },
    )


@pytest.fixture()
def dataarray() -> xr.DataArray:
    irrigation = np.arange(40, 49, dtype=float).reshape((3, 3))
    da = xr.DataArray(
        irrigation,
        coords={
            "lat": np.arange(60, 63),
            "lon": np.arange(70, 73),
        },
    )
    da.rio.write_crs("EPSG:4326", inplace=True)
    da.rio.set_spatial_dims(
        x_dim="lat",
        y_dim="lon",
        inplace=True,
    )
    return da
