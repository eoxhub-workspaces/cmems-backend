import tempfile
from unittest import mock

import pytest
import xarray as xr


@pytest.fixture()
def mock_copernicusmarine_subset(dataset: xr.Dataset):
    with tempfile.TemporaryDirectory() as temp_dir:
        path = f"{temp_dir}/le_file.nc"
        dataset.to_netcdf(path)
        with mock.patch(
            "cmems.app.copernicusmarine.subset", return_value=str(path)
        ) as mocker:
            yield mocker


def test_cmems(client, dataset, mock_copernicusmarine_subset):
    with mock.patch("cmems.app.rioxarray.open_rasterio", return_value=dataset):
        response = client.get(
            "/cmems",
            params={},
        )
    response_data = response.json()

    entry = response_data["Irrigation"][0]
    assert entry["min"] == 40.0
    assert entry["datetime"] == "2020-01-01T00:00:00.000000000"
