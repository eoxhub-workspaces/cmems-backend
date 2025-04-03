import math
from unittest import mock


def test_ch4_returns_some_data(client, dataarray):
    with mock.patch("cmems.app.rioxarray.open_rasterio", return_value=dataarray):
        response = client.get(
            "/ch4-stats",
            params={
                "geometry": """{"type": "Polygon", "coordinates": [
          [
            [60, 70],
            [62, 70],
            [62, 72],
            [60, 72],
            [60, 70]
          ]
         ]}""",
                "date_start": "2024-01-01",
                "date_end": "2024-03-01",
            },
        )
    response_data = response.json()

    assert response_data[0]["max"] == 47.0
    assert response_data[0]["date"] == "2024-01-15"


def test_ch4_handles_nan_values(client, dataarray):
    dataarray[:] = math.nan
    with mock.patch("cmems.app.rioxarray.open_rasterio", return_value=dataarray):
        response = client.get(
            "/ch4-stats",
            params={
                "geometry": """{"type": "Polygon", "coordinates": [
          [
            [60, 70],
            [62, 70],
            [62, 72],
            [60, 72],
            [60, 70]
          ]
         ]}""",
                "date_start": "2024-01-01",
                "date_end": "2024-03-01",
            },
        )
    response_data = response.json()

    assert response_data[0]["max"] is None
