from unittest import mock

import pytest


@pytest.fixture()
def mock_data_load(dataset):
    with (
        mock.patch("cmems.app.pystac_client"),
        mock.patch("cmems.app.odc.stac.load", return_value=dataset),
    ):
        yield


def test_landing_page_loads(client):
    response = client.get("/")
    assert response.json() == {}


@pytest.mark.integration
def test_stats_returns_data_with_actual_server(client):
    # NOTE: this test accesses a live server on the internet
    response = client.get(
        "/stats",
        params={
            "geometry": """{"type": "Polygon", "coordinates": [
          [
            [-2, 41],
            [-1.85, 41],
            [-1.85, 41.1],
            [-2, 41.1],
            [-2, 41]
          ]
         ]}""",
        },
    )
    response_data = response.json()

    assert {"max", "min", "stddev", "mean"} <= set(response_data["Irrigation"].keys())
    assert (
        response_data["Irrigation"]["max"]["2021-10-08T00:00:00.000000000"]
        == 11.935233116149902
    )


def test_stats_returns_data(client, mock_data_load):
    response = client.get(
        "/stats",
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
        },
    )
    response_data = response.json()

    assert {"max", "min", "stddev", "mean"} <= set(response_data["Irrigation"].keys())
    assert response_data["Irrigation"]["max"]["2020-01-01T00:00:00.000000000"] == 56


def test_stats_allows_selecting_polygon(client, mock_data_load):
    response = client.get(
        "/stats",
        params={
            "geometry": """{"type": "Polygon", "coordinates": [
          [
            [59, 70],
            [61, 70],
            [61, 72],
            [59, 70]
          ]
         ]}""",
        },
    )
    response_data = response.json()

    assert {"max", "min", "stddev", "mean"} <= set(response_data["Irrigation"].keys())
    assert response_data["Irrigation"]["max"]["2020-01-01T00:00:00.000000000"] == 50
