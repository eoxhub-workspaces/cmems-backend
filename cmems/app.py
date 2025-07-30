import concurrent.futures
import datetime
import functools
import logging
import math
import re
import tempfile
import time
from typing import Annotated, Any, cast

import copernicusmarine
import numpy as np
import odc.stac
import pydantic
import pystac_client
import rioxarray
import shapely
import structlog
import xarray as xr
from asgi_correlation_id import CorrelationIdMiddleware
from fastapi import FastAPI, Query, Request
from starlette_exporter import PrometheusMiddleware, handle_metrics

logging.basicConfig(level=logging.INFO)

logger = structlog.getLogger()

odc.stac.configure_rio(
    cloud_defaults=True,
    verbose=True,
)

# from cmems.setup_logging import RequestIdLoggingMiddleware, setup_logging

# setup_logging()

app = FastAPI()

# app.add_middleware(RequestIdLoggingMiddleware)
app.add_middleware(PrometheusMiddleware)
app.add_middleware(CorrelationIdMiddleware)
app.add_route("/metrics", handle_metrics)


COMMON_TIME_DIMENSION_NAMES = ["time", "t", "date", "datetime", "time_index"]

COMMON_FIRST_SPACE_DIMENSION_NAMES = ["x", "lat", "latitude"]
COMMON_SECOND_SPACE_DIMENSION_NAMES = ["y", "lon", "longitude"]


@app.get("/")
async def landing_page(request: Request):
    return {}


@app.get("/healthz")
def healthz():
    return {"message": "All is OK!"}


def _process_ch4(file: str, geometry_parsed) -> dict:
    logger.info("start processing file", file=file)
    start = time.time()
    da = cast(
        xr.DataArray,
        rioxarray.open_rasterio(f"s3://dev-byoa-1/vs-test-data/S5-CH4-weekly/{file}"),
    )
    da = da.rio.clip([geometry_parsed], drop=True)
    da = da.where(da != -9999, other=np.nan)
    # TODO: test gdal native solution with statistics call
    result = {
        "min": da.min().values.tolist(),
        "max": da.max().values.tolist(),
        "mean": da.mean().values.tolist(),
        "stddev": da.std().values.tolist(),
    }
    logger.info("finished processing file", file=file, duration=time.time() - start)
    return result


def nan_to_none(a: float) -> float | None:
    return a if not math.isnan(a) else None


FloatNoNan = Annotated[float | None, pydantic.BeforeValidator(nan_to_none)]


class Ch4ResponseItem(pydantic.BaseModel):
    date: datetime.date
    min: FloatNoNan
    max: FloatNoNan
    mean: FloatNoNan
    stddev: FloatNoNan


@app.get("/ch4-stats/")
def ch4_stats(
    geometry: str,
    date_start: datetime.date,
    date_end: datetime.date,
) -> list[Ch4ResponseItem]:
    geometry_parsed = shapely.from_geojson(geometry)

    files = [(date, f) for date, f in ALL_FILES.items() if date_start <= date <= date_end]

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        results = executor.map(
            functools.partial(_process_ch4, geometry_parsed=geometry_parsed),
            [e[1] for e in files],
        )
    return [Ch4ResponseItem(**r, date=e[0]) for e, r in zip(files, results, strict=False)]


@app.get("/cmems")
def cmems(
    dataset_id: str = "cmems_mod_glo_phy-cur_anfc_0.083deg_PT6H-i",
    variables: Annotated[list[str], Query()] = ["vo"],  # noqa: B006
    start_datetime: datetime.datetime = datetime.datetime(2023, 1, 1, 12),
    end_datetime: datetime.datetime = datetime.datetime(2023, 1, 10, 12),
    minimum_longitude: float = -25.5,
    maximum_longitude: float = -25,
    minimum_latitude: float = 17,
    maximum_latitude: float = 17.5,
    minimum_depth: float = 0.5,
    maximum_depth: float = 0,
) -> dict:
    logger.debug(f"Handling cmems request for {dataset_id} {variables}")
    logger.debug(f"Requested date range: {start_datetime} {end_datetime}")
    with tempfile.TemporaryDirectory() as temp_dir:
        result_path = copernicusmarine.subset(
            dataset_id=dataset_id,
            variables=variables,
            start_datetime=start_datetime,
            end_datetime=end_datetime,
            minimum_longitude=minimum_longitude,
            maximum_longitude=maximum_longitude,
            minimum_latitude=minimum_latitude,
            maximum_latitude=maximum_latitude,
            minimum_depth=minimum_depth,
            maximum_depth=maximum_depth,
            service="timeseries",
            disable_progress_bar=True,
            dryRun=False, # make sure download is executed
            output_directory=temp_dir,
        )

        logger.debug(f"downloaded to {result_path}")
        ds = xr.open_dataset(result_path)
        logger.debug(f"opened ds {ds}")
        return statistics(ds, dict_per_time=True)


@app.get("/stats/")
def stats(geometry: str):
    # TODO: limit as parameter with max as fastapi thingy
    # TODO: datetime parameters
    # TODO errror handling as fastapi thingy
    geometry_parsed = shapely.from_geojson(geometry)
    datetime_query = ("2021-01-01T00:00:00Z", "2021-10-20T00:00:00Z")
    logger.debug("Handling stats request")
    # *
    # * load data with odc.load_stac or similar to openeo notebook:
    #     https://github.com/Open-EO/openeo-processes-dask/blob/main/openeo_processes_dask/process_implementations/cubes/load.py#L135
    #     from s3://eox-gitlab-testdata/vs/daily_average_cog
    #      https://stac.eurac.edu/collections/Irrigation_Ebro_basin
    # * add basic parameters as needed, mostly bbox and time range?
    # * output basic statistics

    catalog = pystac_client.Client.open("https://stac.eurac.edu/")
    query = catalog.search(
        max_items=40,
        intersects=geometry_parsed,
        datetime=datetime_query,
        collections=["Irrigation_Ebro_basin"],
    )

    ds = odc.stac.load(
        query.items(),
        bands=["Irrigation"],
        geopolygon=geometry_parsed,
        crs="EPSG:4326",
    )

    # NOTE: this is not necessary if the source data already conveys this
    #       information somehow
    ds.rio.set_spatial_dims(
        x_dim=_matching_dim(ds, COMMON_FIRST_SPACE_DIMENSION_NAMES),
        y_dim=_matching_dim(ds, COMMON_SECOND_SPACE_DIMENSION_NAMES),
        inplace=True,
    )
    ds.rio.write_crs("EPSG:4326", inplace=True)

    clipped = ds.rio.clip([geometry_parsed], all_touched=True)

    return statistics(clipped)


def statistics(ds: xr.Dataset, dict_per_time=False) -> dict:
    time_index = _matching_dim(ds, COMMON_TIME_DIMENSION_NAMES)
    non_time_indexes = [dim for dim in ds.dims if dim != time_index]

    def _serialize_da_with_date(da: xr.DataArray) -> dict:
        return dict(
            zip(
                np.datetime_as_string(da[time_index].values),
                da.values.tolist(),
                strict=True,
            )
        )

    def _stats_for_dataarray(da: xr.DataArray) -> dict | list:
        if time_index:
            data = {
                "min": _serialize_da_with_date(da.min(dim=non_time_indexes)),
                "max": _serialize_da_with_date(da.max(dim=non_time_indexes)),
                "mean": _serialize_da_with_date(da.mean(dim=non_time_indexes)),
                "stddev": _serialize_da_with_date(da.std(dim=non_time_indexes)),
            }
            if dict_per_time:
                return [
                    {
                        "datetime": datetime_key,
                        "min": data["min"][datetime_key],
                        "max": data["max"][datetime_key],
                        "mean": data["mean"][datetime_key],
                        "stddev": data["stddev"][datetime_key],
                    }
                    for datetime_key in data["min"]
                ]
            else:
                return data

        else:
            return {
                "min": da.min().values.tolist(),
                "max": da.max().values.tolist(),
                "mean": da.mean().values.tolist(),
                "stddev": da.std().values.tolist(),
            }

    return {key: _stats_for_dataarray(da) for key, da in ds.items()}


def _matching_dim(ds: xr.Dataset, names: list[str]) -> str | None:
    return next(
        (cast(str, dim) for dim in ds.dims if dim in names),
        None,
    )


if __name__ == "__main__":
    results = stats(
        geometry="""{"type": "Polygon", "coordinates": [
          [
            [-2, 41],
            [-1.85, 41],
            [-1.85, 41.1],
            [-2, 41.1],
            [-2, 41]
          ]
         ]}""",
    )
    from pprint import pprint

    pprint(results)


@app.middleware("http")
async def log_middle(request: Request, call_next):
    start_time = time.time()

    response = await call_next(request)

    ignored_paths = ["/healthz", "/metrics"]
    if request.url.path not in ignored_paths:
        # NOTE: swagger validation failures prevent log_start_time from running
        duration = time.time() - start_time
        logger.info(
            "Request finished",
            method=request.method,
            url=str(request.url),
            duration_ms=duration * 1000,
            content_length=response.headers.get("content-length"),
            status=int(response.status_code),
        )

    return response


ALL_FILES = {
    datetime.datetime.strptime(
        cast(Any, re.search(r"week-(\d+)", file)).group(1), "%Y%m%d"
    ).date(): file
    for file in [
        "s5p-l3grd-ch4-001-week-20180430_cog.tif",
        "s5p-l3grd-ch4-001-week-20180507_cog.tif",
        "s5p-l3grd-ch4-001-week-20180514_cog.tif",
        "s5p-l3grd-ch4-001-week-20180521_cog.tif",
        "s5p-l3grd-ch4-001-week-20180528_cog.tif",
        "s5p-l3grd-ch4-001-week-20180604_cog.tif",
        "s5p-l3grd-ch4-001-week-20180611_cog.tif",
        "s5p-l3grd-ch4-001-week-20180618_cog.tif",
        "s5p-l3grd-ch4-001-week-20180625_cog.tif",
        "s5p-l3grd-ch4-001-week-20180702_cog.tif",
        "s5p-l3grd-ch4-001-week-20180709_cog.tif",
        "s5p-l3grd-ch4-001-week-20180716_cog.tif",
        "s5p-l3grd-ch4-001-week-20180723_cog.tif",
        "s5p-l3grd-ch4-001-week-20180730_cog.tif",
        "s5p-l3grd-ch4-001-week-20180806_cog.tif",
        "s5p-l3grd-ch4-001-week-20180813_cog.tif",
        "s5p-l3grd-ch4-001-week-20180820_cog.tif",
        "s5p-l3grd-ch4-001-week-20180827_cog.tif",
        "s5p-l3grd-ch4-001-week-20180903_cog.tif",
        "s5p-l3grd-ch4-001-week-20180910_cog.tif",
        "s5p-l3grd-ch4-001-week-20180917_cog.tif",
        "s5p-l3grd-ch4-001-week-20180924_cog.tif",
        "s5p-l3grd-ch4-001-week-20181001_cog.tif",
        "s5p-l3grd-ch4-001-week-20181008_cog.tif",
        "s5p-l3grd-ch4-001-week-20181015_cog.tif",
        "s5p-l3grd-ch4-001-week-20181022_cog.tif",
        "s5p-l3grd-ch4-001-week-20181029_cog.tif",
        "s5p-l3grd-ch4-001-week-20181105_cog.tif",
        "s5p-l3grd-ch4-001-week-20181112_cog.tif",
        "s5p-l3grd-ch4-001-week-20181119_cog.tif",
        "s5p-l3grd-ch4-001-week-20181126_cog.tif",
        "s5p-l3grd-ch4-001-week-20181203_cog.tif",
        "s5p-l3grd-ch4-001-week-20181210_cog.tif",
        "s5p-l3grd-ch4-001-week-20181217_cog.tif",
        "s5p-l3grd-ch4-001-week-20181224_cog.tif",
        "s5p-l3grd-ch4-001-week-20181231_cog.tif",
        "s5p-l3grd-ch4-001-week-20190107_cog.tif",
        "s5p-l3grd-ch4-001-week-20190114_cog.tif",
        "s5p-l3grd-ch4-001-week-20190121_cog.tif",
        "s5p-l3grd-ch4-001-week-20190128_cog.tif",
        "s5p-l3grd-ch4-001-week-20190204_cog.tif",
        "s5p-l3grd-ch4-001-week-20190211_cog.tif",
        "s5p-l3grd-ch4-001-week-20190218_cog.tif",
        "s5p-l3grd-ch4-001-week-20190225_cog.tif",
        "s5p-l3grd-ch4-001-week-20190304_cog.tif",
        "s5p-l3grd-ch4-001-week-20190311_cog.tif",
        "s5p-l3grd-ch4-001-week-20190318_cog.tif",
        "s5p-l3grd-ch4-001-week-20190325_cog.tif",
        "s5p-l3grd-ch4-001-week-20190401_cog.tif",
        "s5p-l3grd-ch4-001-week-20190408_cog.tif",
        "s5p-l3grd-ch4-001-week-20190415_cog.tif",
        "s5p-l3grd-ch4-001-week-20190422_cog.tif",
        "s5p-l3grd-ch4-001-week-20190429_cog.tif",
        "s5p-l3grd-ch4-001-week-20190506_cog.tif",
        "s5p-l3grd-ch4-001-week-20190513_cog.tif",
        "s5p-l3grd-ch4-001-week-20190520_cog.tif",
        "s5p-l3grd-ch4-001-week-20190527_cog.tif",
        "s5p-l3grd-ch4-001-week-20190603_cog.tif",
        "s5p-l3grd-ch4-001-week-20190610_cog.tif",
        "s5p-l3grd-ch4-001-week-20190617_cog.tif",
        "s5p-l3grd-ch4-001-week-20190624_cog.tif",
        "s5p-l3grd-ch4-001-week-20190701_cog.tif",
        "s5p-l3grd-ch4-001-week-20190708_cog.tif",
        "s5p-l3grd-ch4-001-week-20190715_cog.tif",
        "s5p-l3grd-ch4-001-week-20190722_cog.tif",
        "s5p-l3grd-ch4-001-week-20190729_cog.tif",
        "s5p-l3grd-ch4-001-week-20190805_cog.tif",
        "s5p-l3grd-ch4-001-week-20190812_cog.tif",
        "s5p-l3grd-ch4-001-week-20190819_cog.tif",
        "s5p-l3grd-ch4-001-week-20190826_cog.tif",
        "s5p-l3grd-ch4-001-week-20190902_cog.tif",
        "s5p-l3grd-ch4-001-week-20190909_cog.tif",
        "s5p-l3grd-ch4-001-week-20190916_cog.tif",
        "s5p-l3grd-ch4-001-week-20190923_cog.tif",
        "s5p-l3grd-ch4-001-week-20190930_cog.tif",
        "s5p-l3grd-ch4-001-week-20191007_cog.tif",
        "s5p-l3grd-ch4-001-week-20191014_cog.tif",
        "s5p-l3grd-ch4-001-week-20191021_cog.tif",
        "s5p-l3grd-ch4-001-week-20191028_cog.tif",
        "s5p-l3grd-ch4-001-week-20191104_cog.tif",
        "s5p-l3grd-ch4-001-week-20191111_cog.tif",
        "s5p-l3grd-ch4-001-week-20191118_cog.tif",
        "s5p-l3grd-ch4-001-week-20191125_cog.tif",
        "s5p-l3grd-ch4-001-week-20191202_cog.tif",
        "s5p-l3grd-ch4-001-week-20191209_cog.tif",
        "s5p-l3grd-ch4-001-week-20191216_cog.tif",
        "s5p-l3grd-ch4-001-week-20191223_cog.tif",
        "s5p-l3grd-ch4-001-week-20191230_cog.tif",
        "s5p-l3grd-ch4-001-week-20200106_cog.tif",
        "s5p-l3grd-ch4-001-week-20200113_cog.tif",
        "s5p-l3grd-ch4-001-week-20200120_cog.tif",
        "s5p-l3grd-ch4-001-week-20200127_cog.tif",
        "s5p-l3grd-ch4-001-week-20200203_cog.tif",
        "s5p-l3grd-ch4-001-week-20200210_cog.tif",
        "s5p-l3grd-ch4-001-week-20200217_cog.tif",
        "s5p-l3grd-ch4-001-week-20200224_cog.tif",
        "s5p-l3grd-ch4-001-week-20200302_cog.tif",
        "s5p-l3grd-ch4-001-week-20200309_cog.tif",
        "s5p-l3grd-ch4-001-week-20200316_cog.tif",
        "s5p-l3grd-ch4-001-week-20200323_cog.tif",
        "s5p-l3grd-ch4-001-week-20200330_cog.tif",
        "s5p-l3grd-ch4-001-week-20200406_cog.tif",
        "s5p-l3grd-ch4-001-week-20200413_cog.tif",
        "s5p-l3grd-ch4-001-week-20200420_cog.tif",
        "s5p-l3grd-ch4-001-week-20200427_cog.tif",
        "s5p-l3grd-ch4-001-week-20200504_cog.tif",
        "s5p-l3grd-ch4-001-week-20200511_cog.tif",
        "s5p-l3grd-ch4-001-week-20200518_cog.tif",
        "s5p-l3grd-ch4-001-week-20200525_cog.tif",
        "s5p-l3grd-ch4-001-week-20200601_cog.tif",
        "s5p-l3grd-ch4-001-week-20200608_cog.tif",
        "s5p-l3grd-ch4-001-week-20200615_cog.tif",
        "s5p-l3grd-ch4-001-week-20200622_cog.tif",
        "s5p-l3grd-ch4-001-week-20200629_cog.tif",
        "s5p-l3grd-ch4-001-week-20200706_cog.tif",
        "s5p-l3grd-ch4-001-week-20200713_cog.tif",
        "s5p-l3grd-ch4-001-week-20200720_cog.tif",
        "s5p-l3grd-ch4-001-week-20200727_cog.tif",
        "s5p-l3grd-ch4-001-week-20200803_cog.tif",
        "s5p-l3grd-ch4-001-week-20200810_cog.tif",
        "s5p-l3grd-ch4-001-week-20200817_cog.tif",
        "s5p-l3grd-ch4-001-week-20200824_cog.tif",
        "s5p-l3grd-ch4-001-week-20200831_cog.tif",
        "s5p-l3grd-ch4-001-week-20200907_cog.tif",
        "s5p-l3grd-ch4-001-week-20200914_cog.tif",
        "s5p-l3grd-ch4-001-week-20200921_cog.tif",
        "s5p-l3grd-ch4-001-week-20200928_cog.tif",
        "s5p-l3grd-ch4-001-week-20201005_cog.tif",
        "s5p-l3grd-ch4-001-week-20201012_cog.tif",
        "s5p-l3grd-ch4-001-week-20201019_cog.tif",
        "s5p-l3grd-ch4-001-week-20201026_cog.tif",
        "s5p-l3grd-ch4-001-week-20201102_cog.tif",
        "s5p-l3grd-ch4-001-week-20201109_cog.tif",
        "s5p-l3grd-ch4-001-week-20201116_cog.tif",
        "s5p-l3grd-ch4-001-week-20201123_cog.tif",
        "s5p-l3grd-ch4-001-week-20201130_cog.tif",
        "s5p-l3grd-ch4-001-week-20201207_cog.tif",
        "s5p-l3grd-ch4-001-week-20201214_cog.tif",
        "s5p-l3grd-ch4-001-week-20201221_cog.tif",
        "s5p-l3grd-ch4-001-week-20201228_cog.tif",
        "s5p-l3grd-ch4-001-week-20210104_cog.tif",
        "s5p-l3grd-ch4-001-week-20210111_cog.tif",
        "s5p-l3grd-ch4-001-week-20210118_cog.tif",
        "s5p-l3grd-ch4-001-week-20210125_cog.tif",
        "s5p-l3grd-ch4-001-week-20210201_cog.tif",
        "s5p-l3grd-ch4-001-week-20210208_cog.tif",
        "s5p-l3grd-ch4-001-week-20210215_cog.tif",
        "s5p-l3grd-ch4-001-week-20210222_cog.tif",
        "s5p-l3grd-ch4-001-week-20210301_cog.tif",
        "s5p-l3grd-ch4-001-week-20210308_cog.tif",
        "s5p-l3grd-ch4-001-week-20210315_cog.tif",
        "s5p-l3grd-ch4-001-week-20210322_cog.tif",
        "s5p-l3grd-ch4-001-week-20210329_cog.tif",
        "s5p-l3grd-ch4-001-week-20210405_cog.tif",
        "s5p-l3grd-ch4-001-week-20210412_cog.tif",
        "s5p-l3grd-ch4-001-week-20210419_cog.tif",
        "s5p-l3grd-ch4-001-week-20210426_cog.tif",
        "s5p-l3grd-ch4-001-week-20210503_cog.tif",
        "s5p-l3grd-ch4-001-week-20210510_cog.tif",
        "s5p-l3grd-ch4-001-week-20210517_cog.tif",
        "s5p-l3grd-ch4-001-week-20210524_cog.tif",
        "s5p-l3grd-ch4-001-week-20210531_cog.tif",
        "s5p-l3grd-ch4-001-week-20210607_cog.tif",
        "s5p-l3grd-ch4-001-week-20210614_cog.tif",
        "s5p-l3grd-ch4-001-week-20210621_cog.tif",
        "s5p-l3grd-ch4-001-week-20210628_cog.tif",
        "s5p-l3grd-ch4-001-week-20210705_cog.tif",
        "s5p-l3grd-ch4-001-week-20210712_cog.tif",
        "s5p-l3grd-ch4-001-week-20210719_cog.tif",
        "s5p-l3grd-ch4-001-week-20210726_cog.tif",
        "s5p-l3grd-ch4-001-week-20210802_cog.tif",
        "s5p-l3grd-ch4-001-week-20210809_cog.tif",
        "s5p-l3grd-ch4-001-week-20210816_cog.tif",
        "s5p-l3grd-ch4-001-week-20210823_cog.tif",
        "s5p-l3grd-ch4-001-week-20210830_cog.tif",
        "s5p-l3grd-ch4-001-week-20210906_cog.tif",
        "s5p-l3grd-ch4-001-week-20210913_cog.tif",
        "s5p-l3grd-ch4-001-week-20210920_cog.tif",
        "s5p-l3grd-ch4-001-week-20210927_cog.tif",
        "s5p-l3grd-ch4-001-week-20211004_cog.tif",
        "s5p-l3grd-ch4-001-week-20211011_cog.tif",
        "s5p-l3grd-ch4-001-week-20211018_cog.tif",
        "s5p-l3grd-ch4-001-week-20211025_cog.tif",
        "s5p-l3grd-ch4-001-week-20211101_cog.tif",
        "s5p-l3grd-ch4-001-week-20211108_cog.tif",
        "s5p-l3grd-ch4-001-week-20220815_cog.tif",
        "s5p-l3grd-ch4-001-week-20220822_cog.tif",
        "s5p-l3grd-ch4-001-week-20220829_cog.tif",
        "s5p-l3grd-ch4-001-week-20231030_cog.tif",
        "s5p-l3grd-ch4-001-week-20231120_cog.tif",
        "s5p-l3grd-ch4-001-week-20240115_cog.tif",
        "s5p-l3grd-ch4-001-week-20240122_cog.tif",
        "s5p-l3grd-ch4-001-week-20240129_cog.tif",
        "s5p-l3grd-ch4-001-week-20240212_cog.tif",
        "s5p-l3grd-ch4-001-week-20240318_cog.tif",
        "s5p-l3grd-ch4-001-week-20240325_cog.tif",
        "s5p-l3grd-ch4-001-week-20240401_cog.tif",
        "s5p-l3grd-ch4-001-week-20240408_cog.tif",
        "s5p-l3grd-ch4-001-week-20240415_cog.tif",
        "s5p-l3grd-ch4-001-week-20240422_cog.tif",
        "s5p-l3grd-ch4-001-week-20240429_cog.tif",
        "s5p-l3grd-ch4-001-week-20240506_cog.tif",
        "s5p-l3grd-ch4-001-week-20240513_cog.tif",
        "s5p-l3grd-ch4-001-week-20240520_cog.tif",
        "s5p-l3grd-ch4-001-week-20240527_cog.tif",
        "s5p-l3grd-ch4-001-week-20240603_cog.tif",
        "s5p-l3grd-ch4-001-week-20240610_cog.tif",
        "s5p-l3grd-ch4-001-week-20240617_cog.tif",
        "s5p-l3grd-ch4-001-week-20240624_cog.tif",
        "s5p-l3grd-ch4-001-week-20240701_cog.tif",
        "s5p-l3grd-ch4-001-week-20240708_cog.tif",
        "s5p-l3grd-ch4-001-week-20240715_cog.tif",
        "s5p-l3grd-ch4-001-week-20240722_cog.tif",
        "s5p-l3grd-ch4-001-week-20240729_cog.tif",
        "s5p-l3grd-ch4-001-week-20240805_cog.tif",
        "s5p-l3grd-ch4-001-week-20240812_cog.tif",
        "s5p-l3grd-ch4-001-week-20240819_cog.tif",
        "s5p-l3grd-ch4-001-week-20240826_cog.tif",
        "s5p-l3grd-ch4-001-week-20240902_cog.tif",
        "s5p-l3grd-ch4-001-week-20240909_cog.tif",
        "s5p-l3grd-ch4-001-week-20240916_cog.tif",
        "s5p-l3grd-ch4-001-week-20240923_cog.tif",
        "s5p-l3grd-ch4-001-week-20240930_cog.tif",
        "s5p-l3grd-ch4-001-week-20241007_cog.tif",
        "s5p-l3grd-ch4-001-week-20241014_cog.tif",
        "s5p-l3grd-ch4-001-week-20241021_cog.tif",
    ]
}
