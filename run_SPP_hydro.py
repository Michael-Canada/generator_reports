# Environment: placebo_api_local

import sys

sys.path.append("/Users/michael.simantov/Documents/mu-placebo-api")
sys.path.append("/Users/michael.simantov/Documents/mu-placebo-api/placebo_api")
sys.path.append("/Users/michael.simantov/Documents/mu-placebo-api/placebo_api/utils")

# from typing import NamedTuple, Dict
# # from placebo_api.utils import api_utils, date_utils
# import api_utils, date_utils
# from placebo.utils import snowflake_utils
# # from placebo_api.utils.date_utils import LocalizedDateTime
# from date_utils import LocalizedDateTime
import pandas as pd
import datetime

# from tqdm import tqdm
# import pickle
import requests
import io
import matplotlib.pyplot as plt

# from matplotlib.patches import Circle
import numpy as np
import os
import re
from datetime import datetime
import pytz

# from placebo.task_producer.commons import ResourceDB, NaturalGasPrice
# from placebo.task_producer.commons import CoalPrice


# eia_923_fuel_receipts_and_costs = pd.read_parquet("eia_923_fuel_receipts_and_costs.parquet")
# Nat_gas_prices = pd.read_parquet('/Users/michael.simantov/Documents/generator_gas_prices/ng_hub_definition_parquet.parquet')
# file = pd.read_parquet('/Users/michael.simantov/Documents/generator_gas_prices/2024-07-16.parquet')

GO_TO_GCLOUD = True
SAVE_RESULTS = True

run = "miso"
this_collection = "miso-se"

run = "spp"
this_collection = "spp-se"

# run = 'ercot'
# this_collection = 'ercot-rt-se'

URL_ROOT = "https://api1.marginalunit.com/muse/api"


def _get_auth():
    return tuple(os.environ["MU_API_AUTH"].split(":"))
    # return tuple(os.environ["SELF"].split(":"))


AUTH = _get_auth()


def _get_dataframe(muse_path, method=requests.get):
    # muse_path is a http path to a muse endpoint
    url = URL_ROOT + muse_path
    # print(url)
    resp = method(url, auth=AUTH)
    if resp.status_code != 200:
        print(resp.text)

    try:
        resp.raise_for_status()
    except:
        print(resp.text)
        print("failed URL:", muse_path)
        return None
    return pd.read_csv(io.StringIO(resp.text))


def _get_data_from_full_URL(url):
    resp = requests.get(url, auth=AUTH)
    if resp.status_code != 200:
        print(resp.text)
    try:
        resp.raise_for_status()
    except:
        print(resp.text)
        print("failed URL:", url)
        return None
    return pd.read_csv(io.StringIO(resp.text))


# Generation List: /{iso}/cluster_generations.csv
# Generation Timeseries: /{iso}/cluster_generation.csv


# return all the collections:
# this_collection = _get_data_from_full_URL(f'https://api1.marginalunit.com/reflow/collections')

# for a collection, return its cases:
this_case = _get_data_from_full_URL(
    f"https://api1.marginalunit.com/reflow/{this_collection}/cases"
)
this_case = this_case.values[-1][0]

# Not helpful:
# Nat_gas_prices = pd.read_parquet('/Users/michael.simantov/Nat_gas_prices.parquet')
# Coal_prices = pd.read_parquet('/Users/michael.simantov/Coal_prices.parquet')
# coal_prices = pd.read_csv('coal_prices_may_2021_jan_2021.csv')
# natural_gas_prices = pd.read_csv('natural_gas_prices_may_2021_jan_2021.csv')


# given a collection (e.g., miso-se) and a case (e.g., miso_se_20240618-1800_AREVA), return its generators:
if GO_TO_GCLOUD:
    generators = _get_data_from_full_URL(
        f"https://api1.marginalunit.com/reflow/{this_collection}/{this_case}/generators"
    )
    generators.to_csv(f"generators_from_reflow_{run}.csv", index=False)
else:
    generators = pd.read_csv(f"generators_from_reflow_{run}.csv")


# from AK:
if GO_TO_GCLOUD:
    # EIA_923_FUEL_COSTS = "gs://marginalunit-placebo-metadata/supporting-files/eia_923_fuel_receipts_and_costs.parquet"
    eia_923_fuel_receipts_and_costs = pd.read_parquet(
        "eia_923_fuel_receipts_and_costs.parquet"
    )
    # eia_923_fuel_receipts_and_costs.to_parquet("eia_923_fuel_receipts_and_costs.parquet", index=False)

    # the following is the latest data about generators that is found when looking at Mosaic:
    all_generators = _get_dataframe(f"/{run}/cluster_generations.csv")

    # USA_NG_HUB_DEFINITION = "gs://marginalunit-placebo-metadata/supporting-files/ng_hub_definition_parquet.parquet"
    ng_hub_definition_parquet = pd.read_parquet("ng_hub_definition_parquet.parquet")
    # ng_hub_definition_parquet.to_parquet("ng_hub_definition_parquet.parquet", index=False)
    all_generators.to_csv(f"all_generators_{run}.csv", index=False)

else:
    eia_923_fuel_receipts_and_costs = pd.read_parquet(
        "eia_923_fuel_receipts_and_costs.parquet"
    )
    ng_hub_definition_parquet = pd.read_parquet("ng_hub_definition_parquet.parquet")
    all_generators = pd.read_csv(f"all_generators_{run}.csv")

######################################################

# Generation List: /{iso}/cluster_generations.csv
# Generation Timeseries: /{iso}/cluster_generation.csv

if GO_TO_GCLOUD:
    URL_4 = f"https://api1.marginalunit.com/pr-forecast/{run}/generators"
    names_all_generators = _get_data_from_full_URL(URL_4)
    names_all_generators.to_csv(f"names_all_generators_{run}.csv", index=False)
else:
    names_all_generators = pd.read_csv(f"names_all_generators_{run}.csv")


# MUSE generation for a specific generator and time range
def find_MUSE_data(generator_uid, from_date, to_date):
    URL_1 = f"https://api1.marginalunit.com/muse/api/{run}/cluster_generation.csv?uid={generator_uid}&from_date={from_date}&to_date={to_date}&resample_rate=1h"
    generation_per_timestamp = _get_data_from_full_URL(URL_1)
    return generation_per_timestamp


# The function gets a string and extract from it a date. For example: the input could be 'miso_se_20240627-1800_AREVA' and the output should be '2024-06-27'
def extract_date_from_string(string, run):
    if run == "miso":
        found_date = string.split("_")[2].split("-")[0]
        return found_date[:4] + "-" + found_date[4:6] + "-" + found_date[6:]
    elif run == "ercot":
        found_date = string.split("_")[3]
        return found_date
    elif run == "spp":
        found_date = string.split("_")[1]
        return found_date[:4] + "-" + found_date[4:6] + "-" + found_date[6:8]


# The function gets a string and extract from it a date. For example: the input could be 'miso_se_20240627-1800_AREVA' and the output should be '1800'
def extract_time_from_string(string, run):
    if run == "miso":
        found_time = string.split("_")[2].split("-")[1]
        return found_time


# Convert 'case' in Actual_generation_reflow to match 'timestamp' in generation_forecast
def convert_case_to_timestamp(case_str):
    # Extract date and time from the case string
    if run == "miso":
        match = re.search(r"(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})", case_str)
        if match:
            date_str = f"{match.group(1)}-{match.group(2)}-{match.group(3)} {match.group(4)}:{match.group(5)}:00"
            # Convert to datetime object
            date_obj = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
            # Assuming generation_forecast.timestamp is in UTC (adjust if necessary)
            date_obj_utc = pytz.utc.localize(date_obj)
            return date_obj_utc
        return None
    elif run == "spp":
        match = re.search(r"stateestimator_(\d{8})(\d{4})", case_str)
        if match:
            date_str = f"{match.group(1)} {match.group(2)}"
            # Convert to datetime object
            date_obj = datetime.strptime(date_str, "%Y%m%d %H%M")
            # Assuming generation_forecast.timestamp is in UTC (adjust if necessary)
            date_obj_utc = pytz.utc.localize(date_obj)
            return date_obj_utc
        return None
    else:
        raise ValueError("Unsupported run type")


num_cases_found = 0
# cntr_not_in_MUSE = 0

df = pd.DataFrame(
    columns=[
        "case_converted",
        "name",
        "unit_id",
        "actual_pg",
        "pmax_Actual",
        "month",
        "capacity_factor",
    ]
)
for ii in range(len(generators)):
    print(f"working on generator: {ii} out of {len(generators)}")
    print("counters 23423345:", num_cases_found)

    # if ii < 1301:
    #     continue
    if SAVE_RESULTS and ii == 0:
        pass
    elif SAVE_RESULTS and ii % 100 == 0:
        df.to_csv(f"all_hydro_data_{run}_{ii}.csv", index=False)
        df = pd.DataFrame(
            columns=[
                "case_converted",
                "name",
                "unit_id",
                "actual_pg",
                "pmax_Actual",
                "month",
                "capacity_factor",
            ]
        )

    # print(generators['name'].values[ii])
    # if not ('doud2' in generators['name'].values[ii].lower()):
    #     continue

    name = generators["name"].values[ii]
    # main_name is part of 'name' that is before the first space:
    main_name = name.split(" ")[0]
    name = generators["name"].values[ii].replace(" ", "%20")
    zone = generators["zone_name"].values[ii]

    ######################################################
    # This is to save development time and is not necessary later:
    if (
        not len(
            names_all_generators[
                names_all_generators["name"].str.contains(main_name) == True
            ]
        )
        > 0
    ):
        continue
    ######################################################

    # MUSE_data = find_MUSE_data(main_name, from_date, to_date)
    # if not MUSE_data is None:
    #     cntr_not_in_MUSE += 1
    # print(18)

    Actual_generation_reflow = _get_data_from_full_URL(
        f"https://api1.marginalunit.com/reflow/{this_collection}/generator?name={name}"
    )
    # IMPORTANT: Actual_generation_reflow's time is local, while generation_forecast's time is in UTC!   #miso_se_20240813-1800_AREVA
    Actual_generation_reflow["case_converted"] = Actual_generation_reflow["case"].apply(
        convert_case_to_timestamp
    )
    Actual_generation_reflow.rename(
        columns={"pg": "actual_pg", "pmin": "pmin_Actual", "pmax": "pmax_Actual"},
        inplace=True,
    )

    latest_Actual_date = extract_date_from_string(
        Actual_generation_reflow.case.values[-1], run
    )
    # extract_time_from_string(Actual_generation_reflow.case.values[-1], run)
    from_date = latest_Actual_date
    # from_date = "2024-07-01"
    generation_forecast_more_info = _get_data_from_full_URL(
        f"https://api1.marginalunit.com/pr-forecast/{run}/generator?days_prior=1&uid={name}&as_of={from_date}T04:00:00-05:00"
    )
    if len(generation_forecast_more_info) == 0:
        continue

    if generation_forecast_more_info["fuel"].nunique() > 1:
        print(
            f"8345877493, {name}, {main_name}, has more than one fuel type ({generation_forecast_more_info['fuel'].nunique()})"
        )
        continue

    if not generation_forecast_more_info["fuel"].values[0] == "HYDRO":
        continue

    scraped_data = pd.DataFrame(
        columns=["case_converted", "name", "unit_id", "actual_pg", "pmax_Actual"]
    )
    scraped_data = pd.concat(
        [
            scraped_data,
            Actual_generation_reflow[
                ["case_converted", "name", "unit_id", "actual_pg", "pmax_Actual"]
            ],
        ],
        ignore_index=True,
    )
    # scraped_data.columns = Index(['case_converted', 'name', 'unit_id', 'actual_pg', 'pmax_Actual'], dtype='object')
    scraped_data.loc[:, "month"] = scraped_data["case_converted"].dt.month

    result = (
        scraped_data.groupby(["name", "unit_id", "month"])
        .agg({"actual_pg": "mean", "pmax_Actual": "mean"})
        .reset_index()
    )
    result["capacity_factor"] = result["actual_pg"] / result["pmax_Actual"]

    # merge result back into df
    scraped_data = scraped_data.merge(
        result[["name", "unit_id", "month", "capacity_factor"]],
        on=["name", "unit_id", "month"],
        how="left",
    ).sort_values(["name", "unit_id", "case_converted"])

    df = pd.concat([df, scraped_data], ignore_index=True)
    num_cases_found += 1

    # for dev purposes:
    if False:
        plt.figure(figsize=(15, 11))
        plt.plot(
            scraped_data["case_converted"],
            scraped_data["actual_pg"] / scraped_data["pmax_Actual"],
            label="ratio of observed pg to Actual Pmax",
            linewidth=1,
        )
        plt.plot(
            scraped_data["case_converted"],
            scraped_data["capacity_factor"],
            label="Capacity Factor",
            linewidth=5,
        )
        plt.legend()
        plt.xlabel("Timestamp")
        plt.ylabel("Capacity Factor")
        plt.title(f"Capacity Factor for Generator {main_name}")
        plt.xticks(rotation=45)  # Rotate x-axis ticks
        plt.tight_layout()  # Adjust layout to prevent clipping of tick-labels
        plt.close()

if SAVE_RESULTS:
    df.to_csv(f"all_hydro_data_{run}_{ii}.csv", index=False)


#
