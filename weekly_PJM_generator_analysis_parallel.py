# Environment: placebo_api_local

import sys

sys.path.append("/Users/michael.simantov/Documents/mu-placebo-api")
sys.path.append("/Users/michael.simantov/Documents/mu-placebo-api/placebo_api")
sys.path.append("/Users/michael.simantov/Documents/mu-placebo-api/placebo_api/utils")

from typing import NamedTuple, Dict

# from placebo_api.utils import api_utils, date_utils
import api_utils, date_utils
from placebo.utils import snowflake_utils

# from placebo_api.utils.date_utils import LocalizedDateTime
from date_utils import LocalizedDateTime
import pandas as pd
import datetime
from tqdm import tqdm
import pickle
import requests
import io
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import os
import re
from datetime import datetime
import pytz

# from scipy.stats import entropy

# from placebo.task_producer.commons import ResourceDB, NaturalGasPrice
# from placebo.task_producer.commons import CoalPrice


# eia_923_fuel_receipts_and_costs = pd.read_parquet("eia_923_fuel_receipts_and_costs.parquet")
# Nat_gas_prices = pd.read_parquet('/Users/michael.simantov/Documents/generator_gas_prices/ng_hub_definition_parquet.parquet')
# file = pd.read_parquet('/Users/michael.simantov/Documents/generator_gas_prices/2024-07-16.parquet')

GO_TO_GCLOUD = True
SAVE_RESULTS = True
CHECK_EXISTENCE_OF_GENERATOR_IN_OUR_LIST = False



# this works for PJM:
# ISO_name = "pjm"
# run = "pjm_20250404"  #pjm_20250401
# run_version = "pjm_20250404"   #pjm_20250401
# this_collection = "pjm-ftr"


ISO_name = "pjm"
run = "pjm_20250515"  #pjm_20250404
run_version = "pjm_20250515"   #pjm_20250404
this_collection = "pjm-ftr"


# ISO_name = "miso"
# run = "miso"
# run_version = "miso"
# this_collection = "miso-se"

# ISO_name = "spp"
# run = 'spp'
# run_version = "spp_20250311"
# run_version = 'spp'
# this_collection = "spp-se"

# ISO_name = "ercot"
# run = "ercot"
# this_collection = "ercot-rt-se"

if ISO_name == 'pjm':
    this_collection = "miso-se"

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
        print("456045876 failed URL:", muse_path)
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
        print("u29384923 failed URL:", url)
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
# if not isinstance(this_case, str)  and ISO_name == "pjm":
#     this_case = str(this_case)[:-2]

# Not helpful:
# Nat_gas_prices = pd.read_parquet('/Users/michael.simantov/Nat_gas_prices.parquet')
# Coal_prices = pd.read_parquet('/Users/michael.simantov/Coal_prices.parquet')
# coal_prices = pd.read_csv('coal_prices_may_2021_jan_2021.csv')
# natural_gas_prices = pd.read_csv('natural_gas_prices_may_2021_jan_2021.csv')


# given a collection (e.g., miso-se) and a case (e.g., miso_se_20240618-1800_AREVA), return its generators:
if GO_TO_GCLOUD:
    
    # if ISO_name == 'pjm':
    #     generators = _get_data_from_full_URL(
    #         f"https://api1.marginalunit.com/pr-forecast/{run_version}/generators")
    # else:
    generators = _get_data_from_full_URL(
        f"https://api1.marginalunit.com/reflow/{this_collection}/{this_case}/generators")


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
    all_generators = _get_dataframe(f"/{ISO_name}/cluster_generations.csv")

    # USA_NG_HUB_DEFINITION = "gs://marginalunit-placebo-metadata/supporting-files/ng_hub_definition_parquet.parquet"
    ng_hub_definition_parquet = pd.read_parquet("ng_hub_definition_parquet.parquet")
    # ng_hub_definition_parquet.to_parquet("ng_hub_definition_parquet.parquet", index=False)
    all_generators.to_csv(f"all_generators_{ISO_name}.csv", index=False)

else:
    eia_923_fuel_receipts_and_costs = pd.read_parquet(
        "eia_923_fuel_receipts_and_costs.parquet"
    )
    ng_hub_definition_parquet = pd.read_parquet("ng_hub_definition_parquet.parquet")
    all_generators = pd.read_csv(f"all_generators_{ISO_name}.csv")

######################################################

# Generation List: /{iso}/cluster_generations.csv
# Generation Timeseries: /{iso}/cluster_generation.csv

if CHECK_EXISTENCE_OF_GENERATOR_IN_OUR_LIST:
    if GO_TO_GCLOUD:
        URL_4 = f"https://api1.marginalunit.com/pr-forecast/{run_version}/generators"
        names_all_generators = _get_data_from_full_URL(URL_4)
        names_all_generators.to_csv(f"names_all_generators_{run}.csv", index=False)

    else:
        names_all_generators = pd.read_csv(f"names_all_generators_{run}.csv")


# MUSE generation for a specific generator and time range
def find_MUSE_data(generator_uid, from_date, to_date):
    URL_1 = f"https://api1.marginalunit.com/muse/api/{run}/cluster_generation.csv?uid={generator_uid}&from_date={from_date}&to_date={to_date}&resample_rate=1h"
    generation_per_timestamp = _get_data_from_full_URL(URL_1)
    return generation_per_timestamp


# def convert_summer_to_winter_time(string):
#     # convert this "2023-11-03 21:00:00-06:00" to this "2023-11-03 20:00:00-06:00"
#     if int(string[11:13]) > 0:
#         string = string[:11] + str(int(string[11:13]) - 1) + string[13:]
#     else:
#         string = string[:11] + '23' + string[13:]
#         string()


def convert_time(string):
    new_string = string[:-6] + "+00:00"
    return new_string


# The function gets a string and extract from it a date. For example: the input could be 'miso_se_20240627-1800_AREVA' and the output should be '2024-06-27'
def extract_date_from_string(string, run):
    if (run == "miso") or (run =="pjm"):
        found_date = string.split("_")[2].split("-")[0]
        return found_date[:4] + "-" + found_date[4:6] + "-" + found_date[6:]
    elif run == "ercot":
        found_date = string.split("_")[3]
        return found_date
    elif run == "spp":
        found_date = string.split("_")[1]
        return found_date[:4] + "-" + found_date[4:6] + "-" + found_date[6:8]
    # elif run == "pjm":
    #     string = str(string)
    #     return string[:4] + "-" + string[4:6] + "-" + string[6:8]


# The function gets a string and extract from it a date. For example: the input could be 'miso_se_20240627-1800_AREVA' and the output should be '1800'
def extract_time_from_string(string, run):
    if (ISO_name == "miso") or (ISO_name == "pjm"):
        found_time = string.split("_")[2].split("-")[1]
        return found_time


# Convert 'case' in Actual_generation_reflow to match 'timestamp' in generation_forecast
def convert_case_to_timestamp(case_str):
    # Extract date and time from the case string
    if (ISO_name == "miso") or ISO_name == "pjm":
        match = re.search(r"(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})", case_str)
        if match:
            date_str = f"{match.group(1)}-{match.group(2)}-{match.group(3)} {match.group(4)}:{match.group(5)}:00"
            # Convert to datetime object
            date_obj = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
            # Assuming generation_forecast.timestamp is in UTC (adjust if necessary)
            date_obj_utc = pytz.utc.localize(date_obj)
            return date_obj_utc
        return None
    elif ISO_name == "spp":
        match = re.search(r"stateestimator_(\d{8})(\d{4})", case_str)
        if match:
            date_str = f"{match.group(1)} {match.group(2)}"
            # Convert to datetime object
            date_obj = datetime.strptime(date_str, "%Y%m%d %H%M")
            # Assuming generation_forecast.timestamp is in UTC (adjust if necessary)
            date_obj_utc = pytz.utc.localize(date_obj)
            return date_obj_utc
        return None
    # elif ISO_name == "pjm":
    #     pass
    else:
        raise ValueError("Unsupported run type")


# num_cases_found = 0
# cntr_not_in_MUSE = 0

def reset_df():
    df = pd.DataFrame(
        columns=[
            "generator_uid",
            "timestamp",
            "actual_pg",
            "fcst_pg",
            "zone_uid",
            "fuel_type",
            "unit_id",
            "pmin_Actual",
            "pmin_Forecast",
            "pmax_Actual",
            "pmax_Forecast",
        ]
    )
    return df


# Function to calculate Mean Absolute Error (MAE)
def max_generation_error(y_true, y_pred):
    return np.max(y_pred - y_true )


def min_generation_error(y_true, y_pred):
    return np.min(y_pred - y_true)


# Function to calculate R-squared (R2)
def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)


def entropy(p, q):
    p = p[p > 0]
    q = q[q > 0]

    return np.sum(p * np.log(p / q))


# Calculate the date 6 months and 3 weeks ago from today with UTC timezone
six_months_ago = pd.Timestamp.now(tz="UTC") - pd.DateOffset(months=6)
six_weeks_ago = pd.Timestamp.now(tz="UTC") - pd.DateOffset(weeks=6)

# prepare a dataframe named results_df to store the data per generator

def reset_results_df():
    results_df = pd.DataFrame(
        columns=[
            "generated_uid", "name","unit_id","fuel_type","zone_uid","RMSE_over_generation","MAE_over_generation","num_hrs_fcst_above_actual_both_non_zero","num_hrs_actual_above_fcst_both_non_zero",
            "total_overgeneration",
            "TP",
            "TN",
            "FP",
            "FN",
            "MAX_GENERATION_ERROR",
            "MIN_GENERATION_ERROR",
            "R_SQUARED",
            "%_running",
            "num_running_hours",
            "HISTORIC_IS_ZERO",
            "FORECAST_IS_ZERO",
            "P_MAX_ACTUAL",
            "P_MAX_FORECAST",
        ]
    )
    return results_df

today_date_str = datetime.now().strftime("%Y-%m-%d")

def run_one_generator(ii):
    # print(f"working on generator: {ii} out of {len(generators)}")

    start_time_print = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d %H:%M:%S")
    start_time = time.time()
    
    # print("counters 23423345:", num_cases_found)

    # if ii < 2201:   #miso
    #     continue

    orig_name = generators["name"].values[ii]

    # if (not "BERGEN" in orig_name.upper()):
    #     return None, None
    # else:
    #     print(18)

    main_name = orig_name.split(" ")[0]
    name = generators["name"].values[ii].replace(" ", "%20")
    zone = generators["zone_name"].values[ii]

    ######################################################
    # This is to save development time and is not necessary later:
    if CHECK_EXISTENCE_OF_GENERATOR_IN_OUR_LIST:
        if (
            not len(
                names_all_generators[
                    names_all_generators["name"].str.contains(main_name) == True
                ]
            )
            > 0
        ):
            return None, None
    ######################################################

    # MUSE_data = find_MUSE_data(main_name, from_date, to_date)
    # if not MUSE_data is None:
    #     cntr_not_in_MUSE += 1
    # print(18)
    # USE THIS:
    Actual_generation_reflow = _get_data_from_full_URL(
        f"https://api1.marginalunit.com/reflow/{this_collection}/generator?name={name}"
    )
    latest_Actual_date = extract_date_from_string(
        Actual_generation_reflow.case.values[-1], ISO_name
    )

    # add a column with the date in prpoper format:
    # IMPORTANT: Actual_generation_reflow's time is local, while generation_forecast's time is in UTC!
    Actual_generation_reflow["case_converted"] = Actual_generation_reflow["case"].apply(
        convert_case_to_timestamp
    )

    # extract_time_from_string(Actual_generation_reflow.case.values[-1], run)
    from_date = latest_Actual_date
    # from_date = '2024-07-01'
    # from_date = '2024-06-27'

    # This is the generation forecast in Mosaic:
    try:
        generation_forecast = _get_data_from_full_URL(
            f"https://api1.marginalunit.com/pr-forecast/{run_version}/generator/lookahead_timeseries?uid={name}&as_of={from_date}T04:00:00-05:00"
        )
        # f"https://api1.marginalunit.com/reflow/{this_collection}/{this_case}/generators"
    except:
        generation_forecast = _get_data_from_full_URL(
            f"https://api1.marginalunit.com/pr-forecast/{run_version}/generator/lookahead_timeseries?uid={name}&as_of={from_date}T04:00:00-05:00"
        )
    try:
        generation_forecast_more_info = _get_data_from_full_URL(
            f"https://api1.marginalunit.com/pr-forecast/{run_version}/generator?days_prior=1&uid={name}&as_of={from_date}T04:00:00-05:00"
        )
    except:
        generation_forecast_more_info = _get_data_from_full_URL(
            f"https://api1.marginalunit.com/pr-forecast/{run_version}/generator?days_prior=1&uid={name}&as_of={from_date}T04:00:00-05:00"
        )
    if (ISO_name == "miso") or (ISO_name == "pjm"):
        # if (generation_forecast is None) or (generation_forecast_more_info is None) or (len(generation_forecast_more_info) == 0 or len(generation_forecast) == 0):
        #     return None, None

        # generation_forecast["fuel"] = generation_forecast_more_info["fuel"].values[0]
        # generation_forecast["pmin"] = generation_forecast_more_info["pmin"].values[0]
        # generation_forecast["pmax"] = generation_forecast_more_info["pmax"].values[0]

        if (generation_forecast is None) or (len(generation_forecast) == 0):
            return None, None

        generation_forecast["fuel"] = 'DUMMY'
        generation_forecast["pmin"] = 'DUMMY'
        generation_forecast["pmax"] = 'DUMMY'

    elif ISO_name == "spp":
        if (generation_forecast is None) or (len(generation_forecast) == 0):
            return None, None

        generation_forecast["fuel"] = 'DUMMY'
        generation_forecast["pmin"] = 'DUMMY'
        generation_forecast["pmax"] = 'DUMMY'

    # if ISO_name == "pjm":
    #     if (generation_forecast is None) or (generation_forecast_more_info is None) or (len(generation_forecast_more_info) == 0 or (len(generation_forecast) == 0)):
    #         return None, None


    # if the generator's pg > 0 whenever the status is True then we may have to set it to "must run":
    print('Michael Simantov')
    Candidate_for_must_run = False
    percent_of_running = -1
    num_running_hours = -1
    # if len(Actual_generation_reflow) > 0:
    #     if (
    #         len(Actual_generation_reflow[Actual_generation_reflow["status"] == True])
    #         > 0
    #     ):
            # pg_when_running = Actual_generation_reflow[
            #     (Actual_generation_reflow["status"] == True)
            #     & (Actual_generation_reflow["case_converted"] > six_months_ago)
            # ]["pg"].values

            # total_number_of_rows = len(
            #     Actual_generation_reflow[
            #         (Actual_generation_reflow["case_converted"] > six_months_ago)
            #     ]
            # )

            # num_occurences_of_0 = sum(pg_when_running == 0)
            # num_occurences_not_0 = sum(pg_when_running != 0)
            # we only want to classify the generator as must_run if it has been running for more than 90% of the time (i.e, do not classify it as must_run if it has been in maintenance for more than 10% of the time)
            # if (
            #     ((num_occurences_not_0 + num_occurences_of_0) > 0)
            #     and (num_occurences_not_0 > 10 * num_occurences_of_0)
            #     and ((num_occurences_not_0 / total_number_of_rows) > 0.9)
            # ):
            #     percent_of_running = num_occurences_not_0 / (
            #         num_occurences_not_0 + num_occurences_of_0
            #     )
            #     num_running_hours = num_occurences_not_0
            #     if percent_of_running > 0.9:
            #         Candidate_for_must_run = True

    # if the dataframe generation_per_timestamp is empty then continue
    if not len(generation_forecast) > 0:
        return None, None
    
    # # IMPORTANT: Actual_generation_reflow's time is local, while generation_forecast's time is in UTC!
    # Actual_generation_reflow["case_converted"] = Actual_generation_reflow[
    #     "case"
    # ].apply(convert_case_to_timestamp)
    # Convert 'timestamp' in generation_forecast to datetime with UTC timezone
    # fmt: off


    print(
        f"{start_time_print}: ({ii}) Generator: {main_name}"
    )



    if (ISO_name == "miso"):
        generation_forecast["timestamp"] = pd.to_datetime(generation_forecast["timestamp"]).dt.tz_localize(None)
        generation_forecast["timestamp"] = generation_forecast["timestamp"].dt.tz_localize("UTC")
    elif (ISO_name == "pjm"):
        generation_forecast["timestamp"] = pd.to_datetime(generation_forecast["timestamp"], utc=True)
    elif ISO_name == "spp":
        generation_forecast["timestamp"] = generation_forecast["timestamp"].apply(convert_time)
    
        # Convert 'case_converted' to datetime with UTC timezone
        Actual_generation_reflow["case_converted"] = pd.to_datetime(
            Actual_generation_reflow["case_converted"], utc=True
        )

        # Convert 'timestamp' to datetime with UTC timezone
        generation_forecast["timestamp"] = pd.to_datetime(
            generation_forecast["timestamp"], utc=True
        )
    # fmt: on

    merged_df = pd.merge(
        Actual_generation_reflow,
        generation_forecast,
        left_on="case_converted",
        right_on="timestamp",
        how="inner",
    )

    columns = [
        "generator_uid",
        "timestamp",
        "actual_pg",
        "fcst_pg",
        "zone_uid",
        "fuel_type",
    ]
    merged_df = merged_df[
        [
            "timestamp",
            "name_x",
            "unit_id",
            "pg",
            "pmin_x",
            "pmin_y",
            "pmax_x",
            "pmax_y",
            "generation",
            # "fuel",
        ]
    ]

    merged_df.rename(
        columns={
            "generation": "fcst_pg",
            "name_x": "generator_uid",
            "pg": "actual_pg",
            "pmin_x": "pmin_Actual",
            "pmin_y": "pmin_Forecast",
            "pmax_x": "pmax_Actual",
            "pmax_y": "pmax_Forecast",
            # "fuel": "fuel_type",
        },
        inplace=True,
    )
    merged_df["zone_uid"] = zone
    merged_df["name"] = main_name

    merged_df = pd.merge(
        merged_df,
        all_generators[["uid", "label","fuel_type"]],
        left_on="name",
        right_on="uid",
        how="left",
    )
    merged_df.drop(columns=["uid"], inplace=True)

    # reorganize the columns:
    merged_df = merged_df[
        [
            "generator_uid",
            "timestamp",
            "actual_pg",
            "fcst_pg",
            "zone_uid",
            "fuel_type",
            "unit_id",
            "pmin_Actual",
            "pmin_Forecast",
            "pmax_Actual",
            "pmax_Forecast",
            "name",
        ]
    ]

    merged_df["must_run"] = Candidate_for_must_run
    merged_df["%_running"] = percent_of_running
    merged_df["num_running_hours"] = num_running_hours

    # df = pd.concat([df, merged_df], ignore_index=True)
    # num_cases_found += 1

    # merged_df > six_months_ago

    data_to_use = merged_df.sort_values(by="timestamp")
    historic_generated_power = data_to_use[data_to_use["timestamp"] > six_weeks_ago][
        "actual_pg"
    ]
    forecast_power = data_to_use[data_to_use["timestamp"] > six_weeks_ago]["fcst_pg"]

    # collect measures about the quality of the generation forecast:
    HISTORIC_IS_CONSTANT_ZERO = np.all(historic_generated_power == 0)
    FORECAST_IS_CONSTANT_ZERO = np.all(forecast_power == 0)
    MAX_GENERATION_ERROR = max_generation_error(
        historic_generated_power, forecast_power
    )
    MIN_GENERATION_ERROR = min_generation_error(
        historic_generated_power, forecast_power
    )
    if FORECAST_IS_CONSTANT_ZERO or HISTORIC_IS_CONSTANT_ZERO == 0:
        R_SQUARED = 1
    else:
        R_SQUARED = r2_score(historic_generated_power, forecast_power)

    # TP is the number of bins where both the actual and forecasted values are non-zero
    # TN is the number of bins where both the actual and forecasted values are zero
    # FP is the number of bins where the actual value is zero but the forecasted value is non-zero
    # FN is the number of bins where the actual value is non-zero but the forecasted value is zero
    TP = np.sum((historic_generated_power != 0) & (forecast_power != 0))
    TN = np.sum((historic_generated_power == 0) & (forecast_power == 0))
    FP = np.sum((historic_generated_power == 0) & (forecast_power > historic_generated_power))
    FN = np.sum((historic_generated_power > forecast_power) & (forecast_power == 0))

    num_hrs_forecast_above_actual_and_both_non_zero = np.sum((historic_generated_power != 0) & (forecast_power != 0) & (forecast_power > historic_generated_power))
    num_hrs_actual_above_forecast_and_both_non_zero = np.sum((historic_generated_power != 0) & (forecast_power != 0) & (forecast_power < historic_generated_power))

    total_overgeneration = np.sum(forecast_power - historic_generated_power)
    RMSE_over_generation = np.sqrt(np.mean((forecast_power - historic_generated_power) ** 2))
    MAE_over_generation = np.mean(np.abs(forecast_power - historic_generated_power))



    # Calculate the bin edges based on the combined data
    # normalized_historic_geherated_power = (
    #     historic_generated_power - np.mean(historic_generated_power)
    # ) / np.std(historic_generated_power)
    # normalized_forecast_power = (forecast_power - np.mean(forecast_power)) / np.std(
    #     forecast_power
    # )
    # combined_data = np.concatenate(
    #     [normalized_historic_geherated_power, normalized_forecast_power]
    # )
    # bin_edges = np.histogram_bin_edges(combined_data, bins=10)

    # # Calculate entropy-based metrics using the same bins
    # actual_distribution = np.histogram(
    #     historic_generated_power, bins=bin_edges, density=True
    # )[0]
    # forecasted_distribution = np.histogram(
    #     forecast_power, bins=bin_edges, density=True
    # )[0]

    # kl_divergence = entropy(actual_distribution, forecasted_distribution)

    # Store the results
    # results = {
    #     "generated_uid": merged_df.generator_uid[0],
    #     "fuel_type": merged_df.fuel_type[0],
    #     "zone_uid": merged_df.zone_uid[0],
    #     "unit_id": merged_df.unit_id[0],
    #     "name": merged_df.name[0],
    #     "%_running": merged_df["%_running"][0],
    #     "num_running_hours": merged_df["num_running_hours"][0],
    #     "HISTORIC_IS_ZERO": HISTORIC_IS_CONSTANT_ZERO,
    #     "FORECAST_IS_ZERO": FORECAST_IS_CONSTANT_ZERO,
    #     "MAX_GENERATION_ERROR": MAX_GENERATION_ERROR,
    #     "MIN_GENERATION_ERROR": MIN_GENERATION_ERROR,
    #     "R_SQUARED": R_SQUARED,
    #     "TP": TP,
    #     "TN": TN,
    #     "FP": FP,
    #     "FN": FN,
    #     "num_hrs_fcst_above_actual_both_non_zero":num_hrs_forecast_above_actual_and_both_non_zero,
    #     "num_hrs_actual_above_fcst_both_non_zero":num_hrs_actual_above_forecast_and_both_non_zero,
    #     "total_overgeneration" : total_overgeneration,
    #     'RMSE_over_generation': RMSE_over_generation,
    #     'MAE_over_generation':MAE_over_generation,
    #     "P_MAX_ACTUAL": merged_df.pmax_Actual.values[0],
    #     "P_MAX_FORECAST": merged_df.pmax_Forecast.values[0],
    # }
    results = {
        "generated_uid": merged_df.generator_uid[0],
        "name": merged_df.name[0],
        "unit_id": merged_df.unit_id[0],
        "fuel_type": merged_df.fuel_type[0],
        "zone_uid": merged_df.zone_uid[0],
        'RMSE_over_generation': RMSE_over_generation,
        'MAE_over_generation':MAE_over_generation,
        "num_hrs_fcst_above_actual_both_non_zero":num_hrs_forecast_above_actual_and_both_non_zero,
        "num_hrs_actual_above_fcst_both_non_zero":num_hrs_actual_above_forecast_and_both_non_zero,
        "total_overgeneration" : total_overgeneration,
        "TP": TP,
        "TN": TN,
        "FP": FP,
        "FN": FN,
        "MAX_GENERATION_ERROR": MAX_GENERATION_ERROR,
        "MIN_GENERATION_ERROR": MIN_GENERATION_ERROR,
        "R_SQUARED": R_SQUARED,        
        "%_running": merged_df["%_running"][0],
        "num_running_hours": merged_df["num_running_hours"][0],
        "HISTORIC_IS_ZERO": HISTORIC_IS_CONSTANT_ZERO,
        "FORECAST_IS_ZERO": FORECAST_IS_CONSTANT_ZERO,
        "P_MAX_ACTUAL": merged_df.pmax_Actual.values[0],
        "P_MAX_FORECAST": merged_df.pmax_Forecast.values[0],
    }

    # add resulta to the results dataframe results_df
    # results_df = results_df.append(results, ignore_index=True)


    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"total time: {elapsed_time:.2f} seconds")

    return merged_df, results


from joblib import Parallel, delayed
import time
import math

# Define batch size
batch_size = 100

# Calculate the number of batches
num_batches = math.ceil(len(generators) / batch_size)

# Process in batches
counter_of_batches = -1
for batch_idx in range(num_batches):
    # if batch_idx < 28:
    #     continue
    counter_of_batches += 1
    # if counter_of_batches < 2:
    #     continue
    df = reset_df()
    results_df = reset_results_df()
    found_results = False


    start_idx = batch_idx * batch_size
    end_idx = min((batch_idx + 1) * batch_size, len(generators))  # Ensure we don't go out of bounds

    # if (batch_idx + 1) < 10 or (batch_idx + 1) > 10:
    #     continue
    # if (batch_idx) < 2:
    #     continue
    print(f"Processing batch {batch_idx + 1} of {num_batches} (generators {start_idx} to {end_idx - 1})")

    # Run Parallel processing for the current batch
    res = Parallel(n_jobs=4)(
        delayed(run_one_generator)(i) for i in range(start_idx, end_idx)
    )

    # Store the results in the correlation matrix
    for ind, results in enumerate(res):
        merged_df = results[0]
        result = results[1]
        if result is None:
            continue

        found_results = True
        df = pd.concat([df, merged_df], ignore_index=True)
        results_df = results_df.append(result, ignore_index=True)

    if SAVE_RESULTS and found_results:
        df.to_csv(f"generator__analyzer_{run_version}__{batch_idx}_{today_date_str}.csv", index=False)
        results_df.to_csv(
            f"generator_forecast_stats__{run_version}__{batch_idx}_{today_date_str}.csv", index=False
        )
    print(
        f"Batch {batch_idx + 1} completed. Results saved to CSV files."
    )


print(18)


######################################################

