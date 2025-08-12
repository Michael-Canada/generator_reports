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

# from placebo.task_producer.commons import ResourceDB, NaturalGasPrice
# from placebo.task_producer.commons import CoalPrice


# eia_923_fuel_receipts_and_costs = pd.read_parquet("eia_923_fuel_receipts_and_costs.parquet")
# Nat_gas_prices = pd.read_parquet('/Users/michael.simantov/Documents/generator_gas_prices/ng_hub_definition_parquet.parquet')
# file = pd.read_parquet('/Users/michael.simantov/Documents/generator_gas_prices/2024-07-16.parquet')

GO_TO_GCLOUD = True
SAVE_RESULTS = True

run = "miso"
this_collection = "miso-se"

# run = "spp"
# this_collection = "spp-se"

# run = "ercot"
# this_collection = "ercot-rt-se"

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


# Calculate the date 6 months ago from today with UTC timezone
six_months_ago = pd.Timestamp.now(tz="UTC") - pd.DateOffset(months=6)

for ii in range(len(generators)):
    print(f"working on generator: {ii} out of {len(generators)}")
    print("counters 23423345:", num_cases_found)

    if ii < 5501:
        continue
    if SAVE_RESULTS and ii == 0:
        pass
    elif SAVE_RESULTS and ii % 100 == 0:
        df.to_csv(f"all_knowledge_df_Jan_20{run}_{ii}.csv", index=False)
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
                "must_run",
            ]
        )

    # print(generators['name'].values[ii])
    # if not ('doud2' in generators['name'].values[ii].lower()):
    #     continue

    name = generators["name"].values[ii]

    # if not "LEON_CRK_LCPCT1" in name:
    #     continue
    # else:
    #     print(18)
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
    # USE THIS:
    Actual_generation_reflow = _get_data_from_full_URL(
        f"https://api1.marginalunit.com/reflow/{this_collection}/generator?name={name}"
    )
    latest_Actual_date = extract_date_from_string(
        Actual_generation_reflow.case.values[-1], run
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
            f"https://api1.marginalunit.com/pr-forecast/{run}/generator/lookahead_timeseries?uid={name}&as_of={from_date}T04:00:00-05:00"
        )
    except:
        generation_forecast = _get_data_from_full_URL(
            f"https://api1.marginalunit.com/pr-forecast/{run}/generator/lookahead_timeseries?uid={name}&as_of={from_date}T04:00:00-05:00"
        )
    try:
        generation_forecast_more_info = _get_data_from_full_URL(
            f"https://api1.marginalunit.com/pr-forecast/{run}/generator?days_prior=1&uid={name}&as_of={from_date}T04:00:00-05:00"
        )
    except:
        generation_forecast_more_info = _get_data_from_full_URL(
            f"https://api1.marginalunit.com/pr-forecast/{run}/generator?days_prior=1&uid={name}&as_of={from_date}T04:00:00-05:00"
        )

    if len(generation_forecast_more_info) == 0 or len(generation_forecast) == 0:
        continue

    generation_forecast["fuel"] = generation_forecast_more_info["fuel"].values[0]
    generation_forecast["pmin"] = generation_forecast_more_info["pmin"].values[0]
    generation_forecast["pmax"] = generation_forecast_more_info["pmax"].values[0]

    # if the generator's pg > 0 whenever the status in True then we have to set it to "must run":
    Candidate_for_must_run = False
    percent_of_running = -1
    num_running_hours = -1
    if len(Actual_generation_reflow) > 0:
        if (
            len(Actual_generation_reflow[Actual_generation_reflow["status"] == True])
            > 0
        ):
            pg_when_running = Actual_generation_reflow[
                (Actual_generation_reflow["status"] == True)
                & (Actual_generation_reflow["case_converted"] > six_months_ago)
            ]["pg"].values

            total_number_of_rows = len(
                Actual_generation_reflow[
                    (Actual_generation_reflow["case_converted"] > six_months_ago)
                ]
            )

            num_occurences_of_0 = sum(pg_when_running == 0)
            num_occurences_not_0 = sum(pg_when_running != 0)
            # we only want to classify the generator as must_run if it has been running for more than 90% of the time (i.e, do not classify it as must_run if it has been in maintenance for more than 10% of the time)
            if (
                ((num_occurences_not_0 + num_occurences_of_0) > 0)
                and (num_occurences_not_0 > 10 * num_occurences_of_0)
                and ((num_occurences_not_0 / total_number_of_rows) > 0.9)
            ):
                percent_of_running = num_occurences_not_0 / (
                    num_occurences_not_0 + num_occurences_of_0
                )
                num_running_hours = num_occurences_not_0
                if percent_of_running > 0.9:
                    Candidate_for_must_run = True

    # if the dataframe generation_per_timestamp is empty then continue
    if len(generation_forecast) > 0:

        # testing to make sure that if we have data in generation_forecast then this generator is at names_all_generators
        if (
            not len(
                names_all_generators[
                    names_all_generators["name"].str.contains(main_name) == True
                ]
            )
            > 0
        ):
            cntr_not_in_names_all_generators += 1
            print(2939487, main_name)

        # # IMPORTANT: Actual_generation_reflow's time is local, while generation_forecast's time is in UTC!
        # Actual_generation_reflow["case_converted"] = Actual_generation_reflow[
        #     "case"
        # ].apply(convert_case_to_timestamp)
        # Convert 'timestamp' in generation_forecast to datetime with UTC timezone
        # fmt: off
        if run == "miso":
            generation_forecast["timestamp"] = pd.to_datetime(generation_forecast["timestamp"]).dt.tz_localize(None)
            generation_forecast["timestamp"] = generation_forecast["timestamp"].dt.tz_localize("UTC")
        elif run == "spp":
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
                "fuel",
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
                "fuel": "fuel_type",
            },
            inplace=True,
        )
        merged_df["zone_uid"] = zone
        merged_df["name"] = main_name

        merged_df = pd.merge(
            merged_df,
            all_generators[["uid", "label"]],
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

        df = pd.concat([df, merged_df], ignore_index=True)
        num_cases_found += 1
        oo = 1


if SAVE_RESULTS:
    df.to_csv(f"all_knowledge_df_Jan_20{run}_{ii}.csv", index=False)
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

print(
    "counters 398475938485:",
    cntr_in_names_all_generators,
    cntr_not_in_names_all_generators,
)
print(18)


######################################################


######################################################


if False:

    # df = pd.DataFrame({
    #     'time_stamp': pd.date_range(start='1/1/2023', periods=100, freq='H'),
    #     'generation': np.random.choice(['GenA', 'GenB', 'GenC'], 100),
    #     'pmin': np.random.rand(100) * 100,
    #     'pmax': np.random.rand(100) * 100,
    #     'Forecast': np.random.rand(100) * 100
    # })

    # prepare a DataFrame with given columns
    # df = pd.DataFrame(columns=['time_stamp', 'Generator_name', 'generation', 'pmin', 'pmax', 'Forecast'])
    df = pd.DataFrame(
        columns=[
            "generator_uid",
            "timestamp",
            "actual_pg",
            "fcst_pg",
            "zone_uid",
            "fuel_type",
        ]
    )

    for constraint_uid in all_generators["uid"]:
        url = f"/{run}/cluster_generation.csv?uid={constraint_uid}&from_date=2024-06-15&to_date=2024-06-20&resample_rate=1h"
        this_generator = _get_dataframe(url)
        if this_generator is not None:
            this_generator["generator_uid"] = constraint_uid
            # change name of columns from 'time_stamp' to 'timestamp':
            this_generator.rename(columns={"time_stamp": "timestamp"}, inplace=True)
            this_generator.rename(columns={"generation": "actual_pg"}, inplace=True)
            # drop the column "pmin":
            this_generator.drop(columns=[["pmin", "pmax"]], inplace=True)

            this_generator["timestamp"] = all_generators[
                all_generators["uid"] == constraint_uid
            ]["fuel_type"].values[0]

            df = pd.concat([df, this_generator], ignore_index=True)
        else:
            print(f"Failed to get data for generator with uid: {constraint_uid}")
        print(this_generator)

    # Sample DataFrame
    # Assume df is a DataFrame with columns: ['Time', 'Generator', 'Pmax', 'Benchmark', 'Forecast']
    # Here is a mock DataFrame structure for example purposes
    # df = pd.DataFrame({
    #     'Time': pd.date_range(start='1/1/2023', periods=100, freq='H'),
    #     'Generator': np.random.choice(['GenA', 'GenB', 'GenC'], 100),
    #     'Pmax': np.random.rand(100) * 100,
    #     'Benchmark': np.random.rand(100) * 100,
    #     'Forecast': np.random.rand(100) * 100
    # })

    # Function to calculate error metrics
    def calculate_error_metrics(df):
        df["Error"] = df["Forecast"] - df["Benchmark"]
        df["Absolute_Error"] = np.abs(df["Error"])
        df["Squared_Error"] = df["Error"] ** 2
        df["Percentage_Error"] = np.abs(df["Error"] / df["Benchmark"]) * 100

        mae = df["Absolute_Error"].mean()
        mse = df["Squared_Error"].mean()
        rmse = np.sqrt(mse)
        mape = df["Percentage_Error"].mean()
        bias = df["Error"].mean()

        return mae, mse, rmse, mape, bias

    # Group by Generator to calculate metrics for each one
    results = df.groupby("Generator").apply(calculate_error_metrics)
    results = pd.DataFrame(
        results.tolist(),
        index=results.index,
        columns=["MAE", "MSE", "RMSE", "MAPE", "Bias"],
    )

    print(results)

    # Bias Analysis Visualization
    plt.figure(figsize=(10, 6))
    plt.bar(results.index, results["Bias"], color="skyblue")
    plt.title("Bias of Forecasts for Each Generator")
    plt.xlabel("Generator")
    plt.ylabel("Bias")
    plt.show()

    # Calculate error correlations across generators
    error_df = df.pivot(index="Time", columns="Generator", values="Error")
    error_correlation = error_df.corr()

    print("Error Correlation Matrix:")
    print(error_correlation)

    # Visualization of Error Correlation Matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(error_correlation, cmap="coolwarm", interpolation="none", aspect="auto")
    plt.colorbar()
    plt.xticks(range(len(error_correlation)), error_correlation.columns, rotation=90)
    plt.yticks(range(len(error_correlation)), error_correlation.columns)
    plt.title("Error Correlation Matrix Across Generators")
    plt.show()

    # Identify and report high correlations
    threshold = 0.7  # Define a threshold for high correlation
    high_correlation_pairs = (
        error_correlation[
            (error_correlation.abs() > threshold) & (error_correlation.abs() < 1.0)
        ]
        .stack()
        .index.tolist()
    )

    if high_correlation_pairs:
        print("Generators with high error correlations:")
        for pair in high_correlation_pairs:
            print(
                f"{pair[0]} and {pair[1]}: {error_correlation.loc[pair[0], pair[1]]:.2f}"
            )
    else:
        print("No high error correlations found between generators.")

    # constraint_uid = '08BTHLEA'
    # early_date_for_MUSE_query = '2024-06-15'
    # to_date = '2024-06-20'
    # url = f'/{run}/cluster_generation.csv?uid={constraint_uid}&from_date={early_date_for_MUSE_query}&to_date={to_date}&resample_rate=1h'
    # this_generator = _get_dataframe(url)

    print(18)
