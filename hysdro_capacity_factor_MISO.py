# environment: placebo_jupyter_env_new_python

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


market_name = "miso"
# market_name = "spp"

this_dir = os.path.dirname(os.path.abspath(__file__))


# find all the files in the directory that start with 'all_hydro_data'
def find_files(directory):
    files = []
    for file in os.listdir(directory):
        if file.startswith(f"all_hydro_data_{market_name}"):
            files.append(file)
    return files


# Function to load the data
def load_data(directory, files):
    data = pd.DataFrame()
    for file in files:
        df = pd.read_csv(directory + "/" + file)
        data = pd.concat([data, df])
    return data


# Function to clean the data
def clean_data(data):
    data = data.drop_duplicates()
    data = data.dropna()
    return data


files = find_files(this_dir)
df = load_data(this_dir, files)
df = clean_data(df)

# all_results = df.groupby('month').agg({'capacity_factor':'mean'}).reset_index().sort_values('month')

all_units_monthly_capacity_factor = pd.DataFrame(
    columns=["name", "unit_id", "month", "capacity_factor"]
)
list_of_monthly_capacity_factor = []

all_units = df[["name", "unit_id"]].drop_duplicates().reset_index(drop=True)
for unit in all_units.values:
    unit_data = df.query("unit_id == @unit[1] and name == @unit[0]").sort_values(
        "case_converted"
    )
    aggregated_monthly = (
        unit_data.groupby("month")
        .agg({"capacity_factor": "mean"})
        .reset_index()
        .sort_values("month")
    )

    # for months that are missing, fill in the missing values with the closest month
    for month in range(1, 13):
        if month not in aggregated_monthly["month"].values:
            aggregated_monthly = pd.concat(
                [
                    aggregated_monthly,
                    pd.DataFrame(
                        {
                            "month": [month],
                            "capacity_factor": [
                                aggregated_monthly["capacity_factor"].mean()
                            ],
                        }
                    ),
                ]
            )

    aggregated_monthly["name"] = unit[0]
    aggregated_monthly["unit_id"] = unit[1]

    aggregated_monthly.sort_values("month", inplace=True)
    all_units_monthly_capacity_factor = pd.concat(
        [all_units_monthly_capacity_factor, aggregated_monthly]
    )

    list_of_monthly_capacity_factor.append(aggregated_monthly["capacity_factor"].values)

    # for dev purposes:
    if False:
        if aggregated_monthly.capacity_factor.min() < -0.2:
            plt.figure(figsize=(15, 11))
            plt.plot(
                aggregated_monthly["month"],
                aggregated_monthly["capacity_factor"],
                label="Capacity Factor",
            )
            plt.scatter(
                aggregated_monthly["month"],
                aggregated_monthly["capacity_factor"],
                label="Capacity Factor",
            )
            plt.legend()
            plt.xlabel("Timestamp")
            plt.ylabel("Capacity Factor")
            plt.title(
                f'Capacity Factor for Generator {unit[0]}: {unit[1]}. min/max: {round(aggregated_monthly["capacity_factor"].min(),1)} / {round(aggregated_monthly["capacity_factor"].max(),1)}'
            )
            plt.xticks(rotation=45)  # Rotate x-axis ticks
            plt.tight_layout()  # Adjust layout to prevent clipping of tick-labels
            plt.close()

print(18)
# find the mean of each row in list_of_monthly_capacity_factor. This is the mean for each ['name', 'unit_id']
mean_capacity_factor = np.mean(list_of_monthly_capacity_factor, axis=1)
# remove the mean_capacity_factor from each row in list_of_monthly_capacity_factor
list_of_monthly_capacity_factor = (
    list_of_monthly_capacity_factor - mean_capacity_factor[:, np.newaxis]
)

u, s, v = np.linalg.svd(list_of_monthly_capacity_factor, full_matrices=False)

# Keep only the first 3 singular values
s[2:] = 0

# Reconstruct the matrix using the first 3 singular values
list_of_monthly_capacity_factor2 = np.dot(u, np.dot(np.diag(s), v))

# add mean_capacity_factor to each row in list_of_monthly_capacity_factor2
clean_capacity_per_month = (
    list_of_monthly_capacity_factor2 + mean_capacity_factor[:, np.newaxis]
)

# Create a DataFrame from clean_capacity_per_month and all_units
clean_capacity_df = pd.DataFrame(clean_capacity_per_month, columns=range(1, 13))

# aggregated_monthly['capacity_factor_clean'] must not be less than 0 or more than 1
# clean_capacity_df = clean_capacity_df.apply(lambda x: 0 if x <= 0 else x)
clean_capacity_df = clean_capacity_df.applymap(lambda x: 0 if x < 0 else x)
clean_capacity_df = clean_capacity_df.applymap(lambda x: 1 if x > 1 else x)

clean_capacity_df["unit_id"] = [unit[1] for unit in all_units.values]
clean_capacity_df["name"] = [unit[0] for unit in all_units.values]

file_name = f"clean_capacity_{market_name.upper()}.csv"
clean_capacity_df.to_csv(file_name, index=False)

# Melt the DataFrame to long format
clean_capacity_df = clean_capacity_df.melt(
    id_vars=["unit_id", "name"], var_name="month", value_name="capacity_factor_clean"
)

# Merge with the original DataFrame
df = df.merge(clean_capacity_df, on=["unit_id", "name", "month"], how="left")

# Plotting for dev pruporses:
for unit in all_units.values:
    unit_data = df.query("unit_id == @unit[1] and name == @unit[0]").sort_values(
        "case_converted"
    )
    aggregated_monthly = (
        unit_data.groupby("month")
        .agg({"capacity_factor": "mean", "capacity_factor_clean": "mean"})
        .reset_index()
        .sort_values("month")
    )

    # in places where aggregated_monthly['capacity_factor'] < 0, aggregated_monthly['capacity_factor_clean'] must also be 0
    # aggregated_monthly['capacity_factor_clean'] = np.where(aggregated_monthly['capacity_factor'] <= 0, 0, aggregated_monthly['capacity_factor_clean'])

    # aggregated_monthly['capacity_factor_clean'] must not be less than 0 or more than 1
    # aggregated_monthly['capacity_factor_clean'] = aggregated_monthly['capacity_factor_clean'].apply(lambda x: 0 if x <= 0 else x)
    # aggregated_monthly['capacity_factor_clean'] = aggregated_monthly['capacity_factor_clean'].apply(lambda x: 1 if x > 1 else x)

    # if True:
    #     if aggregated_monthly.capacity_factor.min() < -.2:
    plt.figure(figsize=(15, 11))
    plt.plot(
        aggregated_monthly["month"],
        aggregated_monthly["capacity_factor"],
        label="Capacity Factor",
    )
    plt.scatter(
        aggregated_monthly["month"],
        aggregated_monthly["capacity_factor"],
        label="Capacity Factor",
    )
    plt.plot(
        aggregated_monthly["month"],
        aggregated_monthly["capacity_factor_clean"],
        label="Capacity Factor Clean",
    )
    plt.legend()
    plt.xlabel("Timestamp")
    plt.ylabel("Capacity Factor")
    plt.title(
        f'Capacity Factor for Generator {unit[0]}: {unit[1]}. min/max: {round(aggregated_monthly["capacity_factor"].min(),1)} / {round(aggregated_monthly["capacity_factor"].max(),1)}'
    )
    plt.xticks(rotation=45)  # Rotate x-axis ticks
    plt.tight_layout()  # Adjust layout to prevent clipping of tick-labels

    # draw horizontals line at 0 and 1
    plt.axhline(y=1, color="r", linestyle="--")
    plt.axhline(y=0, color="r", linestyle="--")
    plt.close()


if False:

    # df = df[df['fuel_type'] == 'HYDRO']
    # all_units = df[['name', 'unit_id']].drop_duplicates()

    # df.loc[:, 'month'] = df['timestamp'].apply(lambda x:x[5:7])

    result = (
        df.groupby(["name", "unit_id", "month"])
        .agg({"actual_pg": "mean", "pmax_Actual": "mean"})
        .reset_index()
    )
    result["capacity_factor"] = result["actual_pg"] / result["pmax_Actual"]

    # merge result back into df
    df = df.merge(
        result[["generator_uid", "unit_id", "month", "capacity_factor"]],
        on=["generator_uid", "unit_id", "month"],
        how="left",
    ).sort_values("timestamp")

    for generator_uid in df["generator_uid"].unique():
        generator = df[df["generator_uid"] == generator_uid]
        plt.figure(figsize=(15, 11))
        plt.plot(
            generator["timestamp"],
            generator["capacity_factor"],
            label="Capacity Factor",
        )
        plt.plot(
            generator["timestamp"],
            generator["actual_pg"] / generator["pmax_Actual"],
            label="Forecast Capacity Factor",
        )
        plt.legend()
        plt.xlabel("Timestamp")
        plt.ylabel("Capacity Factor")
        plt.title(f"Capacity Factor for Generator {generator_uid}")
        plt.xticks(rotation=45)  # Rotate x-axis ticks
        plt.tight_layout()  # Adjust layout to prevent clipping of tick-labels
        plt.close()


if False:
    # Add data such as lat/lon to the dataframe
    if market_name == "miso":
        # copy the file "miso_plant_name_and_coordinates.csv" from '/Users/michael.simantov/Documents' to the same directory as the script
        if not "miso_plant_name_and_coordinates.csv" in os.listdir(this_dir):
            os.system(
                "cp /Users/michael.simantov/Documents/miso_plant_name_and_coordinates.csv ."
            )
        df_EIA_generators = pd.read_csv("miso_plant_name_and_coordinates.csv")
    elif market_name == "spp":
        if not "spp_plant_name_and_coordinates.csv" in os.listdir(this_dir):
            os.system(
                "cp /Users/michael.simantov/Documents/spp_plant_name_and_coordinates.csv ."
            )
        df_EIA_generators = pd.read_csv("spp_plant_name_and_coordinates.csv")

    df = df.join(
        df_EIA_generators.set_index("uid")[
            ["latitude", "longitude", "energy_source_code", "prime_mover_code"]
        ],
        on="generator_uid",
    )

    # Function to calculate error metrics
    def calculate_error_metrics(df):
        # if df['name'].values[0] == 'DPSF':
        #     print(18)
        df["Error"] = df["fcst_pg"] - df["actual_pg"]
        df["Absolute_Error"] = np.abs(df["Error"])
        df["Percentage_Error"] = df["Error"] / df["pmax_Actual"]
        df["Pmax_Error"] = (df["actual_pg"] - df["pmax_Forecast"]) / df[
            "pmax_Forecast"
        ]  # Pmax_diff
        df["Pmax_error_percentage"] = (
            df["Error"] / df["pmax_Actual"]
        ) * 100  # Pmax_error_percentage

        # if the actuala dn forecast are both constant 0 then we cannot say anything about Pmax error
        if df["Error"].max() == 0:
            df["Pmax_error_percentage"] = 0
            df["Pmax_Error"] = 0

        # replace with 0
        df["Percentage_Error"] = df["Percentage_Error"].replace(
            [np.inf, -np.inf, np.nan], 0
        )

        df["actual_fcst_correlation"] = df["actual_pg"].corr(df["fcst_pg"])

        mae = df["Absolute_Error"].mean()
        mpe = df["Percentage_Error"].mean()
        bias = df["Error"].mean()
        corr = df["actual_fcst_correlation"].mean()
        pmax_diff = df["Pmax_Error"].max()
        pmin_diff = df["Pmax_Error"].min()
        Pmax_error_percentage = df["Pmax_error_percentage"].max()

        # if np.log(np.abs(df['Pmax_error_percentage'].max())) > np.log(np.abs(df['Pmax_error_percentage'].min())):
        #     Pmax_error_percentage = np.log(np.abs(df['Pmax_error_percentage'].max()))
        # else:
        #     Pmax_error_percentage = np.log(np.abs(df['Pmax_error_percentage'].min()))

        # return mae, mse, rmse, mape, bias
        return mae, mpe, bias, corr, pmax_diff, pmin_diff, Pmax_error_percentage

    def count_generators(df):
        return df["generator_uid"].nunique()

    # Group by Generator to calculate metrics for each one
    results = df.groupby("generator_uid").apply(calculate_error_metrics)
    results = pd.DataFrame(
        results.tolist(),
        index=results.index,
        columns=[
            "MAE",
            "MPE",
            "Bias",
            "Corr",
            "Pmax_diff",
            "Pmin_diff",
            "Pmax_error_percentage",
        ],
    )
    results["Corr"] = results["Corr"].replace([np.inf, -np.inf, np.nan], 0)

    # add the results to the df dataframe
    df = df.join(results, on="generator_uid")

    ## 1 ##################################### Z O N E ####### M P E ##############################################
    # MPE Analysis Visualization per zone
    if False:
        for column in ["zone_uid", "fuel_type"]:
            for criterion in ["Pmax_error_percentage", "Pmax_diff"]:
                all_results = (
                    df.groupby(column)
                    .agg(
                        {
                            criterion: ["mean", "std"],
                            "Bias": ["mean", "std"],
                            "latitude": "median",
                            "longitude": "median",
                        }
                    )
                    .reset_index()
                )
                all_results.sort_values(column, inplace=True)
                all_results["row_count"] = (
                    df.groupby([column])
                    .apply(count_generators)
                    .rename("temp")
                    .reset_index()
                    .sort_values(column)["temp"]
                    .values
                )
                all_results.sort_values((criterion, "mean"), inplace=True)

                # Remove results with ('MPE','mean') == 0')
                all_results = all_results[all_results[(criterion, "mean")] != 0]
                all_results = all_results[all_results["row_count"] > 1]
                all_results = all_results[all_results[(criterion, "mean")] != np.inf]
                all_results = all_results[all_results[(criterion, "mean")] != -np.inf]
                all_results = all_results.dropna(subset=[(criterion, "mean")])

                # if criterion == 'Pmax_error_percentage':
                #     results = all_results[:10] # Take the 10 zones with the highest MPE
                # else:
                # Take the 10 zones with the highest Pmax_error_percentage
                results = all_results.reindex(
                    all_results[criterion]["mean"].abs().nlargest(10).index
                )

                # Create a figure and a set of subplots, arranged side by side
                fig, axs = plt.subplots(
                    1, 2, figsize=(32, 12), gridspec_kw={"width_ratios": [1, 2]}
                )

                # Plot 1: Bar chart for MPE on the first subplot
                axs[0].bar(
                    results[column],
                    results[criterion]["mean"],
                    yerr=results[criterion]["std"],
                    color="skyblue",
                )
                # add a horizontal line at 0:
                axs[0].axhline(y=0, color="r", linestyle="--")
                axs[0].set_title(f"{criterion} for Each {column}")
                axs[0].set_xlabel(f"{column}")
                axs[0].set_ylabel(f"{criterion}")

                # Calculate mean and standard deviation of MPE for each point
                mape_mean = results[criterion].mean()
                # mape_std = results[criterion].std()

                # Normalize the size and color to make them suitable for plotting
                size_factor = (
                    2000  # Adjust this factor to scale the sizes appropriately
                )

                # Use the mean of MPE to determine the size of the circle
                sizes = (
                    np.sqrt(np.abs(results[criterion] / np.abs(mape_mean)))
                    * size_factor
                )

                # Plot 2: Scatter plot on the second subplot
                sc = axs[1].scatter(
                    results["longitude"],
                    results["latitude"],
                    s=sizes["mean"],
                    c=sizes["std"],
                    cmap="viridis",
                    alpha=0.5,
                    edgecolors="w",
                )
                axs[1].scatter(df["longitude"], df["latitude"], sizes=[0.5] * len(df))
                axs[1].scatter(
                    df[df[column].isin(results[column])]["longitude"],
                    df[df[column].isin(results[column])]["latitude"],
                    sizes=[15] * len(df),
                )
                fig.colorbar(sc, ax=axs[1], label="{criterion} Standard Deviation")
                axs[1].set_xlabel("Longitude")
                axs[1].set_ylabel("Latitude")

                # Loop through the DataFrame and add text above each circle in the scatter plot
                for index, row in results.iterrows():
                    axs[1].text(
                        row["longitude"]["median"],
                        row["latitude"]["median"],
                        f"{row[column].values[0]}\n({row['row_count'].values[0]})",
                        fontsize=10,
                        ha="center",
                        va="bottom",
                        fontweight="bold",
                    )

                plt.suptitle(
                    f"Analysis Visualization per {column}   |   Pmax_diff = (actual_pg - pmax_Forecast) / pmax_Forecast    |    Pmax_error_percentage = max[(fcst_pg - Actual_pg) / pmax_Actual]"
                )

                plt.tight_layout(
                    rect=[0, 0.03, 1, 0.95]
                )  # Adjust the layout to make room for the suptitle

                plt.title(
                    f"Scatter plot with circle size proportional to {criterion} mean and color by {criterion} std"
                )

    print(18)

    ## 2 ########################### F U E L - Z O N E ############ M P E ############################################
    # MPE Analysis Visualization per fuel_type and zone
    if True:
        for column in [
            ("fuel_type", "zone_uid"),
            ("zone_uid", "fuel_type"),
            ("prime_mover_code", "energy_source_code"),
        ]:
            criterion_1 = column[0]
            criterion_2 = column[1]
            for Error_type in ["Pmax_error_percentage", "Pmax_diff"]:
                results = (
                    df.groupby([criterion_1, criterion_2])
                    .agg({Error_type: ["mean", "std"], "Bias": ["mean", "std"]})
                    .reset_index()
                )
                results.sort_values([criterion_1, criterion_2], inplace=True)
                results["row_count"] = (
                    df.groupby([criterion_1, criterion_2])
                    .apply(count_generators)
                    .rename("temp")
                    .reset_index()
                    .sort_values([criterion_1, criterion_2])["temp"]
                    .values
                )
                results.sort_values(
                    [criterion_1, (Error_type, "mean"), criterion_2], inplace=True
                )

                # Remove results with (Error_type,'mean') == 0')
                results = results[results[(Error_type, "mean")] != 0]
                results = results[results["row_count"] > 1]
                results = results[results[(Error_type, "mean")] != np.inf]
                results = results[results[(Error_type, "mean")] != -np.inf]
                results = results.dropna(subset=[(Error_type, "mean")])

                results.reset_index(drop=True, inplace=True)

                plt.figure(figsize=(30, 12))
                results["label"] = (
                    results[criterion_1] + " | " + results[criterion_2].astype(str)
                )
                # plt.bar(results['label'], np.sign(results[Error_type]['mean'])*np.sqrt(np.sqrt(np.abs(results[Error_type]['mean']))), yerr=np.sqrt(np.sqrt(np.abs(results[Error_type]['std']))), color='skyblue')
                fuel_types = results[criterion_1].unique()
                colors = [
                    "cyan",
                    "red",
                ]  # Define more colors if you have more than two fuel_types
                color_map = {
                    f: colors[i % len(colors)] for i, f in enumerate(fuel_types)
                }

                # Apply colors based on criterion
                bar_colors = [color_map[f] for f in results[criterion_1]]

                plt.bar(
                    results["label"],
                    np.sign(results[Error_type]["mean"])
                    * np.sqrt(np.sqrt(np.abs(results[Error_type]["mean"]))),
                    yerr=np.sqrt(np.sqrt(np.abs(results[Error_type]["std"]))),
                    color=bar_colors,
                )
                plt.xticks(rotation=45, ha="right")

                # Extract unique fuel_types to determine color change
                unique_fuel_types = results[criterion_1].unique()
                colors = [
                    "black",
                    "red",
                ]  # Define more colors if you have more than two fuel_types
                # Create a color map for alternating colors
                color_map = {
                    f: colors[i % len(colors)] for i, f in enumerate(unique_fuel_types)
                }

                # Apply alternating colors to x-axis labels based on fuel_type
                tick_labels = plt.gca().get_xticklabels()
                for label in tick_labels:
                    f = label.get_text().split(" | ")[
                        0
                    ]  # Extract criterion from the label
                    label.set_color(color_map[f])

                for index, row in results.iterrows():
                    plt.text(
                        index,
                        np.sign(row[(Error_type, "mean")])
                        * np.sqrt(np.sqrt(np.abs(row[(Error_type, "mean")]))),
                        str(row["row_count"].values),
                        ha="center",
                    )

                plt.axhline(y=0, color="r", linestyle="--")

                plt.title(
                    f"{Error_type} for Each {criterion_1}   |   Pmax_diff = (actual_pg - pmax_Forecast) / pmax_Forecast    |    Pmax_error_percentage = max[(fcst_pg - Actual_pg) / pmax_Actual]"
                )
                plt.xlabel(criterion_1)
                plt.ylabel(Error_type)

    ######### debugged up to here !  ############################

    ############################################################################################

    ############################################################################################

    # For each fuel type, create 2D chart: X is Corr and Y is MPError. Each point is a generator.
    # The size of the point is the MAE and the color is the Bias
    # The title of the chart is the fuel type
    plt.figure(figsize=(15, 11))
    for fuel_type in df["fuel_type"].unique():
        plt.figure(figsize=(15, 11))
        results = df[df["fuel_type"] == fuel_type]

        # Remove results with (Error_type,'mean') == 0')
        results = results[results["MPE"] != 0]
        # results = results[results['row_count'] > 1]
        results = results[results["MPE"] != np.inf]
        results = results[results["MPE"] != -np.inf]
        results = results.dropna(subset=["MPE"])

        this = (
            results.groupby("generator_uid")
            .agg({"Corr": "mean", "MPE": "mean", "MAE": "mean", "Bias": "mean"})
            .reset_index()
        )
        # plt.scatter(results['Corr'], results['MPE'], s=results['MPE'], c=results['Bias'], cmap='viridis', alpha=0.5, edgecolors='w')
        plt.scatter(results["Corr"], results["MPE"])
        plt.colorbar(label="Bias")
        plt.xlabel("Correlation")
        plt.ylabel("MPE")
        plt.title(f"Correlation vs. MPE for {fuel_type}")

    ############################################################################################
    # For each zone, create 2D chart: X is MPE's mean and Y is MPE's std
    # The size of the point is the MAE and the color is the Bias
    # The title of the chart is the zone
    for zone in df["zone_uid"].unique():
        results = df[df["zone_uid"] == zone]
        plt.figure(figsize=(15, 11))
        plt.scatter(
            results["MPE"],
            results["MPE"],
            s=results["MAE"],
            c=results["Bias"],
            cmap="viridis",
            alpha=0.5,
            edgecolors="w",
        )
        plt.colorbar(label="Bias")
        plt.xlabel("MPE Mean")
        plt.ylabel("MPE Std")
        plt.title(f"MPE Mean vs. MPE Std for {zone}")

    ############################################################################################

    # For each zone, create 2D chart: X is Corr and Y is MPError. Each point is a generator.
    # The size of the point is the MAE and the color is the Bias
    # The title of the chart is the zone
    for zone in df["zone_uid"].unique():
        results = df[df["zone_uid"] == zone]
        plt.figure(figsize=(15, 11))
        plt.scatter(
            results["Corr"],
            results["MPE"],
            s=results["MAE"],
            c=results["Bias"],
            cmap="viridis",
            alpha=0.5,
            edgecolors="w",
        )
        plt.colorbar(label="Bias")
        plt.xlabel("Correlation")
        plt.ylabel("MPE")
        plt.title(f"Correlation vs. MPE for {zone}")

    ############################################################################################

    # For each zone, find generators with low correlation between actual and forecast power flow
    # and high MPE
    low_corr_high_mpe = df[(df["Corr"] < 0.5) & (df["MPE"] > 10)]
    low_corr_high_mpe = low_corr_high_mpe.sort_values("MPE", ascending=False)

    if False:
        # Calculate error correlations across generators
        error_df = df.pivot(index="timestamp", columns="fcst_pg", values="MPE")
        error_correlation = error_df.corr()

        print("Error Correlation Matrix:")
        print(error_correlation)

        # Visualization of Error Correlation Matrix
        plt.figure(figsize=(8, 6))
        plt.imshow(
            error_correlation, cmap="coolwarm", interpolation="none", aspect="auto"
        )
        plt.colorbar()
        plt.xticks(
            range(len(error_correlation)), error_correlation.columns, rotation=90
        )
        plt.yticks(range(len(error_correlation)), error_correlation.columns)
        plt.title("Error Correlation Matrix Across Generators")

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
