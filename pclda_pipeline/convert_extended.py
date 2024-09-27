import csv
import os
from datetime import datetime
from typing import Callable
import click
import pandas as pd
from loguru import logger


def to_zip(data: pd.DataFrame, filename: str, archive_name: str, **csv_opts) -> None:
    data.to_csv(filename, compression=dict(method='zip', archive_name=archive_name), **csv_opts)


def compute_weights(filename: str, target_columns: tuple[str, str]) -> pd.DataFrame:
    """Reads a state file and returns document topic weights data frame."""
    source2target = {"#doc": 'document_id', "topic": "topic_id", "typeindex": "token_id"}
    target2source = {v: k for k, v in source2target.items()}

    data: pd.DataFrame = pd.read_csv(
        filename, sep=" ", usecols=list(map(target2source.get, target_columns)), quoting=csv.QUOTE_NONE
    )
    data.columns = [source2target.get(column, column) for column in data.columns]

    group_counts = data.groupby(target_columns).agg(group_count=(target_columns[1], "size"))
    total_counts = data.groupby(target_columns[0]).agg(total_count=(target_columns[1], "size"))

    weights = group_counts.merge(total_counts, left_index=True, right_index=True)

    weights["weight"] = weights.group_count / weights.total_count
    weights.drop(["group_count", "total_count"], axis=1, inplace=True)

    return weights.reset_index() 

#### function which is basically copy of compute_weights but computes counts instead of proportions #####
def compute_counts(filename: str, target_columns: tuple[str, str]) -> pd.DataFrame:
    """Reads a state file and returns document topic counts data frame."""
    source2target = {"#doc": 'document_id', "topic": "topic_id", "typeindex": "token_id"}
    target2source = {v: k for k, v in source2target.items()}

    data: pd.DataFrame = pd.read_csv(
        filename, sep=" ", usecols=list(map(target2source.get, target_columns)), quoting=csv.QUOTE_NONE
    )
    data.columns = [source2target.get(column, column) for column in data.columns]

    group_counts = data.groupby(target_columns).agg(group_count=(target_columns[1], "size"))
    total_counts = data.groupby(target_columns[0]).agg(total_count=(target_columns[1], "size"))

    weights = group_counts.merge(total_counts, left_index=True, right_index=True)

    weights["weight"] = weights.group_count #/ weights.total_count # commented out only this part, to not calculate proportions but totals.
    weights.drop(["group_count", "total_count"], axis=1, inplace=True)

    return weights.reset_index()

########### function which is basically copy of to_document_topic_weights but returns data frame of counts instead of proportions
def to_document_topic_counts(filename: str) -> pd.DataFrame:
    """Reads a state file and returns document-topic weights dataframe."""
    weights = compute_counts(filename, target_columns=["document_id", "topic_id"]) # compute_counts here instead of compute_weights
    return weights

def to_topic_type_counts(filename: str) -> pd.DataFrame:
    """Reads a state file and returns a topic-token weights dataframe."""
    weights = compute_counts(filename, target_columns=["topic_id", "token_id"])
    return weights

########## function which is basically combine_weights but instead sums up all the counts and averages over the number of dataframes that are merged

def combine_counts(target_columns: tuple[str, str], *data: list[pd.DataFrame]) -> pd.DataFrame:
    #Creates a combined doc-topic dataframe with average counts.
    # Combine all the dataframes
    combined_data = pd.concat(data)
    
    #group_counts = combined_data

    # Group by the target columns and calculate the sum of the counts
    group_counts = combined_data.groupby(list(target_columns)).agg(count=("weight", "sum"))
    
    # Calculate the number of dataframes
    num_dataframes = len(data)
    
    # Calculate the average counts and rename the column to "n_z"
    group_counts["n_z"] = group_counts["count"] / num_dataframes
    
    # Drop the original "count" column
    group_counts.drop(columns=["count"], inplace=True)
    
    # Reset the index to make sure target columns are part of the dataframe
    return group_counts.reset_index()


### function which is "to_combined_weights" but for counts instead 
def to_combined_counts(target_folder: str, *filenames: list[str]) -> None:
    config = {
        "document_topic_sufficient_statistics": (to_document_topic_counts, ["document_id", "topic_id"]),
        "topic_type_sufficient_statistics": (to_topic_type_counts, ["topic_id", "token_id"]),
    }

    dataframes: list[pd.DataFrame] = []

    for target, (to_fx, target_columns) in config.items():
        logger.info(f"Computing: {target} {' '.join(filenames)}")
        data: list[pd.DataFrame] = [to_fx(filename) for filename in filenames]

        logger.info(f"Combining: {' '.join(filenames)}")

        weights: pd.DataFrame = combine_counts(target_columns, *data)

        dataframes.append(weights)

        logger.info(f"Storing result in folder: {target_folder}")

        to_zip(weights, f"{target_folder}/{target}.zip", f"{target}.csv", sep='\t', index=True)

    return (dataframes[0], dataframes[1])


########################################################

def combine_weights(target_columns: tuple[str, str], *data: list[pd.DataFrame]) -> pd.DataFrame:
    """Creates a compined and weighed doc-topic dataframe."""
    group_weights = pd.concat(data).groupby(list(target_columns)).agg(weight=("weight", "sum"))
    total_weights = group_weights.groupby(target_columns[0])["weight"].sum()

    group_weights["weight"] = group_weights.weight / total_weights

    return group_weights.reset_index()


def to_document_topic_weights(filename: str) -> pd.DataFrame:
    """Reads a state file and returns document-topic weights dataframe."""
    weights = compute_weights(filename, target_columns=["document_id", "topic_id"])
    return weights

def to_topic_type_weights(filename: str) -> pd.DataFrame:
    """Reads a state file and returns a topic-token weights dataframe."""
    weights = compute_weights(filename, target_columns=["topic_id", "token_id"])
    return weights

# topic_type_weights is now renamed to topic_type_weights
def to_combined_weights(target_folder: str, *filenames: list[str]) -> None:
    config = {
        "document_topic_weights": (to_document_topic_weights, ["document_id", "topic_id"]),
        "topic_type_weights": (to_topic_type_weights, ["topic_id", "token_id"]),
    }

    dataframes: list[pd.DataFrame] = []

    for target, (to_fx, target_columns) in config.items():
        logger.info(f"Computing: {target} {' '.join(filenames)}")
        data: list[pd.DataFrame] = [to_fx(filename) for filename in filenames]

        logger.info(f"Combining: {' '.join(filenames)}")

        weights: pd.DataFrame = combine_weights(target_columns, *data)

        dataframes.append(weights)

        logger.info(f"Storing result in folder: {target_folder}")

        to_zip(weights, f"{target_folder}/{target}.zip", f"{target}.csv", sep='\t', index=True)

    return (dataframes[0], dataframes[1])


def to_dictionary(target_folder: str, *filenames: list[str]) -> None:
    dictionary: pd.DataFrame = None
    for filename in filenames:
        data: pd.DataFrame = pd.read_csv(filename, sep=" ", usecols=[3, 4], quoting=csv.QUOTE_NONE)
        data.columns = ["token_id", "token"]
        data = data.drop_duplicates('token_id').set_index('token_id')
        if dictionary is None:
            dictionary = data
        else:
            data = data[~data.index.isin(dictionary.index)]
            if len(data) > 0:
                dictionary = pd.concat([dictionary, data[~data.index.isin(dictionary.index)]])
    dictionary['dfs'] = 0

    to_zip(
        dictionary, f"{target_folder}/dictionary.zip", "dictionary.csv", sep='\t', index=True, quoting=csv.QUOTE_MINIMAL
    )

    return dictionary


def to_top_types_per_topic(
    target_folder: str, topic_type_weights: pd.DataFrame, dictionary: pd.DataFrame, n_tokens: int = 500
) -> pd.DataFrame:
    tx: Callable[[int, str], str] = dictionary['token'].to_dict().get
    overview: pd.DataFrame = (
        topic_type_weights.groupby('topic_id')
        .apply(lambda x: sorted(list(zip(x["token_id"], x["weight"])), key=lambda z: z[1], reverse=True))
        .apply(lambda x: ' '.join([str(tx(z[0], "")) for z in x][:n_tokens]))
        .reset_index()
    )
    overview.columns = ['topic_id', 'tokens']
    overview['alpha'] = 0.0

    to_zip(
        overview.set_index('topic_id'),
        f"{target_folder}/top_types_per_topic.zip",
        "top_types_per_topic.csv",
        sep='\t',
        index=True,
        quoting=csv.QUOTE_MINIMAL,
    )

    return overview


@click.command(help="python topic_states/convert.py  statefile_1 statefile_2 ... statefile_n")
@click.argument("filenames", nargs=-1, type=click.Path(exists=True))
@click.option("--target-folder", type=str, default=None)
def main(filenames, target_folder: str = None):
    print("Running extended convert-script also calculating sufficient statistics")
    try:
        if target_folder is None:
            target_folder = f"tm-bundle-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        print(f"Combining: {' '.join(filenames)}")

        os.makedirs(target_folder, exist_ok=True)

        _, topic_type_weights = to_combined_weights(target_folder, *filenames)
        _, output = to_combined_counts(target_folder, *filenames) # new line to call the count functions within main(), but not passing any output to to_top_types_per_topic
        dictionary: pd.DataFrame = to_dictionary(target_folder, *filenames)
        to_top_types_per_topic(target_folder, topic_type_weights, dictionary, 500)
    except Exception as e:
        logger.exception(e)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter

    # from click.testing import CliRunner

    # runner: CliRunner = CliRunner()
    # result = runner.invoke(
    #     main,
    #     [
    #         "pclda_pipeline/z_190.csv",
    #         "pclda_pipeline/z_240.csv",
    #         "pclda_pipeline/z_250.csv",
    #     ],
    # )