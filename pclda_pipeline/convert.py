import pandas as pd


def compute_weights(filename: str, target_columns: tuple[str, str]) -> pd.DataFrame:
    """Reads a state file and returns document topic weights data frame."""
    source2target = {"#doc": 'document_id', "topic": "topic_id", "typeindex": "token_id"}
    target2source = {v: k for k, v in source2target.items()}

    data: pd.DataFrame = pd.read_csv(filename, sep=" ", usecols=list(map(target2source.get, target_columns)))
    data.columns = [source2target.get(column, column) for column in data.columns]

    group_counts = data.groupby(target_columns).agg(group_count=(target_columns[1], "size"))
    total_counts = data.groupby(target_columns[0]).agg(total_count=(target_columns[1], "size"))

    weights = group_counts.merge(total_counts, left_index=True, right_index=True)

    weights["weight"] = weights.group_count / weights.total_count
    weights.drop(["group_count", "total_count"], axis=1, inplace=True)

    return weights.reset_index()


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


def to_topic_token_weights(filename: str) -> pd.DataFrame:
    """Reads a state file and returns a topic-token weights dataframe."""
    weights = compute_weights(filename, target_columns=["topic_id", "token_id"])
    return weights
