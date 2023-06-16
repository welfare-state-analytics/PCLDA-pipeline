from collections import Counter
import click
import pandas as pd


def add_relative_weight(x, counter, column):
    return x["counts"] / counter[x[column]]


def get_weights(df, col_a, col_b):
    counter = Counter(df[col_a])
    df_counted = df.groupby([col_a, col_b]).size().reset_index(name="counts")
    df_counted["weight"] = df_counted.apply(
        add_relative_weight, axis=1, counter=counter, column=col_a
    )
    return df_counted.drop(columns=["counts"], inplace=False)


def get_document_topic_weight(df):
    col_a = "document_id"
    col_b = "topic_id"
    return get_weights(df, col_a, col_b)


def get_topic_token_weight(df):
    col_a = "topic_id"
    col_b = "token_id"
    return get_weights(df, col_a, col_b)


def rename_columns(df):
    df.rename(
        columns={"#doc": "document_id", "topic": "topic_id", "typeindex": "token_id"},
        inplace=True,
    )


def process_files(input_files):
    """Process mallet state output files and write document-topic and topic-token weights
        as tsv files to the current directory. The files will be
        named doc_topic_weights.tsv and topic_token_weights.tsv

    Args:
        input_files : list of file paths, assumed to contain mallet state output
    """
    cols = ['#doc', 'topic', "typeindex"]
    dfs = [pd.read_csv(f, sep=" ", usecols=cols) for f in input_files]
    df = pd.concat(dfs)
    rename_columns(df)

    doc_topic_weights = get_document_topic_weight(df)
    topic_token_weights = get_topic_token_weight(df)
    doc_topic_weights.to_csv("doc_topic_weights.tsv", sep="\t", index=False)
    topic_token_weights.to_csv("topic_token_weights.tsv", sep="\t", index=False)


@click.command(
    help="python topic_states/combine_states.py  statefile1 statefile2 ... statefileN"
)
@click.argument("input", nargs=-1, type=click.Path(exists=True))
def main(input):
    """python topic_states/combine_states.py  statefile1 statefile2 ... statefileN"""
    print("Combining:")
    for inp in input:
        print(inp)
    process_files(input)


if __name__ == "__main__":
    main()
