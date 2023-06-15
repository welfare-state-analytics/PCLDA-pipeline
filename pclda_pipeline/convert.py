import pandas as pd

def to_document_topic_weights(filename: str) -> pd.DataFrame:
    """Reads a state file and returns document topic weights data frame."""
    data: pd.DataFrame = pd.read_csv(filename, sep=" ", usecols=[0, 5])
    data.columns = ["document_id", "topic_id"]

    document_topic_counts = data.groupby(["document_id", "topic_id"]).agg(
        doc_topic_count=("topic_id", "size")
    )
    document_counts = data.groupby(["document_id"]).agg(doc_count=("topic_id", "size"))

    document_topic_weights = document_topic_counts.merge(
        document_counts, left_index=True, right_index=True
    )

    document_topic_weights["weight"] = (
        document_topic_weights.doc_topic_count / document_topic_weights.doc_count
    )
    document_topic_weights.drop(["doc_topic_count", "doc_count"], axis=1, inplace=True)

    # if normalize:
    #   document_weights = document_topic_weights.groupby(["document_id"])['weight'].sum()
    #   document_topic_weights['weight'] = document_topic_weights.weight / document_weights

    return document_topic_weights