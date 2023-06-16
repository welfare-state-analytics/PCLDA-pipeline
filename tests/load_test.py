import pandas as pd

from pclda_pipeline.convert import to_document_topic_weights, to_topic_token_weights, combine_weights


def test_to_document_topic_weights():
    data: pd.DataFrame = to_document_topic_weights("pclda_pipeline/z_250.csv")
    assert data is not None
    assert set(data.columns) == set(["document_id", "topic_id", "weight"])


def test_to_topic_token_weights():
    data: pd.DataFrame = to_topic_token_weights("pclda_pipeline/z_250.csv")
    assert data is not None
    assert set(data.columns) == set(["topic_id", "token_id", "weight"])


def test_combine_to_document_topic_weights():
    data: list[pd.DataFrame] = [
        to_document_topic_weights(f"pclda_pipeline/{filename}.csv") for filename in ["z_190", "z_240", "z_250"]
    ]
    document_topic_weights: pd.DataFrame = combine_weights(["document_id", "topic_id"], *data)
    assert document_topic_weights is not None
    assert set(document_topic_weights.columns) == set(["document_id", "topic_id", "weight"])


def test_combine_to_topic_token_weights():
    data: list[pd.DataFrame] = [
        to_topic_token_weights(f"pclda_pipeline/{filename}.csv") for filename in ["z_190", "z_240", "z_250"]
    ]
    topic_token_weights: pd.DataFrame = combine_weights(["topic_id", "token_id"], *data)
    assert topic_token_weights is not None
    assert set(topic_token_weights.columns) == set(["topic_id", "token_id", "weight"])
