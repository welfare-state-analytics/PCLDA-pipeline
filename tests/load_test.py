import pandas as pd

from pclda_pipeline.convert import to_document_topic_weights

def test_load2():
    data: pd.DataFrame = to_document_topic_weights("pclda_pipeline/z_250.csv")
    assert data is not None

