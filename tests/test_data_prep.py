from src.data_prep import load_data


def test_load_data():
    df = load_data("data/raw/data1.csv")
    assert not df.empty
    assert "Churn" in df.columns
