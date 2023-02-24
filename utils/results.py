import pandas as pd


def save_as_df(dict_list, path: str) -> None:
    df = pd.DataFrame.from_dict(dict_list).round(3)  # create dataframe
    df.to_pickle(path)  # save file
    print(df)
    print('Result saved.')


def load_df(path: str) -> pd.DataFrame:
    return pd.read_pickle(path)
