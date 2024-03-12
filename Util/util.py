import pandas as pd


def dict_to_df(dict, total_accuracies=None):
    df = pd.DataFrame.from_dict(dict).transpose()

    # Add one to columns so that they start from 1.
    df.columns += 1
    df["AVG"] = df.mean(axis=1)

    if total_accuracies:
        df["Total AVG"] = total_accuracies

    return df.round(3)


def get_analysis_df(dict):
    df = pd.DataFrame.from_dict(dict).transpose()
    df.columns = ['Training Time', 'Parameter Count']
    return df