import pandas as pd


def get_hyper_parameters(experiment_name: str, method: str, filename: str):
    method = method.lower()
    all_methods = ['mlp', 'lstm', 'tcn']
    if method not in all_methods:
        raise ValueError(f"method type '{method}' is not supported, choose one from {all_methods}")

    _types = {'dataset': str, 'experiment_name': str, 'method': str,
              'lambda1': float, 'batch_size': int, 'wd': float, 'hidden_nodes': int,
              'learning_rate': float, 'hl': int, 'maxlags': int}

    df = pd.read_csv(filename, sep='\t', header=0, dtype=_types)
    return df[(df.experiment_name == experiment_name) & (df.method == method)].iloc[0].to_dict()
