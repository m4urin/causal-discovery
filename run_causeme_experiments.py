from src.utils import get_hyper_parameters
from train_NAVAR import train_NAVAR
import numpy as np
import argparse
import zipfile
import json
import bz2

parser = argparse.ArgumentParser(description='Train NAVAR on CauseMe data')
parser.add_argument('--experiment', metavar='experiment', type=str, help='name of the experiment (e')
parser.add_argument('--method_sha', metavar='method_sha', type=str, help='name of the experiment (e')
parser.add_argument('--method', metavar='experiment', type=str, help=f"choose one of ['mlp', 'lstm', 'tcn']")

args = parser.parse_args()
experiment = args.experiment
method_sha = args.method_sha
method = args.method

hyper_params = get_hyper_parameters(experiment, method, 'experiments/hyper_parameters.tsv')

# prepare results file
results = {}
results["method_sha"] = method_sha
results["parameter_values"] = f"maxlags: {hyper_params['maxlags']}"
results['model'] = experiment.split('_')[0]
results['experiment'] = experiment
results_file = f'experiments/results/{experiment}.json.bz2'
scores = []

# load the data
file = f'experiments/data/{experiment}.zip'
with zipfile.ZipFile(file, "r") as zip_ref:
    datasets = sorted(zip_ref.namelist())
    for dataset in datasets:
        print(f"Training NAVAR on: {dataset}")
        data = np.loadtxt(zip_ref.open(dataset))
        # start training NAVAR
        score_matrix, _, _ = train_NAVAR(data, **hyper_params, dropout=0, epochs=2000,
                                         val_proportion=0.0, check_every=200,
                                         normalize=True, split_timeseries=False)
        scores.append(score_matrix.flatten())
        break

# Save data
print('Writing results ...')
results['scores'] = np.array(scores).tolist()
results_json = bytes(json.dumps(results), encoding='latin1')
with bz2.BZ2File(results_file, 'w') as mybz2:
    mybz2.write(results_json)
