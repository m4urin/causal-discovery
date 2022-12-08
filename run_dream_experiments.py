from src.utils import get_hyper_parameters
from train_NAVAR import train_NAVAR
from evaluate import calculate_AUROC, dream_file_to_causal_matrix
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='Train NAVAR on DREAM3 or CauseMe data')
parser.add_argument('--experiment', metavar='experiment', type=str, help='name of the experiment (e')
parser.add_argument('--evaluate', action='store_true')
parser.add_argument('--method', metavar='experiment', type=str, help=f"choose one of ['mlp', 'lstm', 'tcn']")

args = parser.parse_args()
experiment = args.experiment
method = args.method

hyper_params = get_hyper_parameters(experiment, method, 'experiments/hyper_parameters.tsv')

# load the data
file = f'experiments/data/{experiment}.tsv'
ground_truth_file = f'experiments/data/{experiment}_gt.txt'
data = pd.read_csv(file, sep='\t')
data = data.values[:, 1:]
epochs = 5000

# start training
print(f"Starting training on the data from experiment {experiment}, training for {epochs} iterations.")
score_matrix, _, _ = train_NAVAR(data, **hyper_params, dropout=0, epochs=epochs, val_proportion=0.0,
                                 check_every=500, normalize=True, split_timeseries=False)
# evaluate
print('Done training!')
if args.evaluate:
    ground_truth_matrix = dream_file_to_causal_matrix(ground_truth_file)
    AUROC = calculate_AUROC(score_matrix, ground_truth_matrix, ignore_self_links=True)
    print(f"The AUROC of this model on experiment {experiment} is: {AUROC}")
