import os
import jsonlines
import pandas as pd
import matplotlib.pyplot as plt
from scripts.compute_params import compute_params


def plot_loss(output_dir):
    data = []
    with jsonlines.open(os.path.join(output_dir, 'metrics.json')) as reader:
        for obj in reader:
            if 'metrics/accuracy' not in obj.keys():
                data.append(obj)

    df = pd.json_normalize(data)
    plt.plot(df['iteration'], df['total_loss'], label='train')
    plt.xlabel('iteration #, 1 epoch = %i iterations' %
               compute_params(os.path.join(output_dir, 'config.yaml'))['one_epoch'])
    plt.ylabel('loss')
    plt.legend()
    total_hours = (df['eta_seconds'].max() - df['eta_seconds'].min())/60/60
    plt.title("Model loss, total training time %.2f hours" % total_hours)
    plt.savefig(os.path.join(output_dir, 'train_loss.png'))


if __name__ == '__main__':
    output = "../output/output_dir"
    plot_loss(output)
