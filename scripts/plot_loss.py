import os
import re
import jsonlines
import pandas as pd
import matplotlib.pyplot as plt
from scripts.compute_params import compute_params


def plot_loss(output_dir):
    data = []
    with jsonlines.open(os.path.join(output_dir, 'metrics.json')) as reader:
        for obj in reader:
            if 'total_loss' in obj.keys():
                data.append(obj)

    log = open(os.path.join(output_dir, "log.txt"), 'r')
    training_time = re.search('time:(.+?) \(', log.readlines()[-1]).group(1)

    df = pd.json_normalize(data)
    plt.plot(df['iteration'], df['total_loss'], label='total loss')
    plt.plot(df['iterlsation'], df['validation_loss'], label='validation loss')
    plt.xlabel('iteration #, 1 epoch = %i iterations' %
               compute_params(os.path.join(output_dir, 'config.yaml'))['one_epoch'])
    plt.legend()
    plt.title("Model loss, total training time%s hours" % training_time)
    plt.savefig(os.path.join(output_dir, 'model_loss.png'))


if __name__ == '__main__':
    output = "../output/output_dir"
    plot_loss(output)
