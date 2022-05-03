import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def save_plot(fig, path="plots", tight_layout=True, fig_extension="png", resolution=400):
    """
        Function for saving figures and plots
        :arg
            1. fig: label of the figure
            2. path (optional): output path of the figure
    """
    img_path = os.path.join(".", path)
    os.makedirs(img_path, exist_ok=True)
    fig_path = os.path.join(img_path, fig + "." + fig_extension)

    print("Saving figure...", fig)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(fig_path, format=fig_extension, dpi=resolution)
    print("figure can be found in: ", path)


def save_data(df, name=None, path='./dataset/swat_data/', file_extension='csv'):
    os.makedirs(path, exist_ok=True)
    file_path = os.path.join(path, name + '.' + file_extension)

    df.to_csv(file_path)

