import io
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import pandas as pd
from collections import OrderedDict


def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
  returns it. The supplied figure is closed and inaccessible after this call."""
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=200)
    plt.close(figure)
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)
    return image


def plot_xfer_heatmap(xfer_mapping):
    converted_mapping = {'xfer': [int(k) for k in xfer_mapping.keys()], 'count': [v for _, v in xfer_mapping.items()]}
    # sorted_xfers = OrderedDict(sorted(xfer_mapping.items(), key=lambda x: int(x[0])))
    df = pd.DataFrame.from_dict(converted_mapping)
    # df_formatted = df.pivot('graph', 'xfer', 'count')
    sns.set_theme(style='darkgrid')
    ax = sns.barplot(x='xfer', y='count', data=df)
    return ax.get_figure()
