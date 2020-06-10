import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from os.path import dirname, join
from sklearn.manifold import TSNE
import pickle
import json
import logging

def get_labels(sets, data_path):
    """
    Combine all `simple_labels.json` files into one dict.
    
    Parameters
    ----------
    sets : list of tuples
        tuples of kind ('country', 'region')
    data_path : str
        root path containing country folders
        
    Returns
    -------
    labels_dict : dict
        dict with entries of kind 'id':'label'
    
    """

    labels_dict = {}
    for country, region in sets:
        try:
            with open(join(data_path, country, region, 'roofs_train', 'simple_labels.json')) as label_file:
                labels_region = json.load(label_file)
            labels_dict.update(labels_region)
        except:
            print(f"Error reading labels for region {region}")

    return labels_dict


def get_features(sets, feature_path, model_name, dim, pooling_method):
    """
    Combine all features into one tensor.
    
    Parameters
    ----------
    sets : list of tuples
        tuples of kind ('country', 'region')
    feature_path : str
        path containing features
    model_name : str
        name of the model used to compute features
    dim : int
        input height or width of the model used to compute features
    pooling_method : str
        pooling method at last layer of the model used to compute features
    
    Returns
    -------
    features : ndarray
        Tensor containing features of all specified sets of regions
    
    """
    
    first = True
    for _, region in sets:
        try:
            with open(join(feature_path, f'{model_name}_{dim}_{pooling_method}_{region}_train.pkl'), 'rb') as f:
                features_region = pickle.load(f)
        except:
            logging.warning(f"Error reading training features for region {region}.")
            continue
            
        if first:
            features = features_region
            first = False
        else:
            features = np.concatenate((features, features_region))
            
    return features


def labels_int(labels):
    """
    Converts a list of label strings to a list of label integers according to the materials dict.
    """
    materials = {'concrete_cement':0, 'healthy_metal':1, 'incomplete':2, 'irregular_metal':3, 'other':4}
    
    labels_int = []
    for l in labels:
        labels_int.append(materials[l])
    return labels_int


def thumbnail(image_fp, dec_factor=32):
    """
    Creates a thumbnail of a CO-GeoTIFF in the same folder and stores it as png.

    Parameters
    ----------
    image_fp
        path to the CO-GeoTIFF
    dec_factor
        decimation factor, factor by which the image should be shrinked
    """

    with rasterio.open(image_fp) as src:
        print('Decimation factor = {}'.format(dec_factor))
        new_height = int(src.height // dec_factor)
        new_width = int(src.width // dec_factor)

        b, g, r = (src.read(k, out_shape=(1, new_height, new_width)) for k in (1, 2, 3))

    # Retrieve name of folder that image is stored in
    dir = dirname(image_fp)
    with rasterio.open(join(dir, 'thumbnail.png'), 'w',
            driver='GTiff', width=r.shape[1], height=r.shape[0], count=3,
            dtype=r.dtype) as dst:
        for k, arr in [(1, b), (2, g), (3, r)]:
            dst.write(arr, indexes=k)

            
def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs, vmin=0, vmax=1)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels, fontsize=12)
    ax.set_yticklabels(row_labels, fontsize=12)
    
    #
    ax.set_ylabel('True labels', fontsize=16)
    ax.set_xlabel('Predicted labels', fontsize=16)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts
