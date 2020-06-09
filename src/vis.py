import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def plot_tsne(features, labels, title=None, marker_size=30):
    
    tsne = TSNE(n_components=2, n_jobs=10).fit_transform(features)
    tsne_df = pd.DataFrame({'X':tsne[:,0],
                            'Y':tsne[:,1],
                            'label':labels})
    
    plt.figure(figsize=(10, 10))
    ax = sns.scatterplot(x="X", y="Y", hue="label",
                        palette="muted", s=marker_size,
                        data=tsne_df);
    if title:
        ax.set_title(title, fontsize=16)