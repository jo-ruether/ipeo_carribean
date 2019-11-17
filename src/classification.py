import pickle
from os.path import join
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.multiclass import unique_labels

def collect_all_features():
    regions = ['borde_rural', 'borde_soacha', 'castries',
               'dennery', 'gros_islet', 'mixco_3', 'mixco_1_and_ebenezer']

    for region in regions:
        pickle_dir = join('..', '..', 'pickles')
        feature_fp = join(pickel_dir,
                          'resnet50_feature_matrix_' + region + '.pkl')
