from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input

import numpy as np
from os.path import join, exists, isfile
from os import makedirs, listdir, walk

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib
from sklearn.decomposition import PCA

import pickle

def extract_features_training(region):

    roof_dir = join('..', '..', 'data', region, 'roofs_train')
    materials = {'healthy_metal':0, 'irregular_metal':1,
                 'concrete_cement':2, 'incomplete':3, 'other':4}

    model = ResNet50(weights='imagenet', include_top=False, pooling='max')
    #model.summary()

    # Count the number of roofs
    nof_roofs = 0;
    for material in materials.keys():
        material_fp = join(roof_dir, material)
        nof_this = len([name for name in listdir(material_fp) if isfile(join(material_fp, name))])
        nof_roofs = nof_roofs + nof_this

    labels = []
    resnet50_feature_matrix = np.zeros((nof_roofs, 2048), dtype=float)

    # Walk through all images
    i = 0;
    for material in materials.keys():
        material_fp = join(roof_dir, material)
        for root, dirs, files in walk(material_fp):
            for file in files:
                img_fp = join(material_fp, file)
                label = materials[material]
                labels.append(label)
                print(img_fp, "labeled as ", material, ":", label)

                # Pad if size is too small, preprocess
                img = image.load_img(img_fp, target_size=(224, 224))
                img_data = image.img_to_array(img)
                img_data = np.expand_dims(img_data, axis=0)
                img_data = preprocess_input(img_data)

                # Compute features
                resnet50_feature = model.predict(img_data)
                resnet50_feature_np = np.array(resnet50_feature)
                resnet50_feature_matrix[i] = resnet50_feature_np.flatten()
                i = i + 1

    return resnet50_feature_matrix, labels

def extract_features_test(region):

    roof_dir = join('..', '..', 'data', region, 'roofs_test')

    model = ResNet50(weights='imagenet', include_top=False, pooling='max')
    #model.summary()

    # Count the number of roofs
    nof_roofs = len([name for name in listdir(roof_dir) if isfile(join(roof_dir, name))])

    resnet50_feature_matrix = np.zeros((nof_roofs, 2048), dtype=float)

    # Walk through all images
    i = 0;

    for root, dirs, files in walk(roof_dir):
        for file in files:
            img_fp = join(roof_dir, file)

            # Pad if size is too small, preprocess
            img = image.load_img(img_fp, target_size=(224, 224))
            img_data = image.img_to_array(img)
            img_data = np.expand_dims(img_data, axis=0)
            img_data = preprocess_input(img_data)

            # Compute features
            resnet50_feature = model.predict(img_data)
            resnet50_feature_np = np.array(resnet50_feature)
            resnet50_feature_matrix[i] = resnet50_feature_np.flatten()
            i = i + 1

    return resnet50_feature_matrix


def plot_tSNE(features, labels=None, number_of_materials=5):
    if labels is None:
        labels = np.zeros((features.shape[0]))

    # Visualization_
    #pca_object = PCA(n_components=50)
    #pca_features = pca_object.fit_transform(features)
    tsne_features = TSNE(n_components=2).fit_transform(features)

    # define the colormap
    cmap = plt.cm.jet
    # extract all colors from the .jet map
    cmaplist = [cmap(i) for i in range(cmap.N)]
    # create the new map
    cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)

    # define the bins and normalize
    bounds = np.linspace(0, number_of_materials, number_of_materials + 1)
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    print(tsne_features.shape)

    # Plot D-Vectors
    plt.figure()
    scat = plt.scatter(tsne_features[:, 0], tsne_features[:, 1], c=labels, cmap=cmap, norm=norm)
    cb = plt.colorbar(scat, spacing='proportional', ticks=bounds)
