#!/usr/bin/python3
import matplotlib.pyplot as plt
import itertools
import numpy as np
from PIL import Image

import semantic.utils


def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax


def plot_confusion_matrix(confusion_matrix, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        confusion_matrix = confusion_matrix.astype(
            'float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(confusion_matrix)

    thresh = confusion_matrix.max() / 2.
    for i, j in itertools.product(range(confusion_matrix.shape[0]), range(confusion_matrix.shape[1])):
        plt.text(j, i, confusion_matrix[i, j],
                 horizontalalignment="center",
                 color="white" if confusion_matrix[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def calculate_per_class_iou(predicted, ground_truth, class_details):

    width = np.shape(predicted)[1]
    height = np.shape(predicted)[0]

    ground_truth = semantic.utils.resize_array(
        ground_truth, max(np.shape(predicted)))
    ground_truth = semantic.utils.zero_pad_array(ground_truth, width, height)

    ious = {}

    for class_detail in class_details:

        if class_detail['name'] == 'BG':
            continue

        class_truth = ground_truth == class_detail['id']
        class_predicted = predicted == class_detail['id']

        if area(class_truth) == 0:
            continue

        iou = calculate_iou(class_predicted, class_truth)

        ious[class_detail['name']] = iou

    return ious


def area(a):
    return len(np.where(a == True)[0])


def calculate_iou(a, b):
    area_of_intersection = area(np.logical_and(a, b))
    area_of_union = area(np.logical_or(a, b))

    return area_of_intersection / area_of_union


def evaluate(model, dataset):

    number_of_samples = len(dataset.image_ids)

    image_ids = np.random.choice(
        dataset.image_ids, number_of_samples, replace=False)

    ious = {}
    mious = {}

    for info in dataset.class_info:
        if info['name'] == 'BG':
            continue
        ious[info['name']] = []
        mious[info['name']] = []

    for image_id in image_ids:
        test_image = Image.open(dataset.image_reference(image_id))
        test_mask = dataset.load_pil_mask(image_id, as_array=True)
        prediction = model.predict(test_image)

        iou = calculate_per_class_iou(
            prediction, test_mask, dataset.class_info)

        for item, value in iou.items():
            ious[item].append(value)

    for item, value in ious.items():
        mious[item].append(np.mean([value]))

    return mious
