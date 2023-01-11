import os
import shutil

import torch
from flask import Flask, request, jsonify
from PIL import ImageFile
import numpy as np
import multiprocessing as mp
from skorch import NeuralNetClassifier
import albumentations
from albumentations import pytorch
from torch.utils.data import DataLoader
from Graphs import Graphs
from DenseNet161 import DenseNet161
from flask_cors import CORS

from Messages import Messages

ImageFile.LOAD_TRUNCATED_IMAGES = True

DATASET_PATH = 'images'

flask_app = Flask(__name__)
CORS(flask_app)


def prepare_datasets(dataset_dir, dataset_files):
    data_transforms = albumentations.Compose([
        albumentations.Resize(224, 224),
        albumentations.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        pytorch.transforms.ToTensorV2()
    ])

    dataset = Graphs(dataset_dir, dataset_files, transform=data_transforms)

    return dataset


@flask_app.route("/process_img", methods=['POST'])
def process_image():
    file = request.files['image']
    image_path = DATASET_PATH + '/' + file.filename.replace('.jpg', '')
    os.mkdir(image_path)
    file.save(image_path + '/' + file.filename)

    dataset_dir = image_path
    dataset_files = os.listdir(dataset_dir)
    dataset = prepare_datasets(dataset_dir, dataset_files)
    data_loader = DataLoader(dataset, batch_size=32, pin_memory=True, num_workers=mp.cpu_count())

    densenet_is_a_chart = NeuralNetClassifier(
        module=DenseNet161,
        module__output_features=2,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    densenet_is_a_chart.initialize()
    densenet_is_a_chart.load_params(f_params='models/is_a_chart_densenet161_04.pkl')

    class_names_pred = ['just_image', 'chart']
    pred_classes = np.array([])
    for images, labels in iter(data_loader):
        pred_classes = np.append(pred_classes, densenet_is_a_chart.predict(images))

    if class_names_pred[int(pred_classes[0])] == 'just_image':
        shutil.rmtree(image_path)
        return Messages.no_chart
    else:
        densenet_decept_kind = NeuralNetClassifier(
            module=DenseNet161,
            module__output_features=6,
            device="cuda:0" if torch.cuda.is_available() else "cpu",
        )
        densenet_decept_kind.initialize()
        densenet_decept_kind.load_params(f_params='models/type_of_lie_densenet161_02.pkl')

        class_names_pred_2 = ['chart', 'srtAxY', 'trdStck', '3dpie', 'colNoScl', 'twoYaxes']
        pred_classes = np.array([])
        for images, labels in iter(data_loader):
            pred_classes = np.append(pred_classes, densenet_decept_kind.predict(images))

        shutil.rmtree(image_path)
        return {
            'chart': Messages.just_chart,
            'srtAxY': Messages.srtAxY,
            'trdStck': Messages.trdStck,
            '3dpie': Messages.threedpie,
            'colNoScl': Messages.colNoScl,
            'twoYaxes': Messages.twoYaxes
        }.get(class_names_pred_2[int(pred_classes[0])])


@flask_app.route("/alive", methods=['GET'])
def alive():
    return jsonify({'msg': 'It works'})


if __name__ == "__main__":
    flask_app.run(debug=True)
