import numpy as np
# import tensorflow as tf
from tensorflow import keras
import cv2


def loss_func(y_true, y_pred):
    loss = -1 * (keras.backend.square(1 - y_pred) * y_true * keras.backend.log(keras.backend.clip(y_pred, keras.backend.epsilon(), 1)) +
                 keras.backend.square(y_pred) * (1 - y_true) * keras.backend.log(keras.backend.clip(1 - y_pred, keras.backend.epsilon(), 1)))
    return keras.backend.mean(loss)


class TrackNet():

    def __init__(self):
        self.__model = keras.models.load_model(
            "model/TrackNetv2.h5", custom_objects={'custom_loss': loss_func})
        self.__traj = []

    def __pre_process(self, data):
        data_size = data.shape[0]
        padd_size = 0
        if data_size < 3:
            data = np.append(data, [data[-1]], 0)
            if data_size < 2:
                data = np.append(data, [data[-1]], 0)
            padd_size = 3 - data_size
        elif data_size % 3 != 0:
            data = np.append(data[:data_size - (data_size % 3)], data[-3:], 0)
            padd_size = (3 - (data_size % 3)) % 3
        data = ((np.reshape(
            data, (-1, 9, data.shape[2], data.shape[3]))).astype('float32')) / 255.0
        return data, padd_size

    def __post_process(self, pred, pad_size):
        pred = np.reshape(
            pred, (pred.shape[0] * pred.shape[1], pred.shape[2], pred.shape[3]))
        if pad_size != 0:
            pred = np.delete(pred, np.s_[-3:-3+pad_size], 0)

        pred = pred > 0.5
        pred = pred.astype('uint8')
        for heatmap in pred:
            if np.amax(heatmap) == 0:
                self.__traj.append([0., 0., 0.])
                continue
            contours, _ = cv2.findContours(
                heatmap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            rects = [cv2.boundingRect(contour) for contour in contours]
            max_area = 0
            target = [0, 0, 0, 0]
            for rect in rects:
                area = rect[2] * rect[3]
                if area > max_area:
                    target = rect
            self.__traj.append(
                [1., target[0] + target[2] / 2.0, target[1] + target[3] / 2.0])

    def run(self, data, batch_size=4):
        data, pad_size = self.__pre_process(data)
        pred = self.__model.predict(data, batch_size=batch_size)
        self.__post_process(pred, pad_size)

    def get_result(self):
        traj = self.__traj
        self.__traj = []
        return np.array(traj)
