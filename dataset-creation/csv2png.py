import numpy as np
import pandas as pd
import cv2
import dlib

DS_PATH = '../data/fer2013/fer2013.csv'
DEBUG = True

class Data:
    """
        Initialize the Data utility.
        :param data:
                    a pandas DataFrame containing data from the
                    FER2013 dataset.
        :type file_path:
                    DataFrame
        class variables:
        _x_train, _y_train:
                    Training data and corresopnding labels
        _x_test, _y_test:
                    Testing data and corresopnding labels
        _x_valid, _y_validation:
                    Validation/Development data and corresopnding labels

    """
    def __init__(self, data, shape_model):
        self._images, self._target = [], []
        self._data = data
        self._face_location = dlib.rectangle(0, 0, 48, 48)
        self._shape_predictor = dlib.shape_predictor(shape_model)


    def get_images(self, save=False, out_dir=None):
        assert not(save ^ bool(out_dir)), "Missing output directory in save mode"
        for xdx, x in enumerate(self._data.values):
            pixels = []
            label = None
            for idx, i in enumerate(x[1].split(' ')):
                pixels.append(int(i))

            pixels = np.array(pixels, dtype=np.uint8).reshape((48, 48))
            face_mask = np.zeros((48, 48))

            shape = self._shape_predictor(pixels, self._face_location)
            np_shape = self.shape_to_np(shape)

            for i in np_shape:
                if ( i[1] < 48 and i[1] > 0 ) and ( i[0] < 48 and i[0] > 0 ):
                    face_mask[i[1]][i[0]] = 255

            self._images.append(pixels)
            self._target.append(face_mask)

            if save:
                cv2.imwrite("{}/input/{}.png".format(out_dir, xdx), pixels)
                cv2.imwrite("{}/target/{}.png".format(out_dir, xdx), face_mask)
            
    def shape_to_np(self, shape, dtype="int"):

    	coords = np.zeros((shape.num_parts, 2), dtype=dtype)
    	for i in range(0, shape.num_parts):
    		coords[i] = (shape.part(i).x, shape.part(i).y)

    	return coords


def main():
    data = pd.read_csv(DS_PATH)
    data = Data(
        data,
        shape_model="../data/models/shape_predictor_68_face_landmarks.dat"
    )

    images = data.get_images(save=DEBUG, out_dir='./data_gt')


if __name__ == "__main__":
    main()
