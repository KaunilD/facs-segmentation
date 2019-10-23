import numpy as np
import pandas as pd
import cv2
import dlib
import glob
DS_PATH = '../data/lfw-deepfunneled'
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
        self._face_location = dlib.rectangle(0, 0, 250, 250)
        self._face_detector = dlib.get_frontal_face_detector()
        self._shape_predictor = dlib.shape_predictor(shape_model)


    def get_images(self, save=False, out_dir=None):
        assert not(save ^ bool(out_dir)), "Missing output directory in save mode"
        for xdx, x in enumerate(self._data):
            pixels = cv2.imread(x)
            #pixels = self.resize(pixels, width=224, height=224)
            face_mask = np.zeros(pixels.shape[:2])

            face_location = self._face_detector(pixels, 1)
            for (i, rect) in enumerate(face_location):

                shape = self._shape_predictor(pixels, rect)
                shape = self.shape_to_np(shape)

                (x, y, w, h) = self.rect_to_bb(rect)


                for (i, j) in shape:
                    cv2.circle(face_mask,(i, j), 3, (255), -1)

                image = pixels[x:x+w, y:y+h, :]
                face_mask = face_mask[x:x+w, y:y+h]
                #print(image.size, face_mask.size)
                if w < 100 and h < 100 or image.size < 34668 or face_mask.size < 11556:
                    print("continuing")
                    continue

                self._images.append(image)
                self._target.append(face_mask)

                if save:
                    cv2.imwrite("{}/input/{}.png".format(out_dir, xdx), image)
                    cv2.imwrite("{}/target/{}.png".format(out_dir, xdx), face_mask)


    def shape_to_np(self, shape, dtype="int"):

    	coords = np.zeros((shape.num_parts, 2), dtype=dtype)
    	for i in range(0, shape.num_parts):
    		coords[i] = (shape.part(i).x, shape.part(i).y)

    	return coords

    def rect_to_bb(self, rect):
    	x = rect.left()
    	y = rect.top()
    	w = rect.right() - x
    	h = rect.bottom() - y

    	return (x, y, w, h)

    def resize(self, image, width=None, height=None, inter=cv2.INTER_AREA):
        dim = None
        (h, w) = image.shape[:2]

        if width is None and height is None:
            return image

        if width is None:
            r = height / float(h)
            dim = (int(w * r), height)
        else:
            r = width / float(w)
            dim = (width, int(h * r))

        resized = cv2.resize(image, dim, interpolation=inter)

        return resized


def main():
    data = glob.glob(DS_PATH+"/*/*.jpg")
    data = Data(
        data,
        shape_model="../data/models/shape_predictor_68_face_landmarks.dat"
    )

    images = data.get_images(save=DEBUG, out_dir='./data_gt_lfw')


if __name__ == "__main__":
    main()
