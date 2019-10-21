#### Deep(er) look into Facial landmarks detection-A semantic segmentation based approach.



1. [Dataset Creation](https://github.com/KaunilD/facs-segmentation/blob/master/dataset-creation/csv2png.py):
   1. Facial (uncropped and unaligned) images obtained from FER2013 dataset (ds).
   2. 64 Landmarks for each of the images from ds the have been extracted using [dlib](http://dlib.net/) library.
2. [Model Training](https://github.com/KaunilD/facs-segmentation/blob/master/deep-learning/trainer.py):
   1. Pairs of images (grayscale) and their respective landmark masks are input to the condensed [SegNet](https://arxiv.org/abs/1511.00561) model for pixel level segmentation.
   2. [Condensed implementation]() has fewer layers and smaller filter sizes to prevent overfitting of a relatively lower dimensional dataset as compared to the one discussed in original implementation.



###### Initial Results: Epoch 6

| output                                                       | input                                                        |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![1.png](https://github.com/KaunilD/facs-segmentation/deep-learning/126.png) | ![](https://github.com/KaunilD/facs-segmentation/dataset-creation/data_gt/target/2.png) |
| ![](https://github.com/KaunilD/facs-segmentation/deep-learning/2.png) | ![](https://github.com/KaunilD/facs-segmentation/dataset-creation/data_gt/target/127.png) |
