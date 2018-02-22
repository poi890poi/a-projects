https://www2.cs.duke.edu/courses/fall17/compsci527/notes/hog.pdf
https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/
https://www.programcreek.com/python/example/84776/cv2.HOGDescriptor
https://www.slideshare.net/nahiduzzamanrose/thesis-defence-51416596
https://docs.opencv.org/2.4/doc/tutorials/ml/introduction_to_svm/introduction_to_svm.html
HOG detectMultiScale parameters explained https://www.pyimagesearch.com/2015/11/16/hog-detectmultiscale-parameters-explained/
Histograms of Oriented Gradients for Human Detection https://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf
Face Detection: Histogram of Oriented Gradients and Bag of Feature Method http://worldcomp-proceedings.com/proc/p2013/IPC4143.pdf
Classifier comparison http://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
cv::ml::SVM Class Reference https://docs.opencv.org/3.3.0/d1/d2d/classcv_1_1ml_1_1SVM.html

SIFT
HOG
SURF
HAAR


    svm.train(samples, cv2.ml.ROW_SAMPLE, labels)
    labels must be numpy.int32

pip install memory_profiler

48, 72

2018-02-13 15:12

HNM iteration 02

total samples: 3577
positive samples: 1114
negative samples: 2463
true positive: 974
true negative: 2366
false positive: 97
false negative: 140
correct: 0.9337433603578418
detection rate: 0.874326750448833
error: 0.06625663964215824

HNM iteration 03

total samples: 3577
positive samples: 1114
negative samples: 2463
true positive: 1002
true negative: 2328
false positive: 135
false negative: 112
correct: 0.9309477215543752
detection rate: 0.8994614003590664
error: 0.06905227844562482

(same dataset as last iteration) Increase window size = 96x64

total samples: 3577
positive samples: 1114
negative samples: 2463
true positive: 1008
true negative: 2348
false positive: 115
false negative: 106
correct: 0.9382163824433883
detection rate: 0.9048473967684022
error: 0.061783617556611686

Trained with unnormalized positive samples and RBF

GridSearch

total samples: 3563
positive samples: 1100
negative samples: 2463
true positive: 1011
true negative: 2456
false positive: 7
false negative: 89
correct: 0.9730564131349986
detection rate: 0.9190909090909091
error: 0.026943586865001402

RandomForestClassifier + RandomizedSearch, depth 3

total samples: 3563
positive samples: 1100
negative samples: 2463
true positive: 946
true negative: 2463
false positive: 0
false negative: 154
correct: 0.9567779960707269
detection rate: 0.86
error: 0.043222003929273084

RandomForestClassifier + RandomizedSearch, depth 4

total samples: 3563
positive samples: 1100
negative samples: 2463
true positive: 949
true negative: 2463
false positive: 0
false negative: 151
correct: 0.9576199831602582
detection rate: 0.8627272727272727
error: 0.04238001683974179