Dependency:
    apt
    pip
        opencv-contrib-python
        wget
        zip
        python3-tk
        tensorflow
        keras
        h5py
        pydot
        graphviz
        matplotlib

Resources:
    https://github.com/lakshayg/tensorflow-build
    matplotlib error - https://stackoverflow.com/questions/37604289/tkinter-tclerror-no-display-name-and-no-display-environment-variable

pydot and Graphviz issue:
    https://github.com/keras-team/keras/issues/3216
    https://github.com/pydot/pydot-ng
    https://stackoverflow.com/questions/18438997/why-is-pydot-unable-to-find-graphvizs-executables-in-windows-8
    https://graphviz.gitlab.io/download/
    
    Checklist:
        - Install Graphviz
        - Install pydot_ng
        - Import pydot_ng as pydot

Questions:
    - How to calculate network capacity and how does it affect training and accuracy?
    - How to evaluate overfitting?
    - Where to put normalization layer, before or after activation, or both?
    - How to choose parameters: number of feature maps, number of units in FC layer, rate of dropout, number of batch size and epochs?

References:
    https://navoshta.com/traffic-signs-classification/
    http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf
    https://github.com/f00-/mnist-lenet-keras
    https://github.com/bguisard/CarND-Traffic-Sign-Classifier-Project
