# cvprproject
CVPR Project for the final exam


# Usage
The file `main.py` contains most of the exercise. The `dict`s at the beginning can be used to select which parts to trian and which to load from files once trained (when set to train, the models will automatically be saved to a file).

The file `ensemble.py` contains the code to generate the networks in the ensemble, afterwards, running `ensemblepredict.py` will perform predictions on the networks trained with the first file and calculate the accuracy.

Other files contain functions used to make the code in the aforementioned files easier to read.

The MATLAB file `transferalexnet.m` will perform transfer learning using the Alexnet architecture.

The MATLAB file `trainsvmr.m` will train SVM decision tree classifiers in MATLAB.

The file `writetolibsvmfile.m` will write the activations of a selected layer in the Alexnet architecture to a file formatted for use with LIBSVM. The activations over the training and test dataset from a fully connected layer are given in the the two .txt files. The LIBSVM command line tool was used as follows:
````
svm-train -s 0 -t 0 trainlibsvmr.txt linmodelfile
````