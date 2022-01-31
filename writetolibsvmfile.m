%% Save to libsvm
% This MATLAB file will access the activations of Alexnet and then save the
% activations in a file type suitable for training a multiclass classifier
% using the SVM tool libsvm.

% For reproducibility purposes, we set the seed.
rng(1)
% First, build the datastore
traindatasetPath = fullfile('data','train');
testdatasetPath = fullfile('data','test');
imdsTrain = imageDatastore(traindatasetPath, ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');
imdsTest = imageDatastore(testdatasetPath, ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');

% Get the labels
YTrain = imdsTrain.Labels;
YTest = imdsTest.Labels;

% From categorical to integers
YTrainint = grp2idx(YTrain);
YTestint = grp2idx(YTest);

uniqueLabels  = unique(YTrain);
nclasses = length(uniqueLabels);
%% Import the Alexnet network with the pretrained weights

net = alexnet();

%analyzeNetwork(net);

%% Resizing the inputs
% The input to the network is a 227x227x3 image, but we are only
% considering 64x64x1 images. So the images need to be resized.

inputSize = net.Layers(1).InputSize;

% Training dataset
resizedTrainDataset = augmentedImageDatastore(inputSize(1:2),imdsTrain,"ColorPreprocessing","gray2rgb");
% Test dataset
resizedTestDataset = augmentedImageDatastore(inputSize(1:2),imdsTest,"ColorPreprocessing","gray2rgb");

%% Now get the activations
% We'll get the activations from the last convulational layer
layer = "conv5"; % change to fc7 to get the 4096 features of the last convulational layer
featuresTrain = activations(net,resizedTrainDataset,layer,'OutputAs','rows');
featuresTest = activations(net,resizedTestDataset,layer,'OutputAs','rows');
labels = unique(YTrain);
%% Rescale, it is important
featuresTrain = rescale(featuresTrain,-1,1);
featuresTest = rescale(featuresTest,-1,1);
%% Train part
tmp = size(featuresTrain);
nfeats = tmp(2);
nobs = tmp(1);
% Start building the text file, first convert categorical to integer,
% remembering the order
% for jind=1:length(labels)
%     tmpmatrix(:,jind) = jind.*(labels(jind) == YTrain);
% end
% finalarr = zeros(nobs,1);
% for jind = 1:length(labels)
%     finalarr = finalarr + tmpmatrix(:,jind);
% end
% clear tmpmatrix
%%
% First, define the format
fspec = "%d "; % Label
for kind=1:nfeats
    fspec = append(fspec,"%d:");
    fspec = append(fspec,"%.8f"," ");
end
%%
fileID = fopen('trainlibsvmr.txt','w');
for iind=1:nobs
    fprintf(fileID,"%d ",YTrainint(iind));
    for jind=1:nfeats
        fprintf(fileID,"%d:%.8f ",jind,featuresTrain(iind,jind));
    end
    fprintf(fileID,"\n",[]);
end
%% Test part
tmp = size(featuresTest);
nfeats = tmp(2);
nobs = tmp(1);
% Start building the text file, first convert categorical to integer,
% remembering the order
% for jind=1:length(labels)
%     tmpmatrix(:,jind) = jind.*(labels(jind) == YTest);
% end
% finalarr = zeros(nobs,1);
% for jind = 1:length(labels)
%     finalarr = finalarr + tmpmatrix(:,jind);
% end
% clear tmpmatrix
%%
% First, define the format
fspec = "%d "; % Label
for kind=1:nfeats
    fspec = append(fspec,"%d:");
    fspec = append(fspec,"%.8f"," ");
end
%%
fileID = fopen('testlibsvmr.txt','w');
for iind=1:nobs
    fprintf(fileID,"%d ",YTestint(iind));
    for jind=1:nfeats
        fprintf(fileID,"%d:%.8f ",jind,featuresTest(iind,jind));
    end
    fprintf(fileID,"\n",[]);
end

%% Now write to file the actual classes in the test set, for testing purposes
fileID = fopen('actualclasses.txt','w');
numids = grp2idx(YTest);
for iind = 1:length(YTest)
    fprintf(fileID,"%d\n",numids(iind));
end