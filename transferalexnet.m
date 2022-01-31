%% Introduction
% This program will access the activations of an inner layer of the Alexnet
% architecture, and then train an SVM based on those activations. It will
% then perform predictions on the test dataset.

% For reproducibility purposes, we set the seed.
rng(1)
% First, build the datastore
traindatasetPath = fullfile('data','train');
testdatasetPath = fullfile('data','test');
imdsbase = imageDatastore(traindatasetPath, ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');
[imdsTrain,imdsVal] = splitEachLabel(imdsbase,0.85,'randomized');
imdsTest = imageDatastore(testdatasetPath, ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');

% Get the labels
YTrain = imdsTrain.Labels;
YTest = imdsTest.Labels;
%% Import the Alexnet network with the pretrained weights

net = alexnet();

%% Build the model
notrain = net.Layers(1:end-3);

% Set to untrainable
for iind=1:numel(notrain)
    % Check if they are trainable
    if isprop(notrain(iind),"WeightLearnRateFactor")
       notrain(iind) = notrain(iind).setLearnRateFactor("Weights",0);
    end
    if isprop(notrain(iind),"BiasLearnRateFactor")
       notrain(iind) = notrain(iind).setLearnRateFactor("Bias",0);
    end
end

% Now finally stack everything up
newlayers = [notrain,
  fullyConnectedLayer(15);
  softmaxLayer("Name","prob");
  classificationLayer("Name","class","Classes",unique(YTrain));
];

analyzeNetwork(newlayers);
%% Preprocess images so they can be fed to the network
timdsTrain = augmentedImageDatastore(newlayers(1).InputSize(1:2),imdsTrain,"ColorPreprocessing","gray2rgb");
timdsVal = augmentedImageDatastore(newlayers(1).InputSize(1:2),imdsVal,"ColorPreprocessing","gray2rgb");
timdsTest = augmentedImageDatastore(newlayers(1).InputSize(1:2),imdsTest,"ColorPreprocessing","gray2rgb");

%% Training options
myoptions = trainingOptions("sgdm","MaxEpochs",1000,"MiniBatchSize",32,"ValidationData",timdsVal,"InitialLearnRate",1e-4,ValidationPatience=15,ExecutionEnvironment="parallel",Shuffle='every-epoch',Plots='training-progress');

%% Train
mynet = trainNetwork(timdsTrain,newlayers,myoptions)

%% Classify and score
[YPred,scores] = classify(mynet,timdsTest);