%% Introduction
% This program will access the activations of an inner layer of the Alexnet
% architecture, and then train an SVM based on those activations. It will
% then perform predictions on the test dataset.

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
layer = "conv5";
featuresTrain = activations(net,resizedTrainDataset,layer,'OutputAs','rows');
featuresTest = activations(net,resizedTestDataset,layer,'OutputAs','rows');
%% Train an SVM
% An SVM using the one against rest with decision tree will be trained.
% In order to avoid a possible bad split at the top of the tree, several
% SVMs will be randomly trained, each with a random order of classes (this is achieved by randomly picking which class will be the split at the first level of the tree). When predicting, the final prediction 
% will be given as a result of a vote of the trained SVMs. This is a sort
% of random forest but applied to SVMs.

% First, declare the array which will contain the trained SVMs.
numberofSVMs = 30;
% Useful later
numclasses = 15;
trainoptions = statset("UseParallel",true);

% Get the number of labels (it's 15 in the set we're considering)
uniqueLabels = unique(YTrain);
numberofLabels = length(uniqueLabels);
if numberofLabels ~= 15
    error("Wrong number of labels!")
end

lineartemplatesvm = templateSVM("KernelFunction","polynomial",'PolynomialOrder',6);
% Main loop: this will train all the SVMs
for iind = 1:(numberofSVMs)
%     % Pick a label: this will be the  top split
%     topSplit = uniqueLabels(randi(numberofLabels,1));
%     % Now train the cascade of SVMs (local loop)
%  The Matlab function for training a multiclass SVM will do a decision
%  tree with the left branches being in the order of the ClassNames array
%  that's passed as argument. So at each run of the loop, we pass a random
%  permutation of the array. (option 'ordinal' means decision tree).
    % To make sure that the first split is always different, we generate
    % random permutations until the first element of the array is of the
    % class we want
    wantedclass = mod(iind,numclasses) + 1;
    iterationLabels = uniqueLabels(randperm(numberofLabels,numberofLabels));
    while (iterationLabels(1) ~= uniqueLabels(wantedclass))
        iterationLabels = uniqueLabels(randperm(numberofLabels,numberofLabels));
    end
    % Now train the SVM
    % try with learners linear, good results
    trainedSvm = fitcecoc(featuresTrain,YTrain, ...
        'Coding','ordinal','Learners',templateLinear,'ClassNames',iterationLabels,'Options',trainoptions);
    SvmCell(iind) = {trainedSvm};
end

%% Prediction
% Using the previously gathered test set, we predict the classes by voting
% Build an array containing predictions and true label

numtopredict = size(featuresTest);
numtopredict = numtopredict(1);
result = table(0,uniqueLabels(1),uniqueLabels(2),numberofSVMs);
result.Properties.VariableNames = {'ObservationNumber' 'PredictedLabel' 'TrueLabel' 'NumberofVotes'}

% Number of correctly classified observations
totalCorrect = 0;
%[testsmv,numVotes] = voteSvms(featuresTest(1,:),p,uniqueLabels);
for kind=1:numtopredict
    [vote,numVotes] = voteSvms(featuresTest(kind,:),SvmCell,uniqueLabels);
    trueLab = YTest(kind);
    result(kind,:) = table(kind,vote,trueLab,numVotes);
    totalCorrect = totalCorrect+ (trueLab == vote);
end
accuracy= totalCorrect/numtopredict

%% Optional: save a sample of the resulting table to a LaTeX file
% Couldn't make it work
% Uses the function latexTable by  Author:       Eli Duenisch
% Contributor:  Pascal E. Fortin.
% 
% mytable = result(randperm(numtopredict,6),:);
% inputdata(:,1) = string(mytable.ObservationNumber);
% inputdata(:,2) = string(mytable.PredictedLabel);
% inputdata(:,3) = string(mytable.TrueLabel);
% inputdata(:,4) = string(mytable.NumberofVotes);
% newtable = table(mytable.ObservationNumber,inputdata(:,2),inputdata(:,3),mytable.NumberofVotes);
% input.data = newtable;
% input.tableColLabels = result.Properties.VariableNames;
% input.makeCompleteLatexDocument = 0;
% mylatex = latexTable(input);
%% Save the important parts to file
save("trainsvmresultactual.mat","SvmCell","accuracy","result")
%%
function [vote,numberofVotes] = voteSvms(inputData,svmModels,labels)
    % Returns a tuple [a,b] where [a] is a categorical giving the label
    % that received the most votes, while [b] is an integer containing the
    % number of votes that [a] received.
    % svmModels: cell array containing the trained SVMs (multi-class)
    numberSvms = length(svmModels);
    numLabels = length(labels);
        % Sum prediction arrays, or rather: build matrix (columns are labels)
    % and then sum column by column
    predictionMatrix = zeros(numberSvms,numLabels);
    for iind = 1:numberSvms
        prediction = predict(svmModels{iind},inputData);
        % Set all elements in the row to zero unless th
        predictionMatrix(iind,:) = (labels == prediction);
    end
    % Now compute the predictions:
    for jind=1:numLabels
        completePrediction(jind) = sum(predictionMatrix(:,jind));
    end
    [numberofVotes,prevote] = max(completePrediction);
    vote = labels(prevote); % Needs to be categorical
end