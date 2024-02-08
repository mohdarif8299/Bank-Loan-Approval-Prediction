test_logisticdata = readtable('testdata_logistic.csv');
load('logisticmodel.mat');

% Make predictions on the test data
lr_predictions = predict(bestLRModel, test_logisticdata);
% Evaluate accuracy

accuracy = sum(lr_predictions == ytest) / numel(ytest);
disp(['Logitic Regression Accuracy: ', num2str(accuracy)]);
