% Load the test data
test_randomforestdata = readtable('testdata_randomforest.csv');
load('optimizedrandomforest.mat');

% Make predictions on the test data
rf_predictions = predict(bestRFModel, test_randomforestdata);

% Evaluate accuracy
accuracy = sum(rf_predictions == y_test) / numel(y_test);
disp(['Random Forest Accuracy: ', num2str(accuracy)]);
