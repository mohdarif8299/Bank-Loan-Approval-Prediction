% Loading Original Dataset
df = readtable('bank_data.csv');
% Checking for NaN Values
sum(ismissing(df))
% Cleaning the income_annual column and filling with mean
mean_income = mean(df.income_annual(df.income_annual > 0));
df.income_annual = fillmissing(df.income_annual, 'constant', mean_income);
% Cleaning the residential_assets_value as it contains some negative values
% Extract 'residential_assets_value' column
residential_assets_value = df.residential_assets_value;
positive_values = residential_assets_value(residential_assets_value > 0);
% Calculate mean of positive values
mean_residential_assets_value = mean(positive_values);
% Replace values less than or equal to 0 with the mean value
residential_assets_value(residential_assets_value <= 0) = mean_residential_assets_value;
% Update the table with the modified column
df.residential_assets_value = residential_assets_value;
% Cleaning bank assets
% Extract 'bank_asset_value' column
bank_asset_value = df.bank_asset_value;
% Calculating the mean of positive values
mean_bank_asset_value = mean(bank_asset_value(bank_asset_value > 0));
% Finding indices of missing values
missing_values = isnan(bank_asset_value);
% Filling missing values with the mean value
bank_asset_value(missing_values) = mean_bank_asset_value;
% Updating the table with the modified column
df.bank_asset_value = bank_asset_value;
sum(ismissing(df))
% Dropping Irrelevant features
df(:, {'Var1', 'id'}) = [];
% Doing Label Encoding
df.education = grp2idx(df.education);
df.self_employed = grp2idx(df.self_employed);
% Mapping Loan Status with 1 and 0
loan_status = df.loan_status;
loan_status = strrep(loan_status, 'Approved', '1');
loan_status = strrep(loan_status, 'Rejected', '0');
% Convert the column to numeric values
df.loan_status = str2double(loan_status);

summary(df)

% Applying Random Forest
% Calculate the number of test samples
numTestSamples = floor(0.2 * height(df));

% Randomly shuffle the rows of the dataframe
rng(42);
shuffledIdx = randperm(height(df));
df_shuffled = df(shuffledIdx, :);

% Split the data
X_train = df_shuffled(1:end-numTestSamples, ~strcmp('loan_status', df.Properties.VariableNames));
X_test = df_shuffled(end-numTestSamples+1:end, ~strcmp('loan_status', df.Properties.VariableNames));
y_train = df_shuffled.loan_status(1:end-numTestSamples);
y_test = df_shuffled.loan_status(end-numTestSamples+1:end);

% Convert tables to matrices for compatibility with TreeBagger
X_train_matrix = table2array(X_train);

% Train the Random Forest
rf_classifier = TreeBagger(100, X_train_matrix, y_train, ...
                           'Method', 'classification', ...
                           'OOBPrediction', 'On', ...
                           'MinLeafSize', 1, ...
                           'OOBPredictorImportance', 'on');

% Convert X_test to matrix
X_test_matrix = table2array(X_test);

% Make predictions
[y_pred, ~] = predict(rf_classifier, X_test_matrix);
y_pred = str2double(y_pred);

% Convert y_test to matrix if it's not already
%y_test_matrix = table2array(y_test);

% Calculate accuracy
accuracy = sum(y_pred == y_test) / length(y_test);

% Display results
disp(['Random Forest Accuracy Before Feature Engineering: ', num2str(accuracy)]);


% Will Do Feature Engineering by removing top two columns
% Calculate the correlation matrix
correlation_matrix = corrcoef(table2array(df), 'rows', 'pairwise');
[~, idx] = sort(abs(correlation_matrix(:, strcmp(df.Properties.VariableNames, 'loan_status'))), 'descend');
top_2_columns = df.Properties.VariableNames(idx(2:3));
% Remove these top 2 columns from the dataset
df(:, top_2_columns) = [];
%Display the modified table
%summary(df)


% Random Forest After Feature Engineering
% Calculate the number of test samples
numTestSamples = floor(0.2 * height(df));

% Randomly shuffle the rows of the dataframe
rng('default'); % For reproducibility
shuffledIdx = randperm(height(df));
df_shuffled = df(shuffledIdx, :);

% Split the data
X_train = df_shuffled(1:end-numTestSamples, ~strcmp('loan_status', df.Properties.VariableNames));
X_test = df_shuffled(end-numTestSamples+1:end, ~strcmp('loan_status', df.Properties.VariableNames));
y_train = df_shuffled.loan_status(1:end-numTestSamples);
y_test = df_shuffled.loan_status(end-numTestSamples+1:end);

% Convert tables to matrices for compatibility with TreeBagger
X_train_matrix = table2array(X_train);

% Train the Random Forest
% Train the Random Forest
rf_classifier = TreeBagger(100, X_train_matrix, y_train, ...
                           'Method', 'classification', ...
                           'OOBPrediction', 'On', ...
                           'MinLeafSize', 1, ...
                           'OOBPredictorImportance', 'on');

% Convert X_test to matrix
X_test_matrix = table2array(X_test);

% Make predictions
[y_pred, scores] = predict(rf_classifier, X_test_matrix);
y_pred = str2double(y_pred);

% Calculate accuracy
accuracy = sum(y_pred == y_test) / length(y_test);

% Display results
disp(['Random Forest Accuracy: ', num2str(accuracy)]);

% Optimizing the Random Forest
% Defining Range of Parameters
numTreesGrid = [10, 50, 100, 200];  % Number of trees

% Initialize vector to store accuracy for each number of trees
accuracyResults = zeros(1, length(numTreesGrid));
bestRFAccuracy = 0;
bestRFModel = [];
bestNumTrees = 0;
rf_training_time = 0;
rf_predict_time = 0;
time=0;
% Start timing
tic;

% Applying Grid Search 
for i = 1:length(numTreesGrid)
     tic;
     model = fitcensemble(X_train, y_train, ...
                          'Method', 'Bag', ...
                          'NumLearningCycles', numTreesGrid(i));
     rf_time = toc;
     rf_training_time = max(time, rf_training_time);
     tic;
     y_pred_train = predict(model, X_train);
     [y_pred,score] = predict(model, X_test);
     accuracy = sum(y_pred == y_test) / numel(y_test);
     rf_predict = toc;
     rf_predict_time = max(rf_predict_time, rf_predict);
     accuracyResults(i) = accuracy;
 
     % Updating best model if current model is better
     if accuracy > bestRFAccuracy
         bestRFAccuracy = accuracy;
         bestRFModel = model;
         bestNumTrees = numTreesGrid(i);
     end
end

% Stop timing
elapsedTime = toc;
training_error = sum(y_pred_train ~= y_train) / length(y_train);
test_error = sum(y_pred ~= y_test) / length(y_test);

% Display the best parameters and model accuracy
fprintf('Best Model Parameters:\n');
fprintf('NumLearningCycles: %d\n', bestNumTrees);

disp(['Optimized Random Forest Accuracy: ', num2str(accuracy)]);
fprintf('Grid search took %.2f seconds.\n', elapsedTime);
fprintf('Training time took %.2f seconds.\n', rf_training_time);
fprintf('Predict time took %.2f seconds.\n', rf_predict_time);
fprintf('Test Error %.2f.\n', test_error);

% Find the RUC
[roc_X, roc_Y, ~, RF_AUC] = perfcurve(y_test, score(:,2), 1);

% Plot ROC curve
figure;
plot(roc_X, roc_Y);
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title(['ROC Curve for Random Forest (AUC = ' num2str(RF_AUC) ')']);

% Visualizing the Number of Trees used
figure;
plot(numTreesGrid, accuracyResults, '-o');
xlabel('NumLearningCycles (Number of Trees)');
ylabel('Accuracy');
title('Random Forest Model Accuracy vs Number of Trees');
grid on

% Confusion Matrix
if ~isempty(bestRFModel)
    y_pred_best = predict(bestRFModel, X_test);
    confMatrix = confusionmat(y_test, y_pred_best); 
    figure;
    confusionchart(confMatrix, {'0', '1'});
    title(sprintf('Confusion Matrix For Random Forest', bestNumTrees));
end

writetable(X_test, 'testdata_randomforest.csv');
save('optimizedrandomforest.mat','bestRFModel')