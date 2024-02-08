% Loading the original dataset
data = readtable('bank_data.csv');
% checking for NaN Values
sum(ismissing(data))
% cleaning the income_annual column and filling with mean
mean_income = mean(data.income_annual(data.income_annual > 0));

data.income_annual = fillmissing(data.income_annual, 'constant', mean_income);
% cleaning the residential_assets_value as it contains some negative values
% Extract 'residential_assets_value' column
residential_assets_value = data.residential_assets_value;
positive_values = residential_assets_value(residential_assets_value > 0);
% Calculate mean of positive values
mean_residential_assets_value = mean(positive_values);
% Replace values less than or equal to 0 with the mean value
residential_assets_value(residential_assets_value <= 0) = mean_residential_assets_value;
% Update the table with the modified column
data.residential_assets_value = residential_assets_value;
% Cleaning bank assets
% Extract 'bank_asset_value' column
bank_asset_value = data.bank_asset_value;
% Calculate the mean of positive values
mean_bank_asset_value = mean(bank_asset_value(bank_asset_value > 0));
% Find indices of missing values
missing_values = isnan(bank_asset_value);
% Fill the bank_assets missing values with the mean value
bank_asset_value(missing_values) = mean_bank_asset_value;
% Update the table with the modified column
data.bank_asset_value = bank_asset_value;
sum(ismissing(data))
% Dropping Irrelevant features
data(:, {'Var1', 'id'}) = [];

% Doing Label Encoding to convert categorical into numerical
data.education = grp2idx(data.education);
data.self_employed = grp2idx(data.self_employed);

%Mapping the loan_status column to numeric values
loan_status = data.loan_status;
loan_status = strrep(loan_status, 'Approved', '1');
loan_status = strrep(loan_status, 'Rejected', '0');
data.loan_status = str2double(loan_status);

numTestSamples = floor(0.2 * height(data));

% Randomly shuffle the rows of the dataframe
rng('default');
shuffledIdx = randperm(height(data));
df_shuff = data(shuffledIdx, :);

% Splitting the data
Xtrain = df_shuff(1:end-numTestSamples, ~strcmp('loan_status', df_shuff.Properties.VariableNames));
Xtest = df_shuff(end-numTestSamples+1:end, ~strcmp('loan_status', df_shuff.Properties.VariableNames));
ytrain = df_shuff.loan_status(1:end-numTestSamples);
ytest = df_shuff.loan_status(end-numTestSamples+1:end);

trainData = [Xtrain, table(ytrain)];

% Fit logistic regression model on TRAINING data
predictorVars = trainData.Properties.VariableNames(~strcmp('ytrain', trainData.Properties.VariableNames));
formula = ['ytrain ~ ', strjoin(predictorVars, ' + ')];

% Training the model
model = fitglm(trainData, formula, 'Distribution', 'binomial');
training_time = toc;
% Preparing test data
testData = [Xtest, table(ytest)];

% Predicting on test data
preds = predict(model, testData(:, predictorVars));

% Convert probabilities to binary predictions
predsBinary = preds > 0.5;

% Evaluating accuracy
accuracy = sum(predsBinary == ytest) / numel(ytest);
disp(['Logistic Baseline Moedel Accuracy: ', num2str(accuracy)]);
mu = mean(Xtrain);
sigma = std(Xtrain);
Xtrain_std = (Xtrain - mu) ./ sigma;
Xval_std = (Xtest - mu) ./ sigma;  % Standardize test data outside the loop

% Define hyperparameters
C_values = [1, 10];
penalties = {'ridge', 'lasso'};

% Initialize variables to track the best model
bestLRModel = [];
bestLRAccuracy = 0;
lr_predict_time = 0;
bestLRParams = struct('C', [], 'penalty', []);
% Loop over the grid
for c = C_values
    for penalty = penalties
        % Template for binary learner
        t = templateLinear('Regularization', penalty{1}, 'Lambda', 1/c);
        tic;
        % Train the model (change to fitclinear if binary classification)
        op_model = fitcecoc(Xtrain_std, ytrain, 'Learners', t, 'Coding', 'onevsall');
        time = toc;
        training_time = max(time, training_time);
        
        tic;
        % Predict on the validation set
        [ypred, score] = predict(op_model, Xval_std);
        lr_predict = toc;
        lr_predict_time = max(lr_predict_time, lr_predict);
        
        % Calculate accuracy    
        currentAccuracy = sum(ypred == ytest) / length(ytest);
        
        % Update best model if current model is better
        if currentAccuracy > bestLRAccuracy
            bestLRModel = op_model;
            bestLRAccuracy = currentAccuracy;
            bestLRParams.C = c;
            bestLRParams.penalty = penalty{1};
        end
    end
end


lr_test_error = sum(ypred ~= ytest) / length(ytest);

disp(training_time)
fprintf('Predicting Time %.2f seconds.\n', lr_test_error);
fprintf('Train Error %.2f.\n', lr_predict_time);
disp(['Optimized LR Accuracy: ', num2str(bestLRAccuracy)])

%Displaying the bestLRModel, bestLRAccuracy, and bestLRParams
disp(bestLRParams)

%Compute confusion matrix for the best model
if ~isempty(bestLRModel)
    ypredbest = predict(bestLRModel, Xtest);
    LR_confMatrix = confusionmat(ytest, ypredbest); 
    % Displaying the confusion matrix
    figure;
    confusionchart(LR_confMatrix, {'0', '1'});
    title(sprintf('Confusion Matrix for Optimized Logistic Regression Model'));
end

%Displaying the RUC Curve
[rocX, rocY, ~, LR_AUC] = perfcurve(ytest, score(:, 2), 1);
%Plot ROC curve
figure;
plot(rocX, rocY);
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title(['ROC Curve Logistic Regression (AUC = ' num2str(LR_AUC) ')']);
save('logisticmodel.mat','bestLRModel')
writetable(Xval_std, 'testdata_logistic.csv');
