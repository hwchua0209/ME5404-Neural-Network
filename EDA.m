%% Exploratory Data Analysis

load('train.mat');
load('test.mat');

% Training data
spam_tr     = length(find(train_label == 1));
non_spam_tr = length(find(train_label == -1));
train    = [spam_tr, non_spam_tr];

% Testing data
spam_te     = length(find(test_label == 1));
non_spam_te = length(find(test_label == -1));
test = [spam_te, non_spam_te];

% Plotting graph
figure;

subplot(1,2,1);
pie(train);
title('Training Set');

subplot(1,2,2);
pie(test);
title('Testing Set');
legend('spam', 'non-spam', 'Location', 'best');
