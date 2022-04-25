%% Project 1: SVM for Classification of Spam Email Messages Task 1 & 2

clc;
clear;
rng(0);

%%%%%%%%%%%%%%%%%%%%% Variables %%%%%%%%%%%%%%%%%%%%%%%
% Change these values for various configuration

C = 2.1; % To choose from C = 0.1, 0.6, 1.1, 2.1, 10e6
p = 1;   % To choose from p = 1, 2, 3, 4, 5
sigma = 3; % For RBF Kernel. To choose from sigma = 0(when not selected), 0.1, 1, 2, 3, 4
threshold = 1e-4;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Load Training & Testing Set
load('train.mat');
load('test.mat');

% Data Standardization
feature_mean = mean(train_data, 2);
feature_std  = std(train_data, 0, 2);

norm_train = (train_data - feature_mean) ./ feature_std;
norm_test  = (test_data - feature_mean) ./ feature_std;

% Check if all label are within set of {-1,+1}
x = find(train_label ~= 1 & train_label ~= -1, 1);
if ~isempty(x)
   error('Train_label is not within set of {-1,+1}.')
end

y = find(test_label ~= 1 & test_label ~= -1, 1);
if ~isempty(y)
   error('Test_label is not within set of {-1,+1}.')
end

% Apply Mercer's Condition
[K, H] = mercer(p, C, sigma, norm_train, train_label);
 
% Calculate Lagrange Multiplier alpha
f   = -ones(2000, 1);
A   = [];
B   = [];
Aeq = train_label';
beq = 0;
ib  = zeros(2000,1);
ub  = ones(2000,1) * C;
x0  = [];
options = optimset('LargeScale', 'off', 'MaxIter', 1000);

alpha = quadprog(H, f, A, B, Aeq, beq, ib, ub, x0, options);

% Find w and b
w = zeros(size(train_data, 1), 1);

% Hard Margin SVM with Linear Kernel
if (p == 1 && C == 10e6)
   for i = 1:length(train_label) 
       w = w + (alpha(i) * train_label(i) * norm_train(:, i));
   end
   sv      = find(alpha > threshold);
   sv_rand = randi(length(sv));
   b       = (1 ./ train_label(sv_rand)) - w' * norm_train(:, sv_rand);

% SVM with RBF Kernel
elseif sigma ~= 0
   sv_poly = find(alpha > threshold);
   w = sum(alpha .* train_label .* K);
   b = mean(train_label(sv_poly) - w(sv_poly), 'all');

% SVM with Polynomial Kernel
else
   sv_poly = find(alpha > threshold);
   w = sum(alpha .* train_label .* K);
   b = mean(train_label(sv_poly) - w(sv_poly), 'all');
end

% Discriminant Function
% Training Set
g_train = discriminant_func(alpha, p, C, sigma, norm_train, train_label, norm_train, b);

% Testing Set
g_test  = discriminant_func(alpha, p, C, sigma, norm_train, train_label, norm_test, b);

% Prediction with trained SVM
% Training Set
train_pred = zeros(length(train_label), 1);
for i = 1:length(g_train)
    if g_train(i) > 0 
        train_pred(i) = 1;
    else
        train_pred(i) = -1;
    end    
end

% Testing Set
test_pred = zeros(length(test_label), 1);
for i = 1:length(g_test)
    if g_test(i) > 0 
        test_pred(i) = 1;
    else
        test_pred(i) = -1;
    end    
end

% Find SVM Prediction Accuracy
train_acc = 1 - sum((train_pred - train_label) ~= 0) / length(train_label);
test_acc  = 1 - sum((test_pred - test_label) ~= 0) / length(test_label);

fprintf('Condition: p = %d, C = %d, sigma = %d\n', p, C, sigma);
fprintf('Training Accuracy: %.2f%%\n', train_acc * 100);
fprintf('Testing Accuracy : %.2f%%\n', test_acc * 100);

% Save Best Model for Task 3
if (sigma == 3 && C == 2.1)
    save(['best_model'], 'feature_mean', 'feature_std', 'b', 'alpha', 'norm_train', 'train_label');
end