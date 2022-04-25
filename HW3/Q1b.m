%% Homework 3 Q1b

clc;
clear;

% Training and Testing Data Set
x_train = [-1.6:0.08:1.6];
x_test  = [-1.6:0.01:1.6];
m_train = size(x_train,2);
y_train = 1.2 * sin(pi * x_train) - cos(2.4 * pi * x_train) + 0.3 * randn(1,m_train);
y_test  = 1.2 * sin(pi * x_test) - cos(2.4 * pi * x_test);

% Parameters for RBF Fix Center
center_idx = randsample(m_train,20);
x_train_20 = x_train(center_idx);
d_max      = max(max(dist(x_train_20)));

% Gaussian Functions for Training Set
phi_train = exp((-m_train / d_max^2) * (dist(x_train.',x_train_20).^2)); 
w = pinv((phi_train.' * phi_train)) * phi_train.' * y_train.';

% Gaussian Functions for Testing Set
phi_test = exp((-m_train / d_max^2) * (dist(x_test.',x_train_20).^2));
y_pred = (phi_test * w).';

% MSE of this Function Approximation
err = mse(y_pred, y_test);
fprintf('The MSE is %.2f\n', err);

% Plot of Trained Output vs Desired Output
figure(1);
hold on;
plot(x_test, y_test, 'LineWidth', 2);
plot(x_test, y_pred, 'LineWidth', 2);
plot(x_train, y_train, 'ok', 'LineWidth', 2);
legend('Desired Output Function','Estimated Output Function','Training Data');