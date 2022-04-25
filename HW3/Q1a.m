%% Homework 3 Q1a

clc;
clear;

% Standard Deviation
std = 0.1;

% Training and Testing Data Set
x_train = [-1.6:0.08:1.6];
x_test  = [-1.6:0.01:1.6];

n_train = size(x_train,2);

y_train = 1.2 * sin(pi * x_train) - cos(2.4 * pi * x_train) + 0.3 * randn(1,n_train);
y_test  = 1.2 * sin(pi * x_test) - cos(2.4 * pi * x_test);

% Gaussian Functions for Training Set
r_train   = dist(x_train);
phi_train = exp(-r_train.^2 / (2 * std^2)); 

w = phi_train \ y_train.';

% Gaussian Functions for Testing Set
r_test   = dist(x_train.',x_test);
phi_test = exp(-r_test.^2 / (2 * std^2)); 

y_pred = (phi_test.' * w).';

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