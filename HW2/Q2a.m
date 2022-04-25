clc;
clear;

% No of Hidden Layer
n = 10;

% Epochs
epoch = 1000;

% Training and Testing Data Set
x_train = [-1.6:0.05:1.6];
x_test  = [-1.6:0.01:1.6];


t_train = 1.2*sin(pi*x_train) - cos(2.4*pi*x_train);
t_test  = 1.2*sin(pi*x_test) - cos(2.4*pi*x_test);

% Change the input to cell array form for sequential training

x_cell = num2cell(x_train);
t_cell = num2cell(t_train);

% Construct and configure the MLP
net = fitnet(n,'trainlm');
net.trainParam.lr = 0.1;
    
for i = 1:epoch
    idx = randperm(65);
    net = adapt(net, x_train(idx), t_train(idx));

    % Get Predicted Training Output with Trained Network
    y_train = net(x_train);
    
    % Get Predicted Testing Output with Trained Network
    y_test = net(x_test);
    
end

% Plot of Trained Output vs Desired Output
figure(1);
title('Comparison of Trained Output Function from Network vs Desired Output Function')
scatter(x_test, t_test, 'LineWidth', 2);
hold on;
plot(x_test, y_test, 'LineWidth', 2);
legend('Desired Output Function','Estimated Output Function');

% Compute Outputs of MLP when x=-3 and 3
t_pos = 1.2*sin(pi*3) - cos(2.4*pi*3);
t_neg = 1.2*sin(pi*-3) - cos(2.4*pi*-3);

pred_pos = net(3);
pred_neg = net(-3);

fprintf('The output value from function for x = 3 : %.4f\n', t_pos);
fprintf('The output value from function for x = -3: %.4f\n', t_neg);
fprintf('The output value from approx function for x = 3  : %.4f\n', pred_pos);
fprintf('The output value from approx function for x = -3 : %.4f\n', pred_neg);