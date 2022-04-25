%% Homework 3 Q3a

clc;
clear;

% Set Random Seed
rng(42);

x = linspace(-pi, pi, 400);
trainX = [x; sinc(x)];

w = som(trainX, 1, 40, 500);

figure();
plot(trainX(1,:), trainX(2,:), '+r' ,w(1,:),w(2,:),'-ok');
axis equal;
legend('Training Data', 'SOM');
