%% Homework 3 Q3b

clc;
clear;

% Set Random Seed
rng(42);

X  = randn(800, 2);
s2 = sum(X.^2, 2);
trainX = (X.*repmat(1*(gammainc(s2/2,1).^(1/2))./sqrt(s2),1,2))';

w = som(trainX, 8, 8, 500);

figure();
hold on
plot(trainX(1,:),trainX(2,:),'+r'); 
for i = 1:8
    plot(w(1,(8*i-7):(8*i)),w(2,(8*i-7):(8*i)),'-ok');
    plot(w(1,i:8:64),w(2,i:8:64),'-ok');
end
axis equal;
legend('Training Data', 'SOM');
