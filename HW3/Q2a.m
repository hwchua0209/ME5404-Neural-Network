%% Homework 3 Q2a

clc;
clear;

% Load the Training and Testing Data
[TrData, TrLabel, TeData, TeLabel] = loadmnist;

sample = length(TrData);

% Standard Deviation
std = 100;

% Gaussian Functions for Training Set
r_train   = dist(TrData);
phi_train = exp(-r_train.^2 / (2 * std^2));

% Gaussian Functions for Testing Set
r_test   = dist(TrData.',TeData);
phi_test = exp(-r_test.^2 / (2 * std^2)); 

for lambda = [0 0.001 0.1 1 10 100 1000]
    if lambda == 0
       w = phi_train \ TrLabel.';
    else
       w = pinv((phi_train.' * phi_train) + lambda * eye(sample)) * phi_train.' * TrLabel.';
    end
    
    TrPred = phi_train.' * w;
    TePred = phi_test.' * w;
    
    % Evaluate Performance of RBFN
    TrAcc = zeros(1,1000);
    TeAcc = zeros(1,1000);
    thr   = zeros(1,1000);
    TrN   = length(TrLabel);
    TeN   = length(TeLabel);
    for i = 1:1000
        t = (max(TrPred)-min(TrPred)) * (i-1)/1000 + min(TrPred); 
        thr(i) = t;
        TrAcc(i) = (sum(TrLabel(TrPred<t)==0) + sum(TrLabel(TrPred>=t)==1)) / TrN;
        TeAcc(i) = (sum(TeLabel(TePred<t)==0) + sum(TeLabel(TePred>=t)==1)) / TeN;
    end
    
    figure;
    plot(thr,TrAcc,'.- ',thr,TeAcc,'^-');
    legend('tr','te');
    xlabel('Threshold');
    ylabel('Accuracy');
    xlim([0 1])
    title(['Lambda = ' num2str(lambda)])
    
end


