%% Homework 3 Q2c

clc;
clear;

% Set Random Seed
rng(5);

% Load the Training and Testing Data
[TrData, TrLabel, TeData, TeLabel] = loadmnist;

sample = size(TrData, 2);

% Chose 2 Random Center Point for K-means CLustering
center = 2;
center_idx = randsample(sample,center);
TrCenter_1 = TrData(:, center_idx(1));
TrCenter_2 = TrData(:, center_idx(2));

% Initialize Parameter
cluster  = zeros(size(TrData, 2), 1);
k = 0;

% K-means Clustering
while true
    
    k = k + 1;
    center_new1  = TrCenter_1;
    center_new2  = TrCenter_2;
    prev_cluster = cluster;
    
    for i = 1:size(TrData, 2)
        x = TrData(:, i);
        dist1 = norm(double(x) - double(center_new1));
        dist2 = norm(double(x) - double(center_new2));
        
        if dist1 < dist2
           cluster(i) = 1;
        else
           cluster(i) = 2;
        end
    end
    
    % Check Convergence
    if sum(abs(prev_cluster - cluster)) == 0
       break
    end
    
   % Calculate New Center
   TrCenter_1 = mean(TrData(:, cluster == 1), 2);
   TrCenter_2 = mean(TrData(:, cluster == 2), 2);
end

TrCenter = [TrCenter_1 TrCenter_2];

% RBFN
d_max = max(max(dist(TrData.', TrCenter)));

phi_train = exp(-(dist(TrData.',TrCenter).^2)*2 / d_max^2);
w = pinv((phi_train.' * phi_train)) * phi_train.' * TrLabel.';

phi_test  = exp(-(dist(TeData.',TrCenter).^2)*2 / d_max^2);

TrPred = phi_train * w;
TePred = phi_test * w;

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

% Plots of K-means Clustering Centers
figure;
subplot(1, 2, 1);
imshow(reshape(center_new1(:, 1),28,28));
title('Center 1 From K-Means Clustering');

subplot(1, 2, 2);
imshow(reshape(center_new2(:, 1),28,28));
title('Center 2 From K-Means Clustering');

loadmnistq2c;







