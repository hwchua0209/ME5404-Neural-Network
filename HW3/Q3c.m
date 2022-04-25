%% Homework 3 Q3c

load Digits.mat;

% Set Random Seed
rng(42);

% Load Data
trainIdx = find(~(train_classlabel==1 | train_classlabel==2));
Train_ClassLabel = train_classlabel(trainIdx);
Train_Data = train_data(:,trainIdx);

testIdx = find(~(test_classlabel==1 | test_classlabel==2));
Test_ClassLabel = test_classlabel(testIdx);
Test_Data = test_data(:,testIdx);

%% Q3c1
w = som(Train_Data, 10, 10, 1000);

% Display Semantic Map
figure;
tlo = tiledlayout(10, 10, 'TileSpacing', 'tight');
for i = 1:100
    ax = nexttile(tlo);
    image = reshape(w(:, i), 28, 28);
    imshow(image, [0 1]);
end

%% Q3c2

% Label Winner Neurons with Correct Label
winner_lbl = zeros(100, 1);

for idx = 1:100
    x = Train_Data(:, idx);
    [~, min_idx] = min(dist(w', x));
    winner_lbl(min_idx) = Train_ClassLabel(idx);
end

% Find Training Accuracy
tr_winner = zeros(600, 1);

for idx = 1:600
    x = Train_Data(:, idx);
    [~, min_idx] = min(dist(w', x));
    tr_winner(idx) = winner_lbl(min_idx);
end

tr_acc = sum(tr_winner == Train_ClassLabel')/ 600;
fprintf('The training accuracy for this SOM is %.3f\n', tr_acc);

% Find Testing Accuracy
te_winner = zeros(60, 1);

for idx = 1:60
    x = Test_Data(:, idx);
    [~, min_idx] = min(dist(w', x));
    te_winner(idx) = winner_lbl(min_idx);
end

te_acc = sum(te_winner == Test_ClassLabel')/ 60;
fprintf('The testing accuracy for this SOM is %.3f\n', te_acc);
