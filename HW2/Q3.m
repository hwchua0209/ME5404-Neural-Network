clc;
clear;
warning('off', 'MATLAB:MKDIR:DirectoryExists');

%% GroupID = 2 

%% Data Extraction and Formatting

animal_files = dir('group_2/deer/*.jpg');
object_files = dir('group_2/ship/*.jpg');

img_train = zeros(900, 1024);
img_test  = zeros(100, 1024);
lbl_train = zeros(900, 1);
lbl_test  = zeros(100, 1);

% Extract Training File for Animal
for i = 1:450
    img_name = animal_files(i).name;
    img = imread(['group_2/deer/', img_name]);
    img = rgb2gray(img);
    img = reshape(img, [1, 1024]);
    img_train(i, :) = img;
    lbl_train(i, :) = 1;
end

% Extract Testing File for Animal
for i = 451:500
    img_name = animal_files(i).name;
    img = imread(['group_2/deer/', img_name]);
    img = rgb2gray(img);
    img = reshape(img, [1, 1024]);
    img_test(i-450, :) = img;
    lbl_test(i-450, :) = 1;
end 

% Extract Training File for Man-made Object
for i = 1:450
    img_name = object_files(i).name;
    img = imread(['group_2/ship/', img_name]);
    img = rgb2gray(img);
    img = reshape(img, [1, 1024]);
    img_train(i+450, :) = img;
    lbl_train(i+450, :) = 0;
end

% Extract Testing File for Man-made Object
for i = 451:500
    img_name = object_files(i).name;
    img = imread(['group_2/ship/', img_name]);
    img = rgb2gray(img);
    img = reshape(img, [1, 1024]);
    img_test(i-400, :) = img;
    lbl_test(i-400, :) = 0;
end

%% a) Rosenblatt's Perceptron

% Initialize parameters
lr_a = 0.1;

% Add Bias to Training and Testing set
img_train_a = [ones([900,1]) img_train];
img_test_a  = [ones([100,1]) img_test];

[train_acc, test_acc, epoch] = perceptron(img_train_a, img_test_a, lbl_train, lbl_test, lr_a);
fprintf('Q3a\n');
fprintf('Accuracy of Rosenblatt perceptron (train): %.4f\n', train_acc);
fprintf('Accuracy of Rosenblatt perceptron (test): %.4f\n', test_acc);
fprintf('Number of epochs before converge: %f\n', epoch);

%% b) Rosenblatt's Perceptron with Normalization

img_train_mean = mean(img_train, 'all');
img_train_std  = std(img_train, 1, 'all');

img_train_norm = (img_train - img_train_mean) / img_train_std;
img_test_norm  = (img_test - img_train_mean) / img_train_std;

% Initialize parameters
lr_b = 0.1;

% Add Bias to Training and Testing set
img_train_b = [ones([900,1]) img_train_norm];
img_test_b  = [ones([100,1]) img_test_norm];

[train_acc, test_acc, epoch] = perceptron(img_train_b, img_test_b, lbl_train, lbl_test, lr_b);
fprintf('Q3b\n');
fprintf('Accuracy of Rosenblatt perceptron (train): %.4f\n', train_acc);
fprintf('Accuracy of Rosenblatt perceptron (test): %.4f\n', test_acc);
fprintf('Number of epochs before converge: %f\n', epoch);

%% c) & d) MLP (Batch Training)

img_train_c = img_train.';
img_test_c  = img_test.';
lbl_train_c = lbl_train.';
lbl_test_c  = lbl_test.';

net_c = patternnet(100);
net_c.trainFcn = 'traingdx';
net_c.trainParam.epochs = 5000;
net_c.divideParam.trainRatio  = 1.0;
net_c.divideParam.valRatio  = 0;
net_c.divideParam.testRatio = 0;
net_c.performParam.regularization = 0.5; % Unmask for Q3d. 
[net_c,tr] = train(net_c, img_train_c, lbl_train_c);

y_train_c = net_c(img_train_c);
y_test_c  = net_c(img_test_c);

class_y_train_c = classification(y_train_c);
class_y_test_c  = classification(y_test_c);
train_acc = 1 - mean(abs(class_y_train_c - lbl_train_c));
test_acc  = 1 - mean(abs(class_y_test_c - lbl_test_c));

fprintf('Q3c\n');
fprintf('Accuracy of MLP (train): %.4f\n', train_acc);
fprintf('Accuracy of MLP (test): %.4f\n', test_acc);

%% e) MLP (Sequential Training)

img_train_e = img_train.';
img_test_e  = img_test.';
lbl_train_e = lbl_train.';
lbl_test_e  = lbl_test.';

epochs = 100;

net_e = patternnet(100);
net_e.trainFcn = 'traingdx';
net_e.divideParam.trainRatio= 1.0;
net_e.divideParam.valRatio  = 0;
net_e.divideParam.testRatio = 0;

train_acc = zeros(epochs,1);
test_acc  = zeros(epochs,1);

for i = 1 : epochs
    
%     display(['Epoch: ', num2str(i)])
    
    idx = randperm(900);
    net_e = adapt(net_e, img_train_e(:,idx), lbl_train_e(:,idx));
    
    y_train_e = net_e(img_train_e);
    y_test_e  = net_e(img_test_e);
    
    class_y_train_e = classification(y_train_e);
    class_y_test_e  = classification(y_test_e);
    train_acc = 1 - mean(abs(class_y_train_e - lbl_train_e));
    test_acc  = 1 - mean(abs(class_y_test_e - lbl_test_e));
end

fprintf('Q3e\n');
fprintf('Accuracy of MLP Sequential (train): %.4f\n', train_acc);
fprintf('Accuracy of MLP Sequential (test): %.4f\n', test_acc);

%% Rosenblatt's Perceptron Function

function [train_acc, test_acc, epoch] = perceptron(train, test, lbl_train, lbl_test, lr)
    % Initialize parameters
    w = rand([1025, 1]);
    epoch = 0;
    
    % Induced Local Field
    v_train = train * w;
    y_train = hardlim(v_train);
    
    while sum(abs(y_train - lbl_train)) ~= 0
        epoch = epoch + 1;
        for i = 1:size(train, 1)
            v = train(i, :) * w;
            y = hardlim(v);
            e = lbl_train(i) - y;
            w = w + lr * e * train(i, :).';
        end
        y_train = hardlim(train * w);
    end

    v_test  = test * w;
    y_test  = hardlim(v_test);
    
    train_acc = 1 - mean(abs(y_train - lbl_train));
    test_acc = 1 - mean(abs(y_test - lbl_test));
end

%% Function to do Classification

function [class] = classification(y)

class = zeros(1, size(y, 2));

    for i = 1:size(y, 2)
        if y(i) >= 0.5
            class(:, i) = 1;
        else
            class(:, i) = 0;
        end
    end
end