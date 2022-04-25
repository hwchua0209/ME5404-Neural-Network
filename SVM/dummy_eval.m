clc;
clear;

load('train.mat');
load('test.mat');

data  = [train_data test_data];
label = [train_label' test_label'];

idx = randperm(length(data), 600);

eval_data  = data(:, idx);
eval_label = label(:, idx)';

save(['eval.mat'], 'eval_data', 'eval_label');