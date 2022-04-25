%% Project 1: SVM for Classification of Spam Email Messages Task 3

%%%%%%%%%%%%%%%%%%%%% Variables %%%%%%%%%%%%%%%%%%%%%%%%%
% Do Not Change. This is the best model from Task 1 & 2 %
C = 2.1;                                                %   
p = 1;                                                  %
sigma = 3;                                              %
threshold = 1e-4;                                       %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Load Best Model Parameters and Eval Set
load('best_model.mat');
load('eval.mat');

% Data Standardization
norm_eval = (eval_data - feature_mean) ./ feature_std;

% % Check if all label are within set of {-1,+1}
e = find(eval_label ~= 1 & eval_label ~= -1, 1);
if ~isempty(e)
   error('Eval_label is not within set of {-1,+1}.')
end

% Discriminant Function
% Evaluation Set
g_eval = discriminant_func(alpha, p, C, sigma, norm_train, train_label, norm_eval, b);

% Prediction with trained SVM
% Evaluation Set
eval_predicted = zeros(length(eval_label), 1);
for i = 1:length(g_eval)
    if g_eval(i) > 0 
        eval_predicted(i) = 1;
    else
        eval_predicted(i) = -1;
    end    
end
 
% Find SVM Prediction Accuracy
eval_acc = 1 - sum((eval_predicted - eval_label) ~= 0) / length(eval_label);

fprintf('Condition: p = %d, C = %d, sigma = %d\n', p, C, sigma);
fprintf('Evaluation Accuracy: %.2f%%\n', eval_acc * 100);