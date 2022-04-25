%% ME 5404 RL Project Task 2

clc;
clear;
rng(0); % For Consistent Result

%%%%%%%%%%%%%%%%%%%%% Variables %%%%%%%%%%%%%%%%%%%%%%%
% Do Not Change the Variable Here
gamma = 0.9;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Load task1.mat
load qeval.mat;

% Parameters to Record
[num_state, num_action] = size(reward);
time  = zeros(10, 1);
Q_op  = zeros(num_state, num_action);

% Run Learning Process
for run = 1:10
    
    % Initialize Value
    disp(['Run: ', num2str(run)]);
    Q     = zeros(num_state, num_action);
    trial = 1; % time step
    tic;
    while trial < 3001 
        state_k = 1;
        k = 1;
        Q_old = Q;
        % For Each Trial
        while state_k ~= 100
            % Get epsilon and alpha from time step
            [epsilon, alpha] = get_param(1, k);

            % Select Action
            valid_action = find(reward(state_k, :) ~= -1);
            action = get_action(epsilon, Q, valid_action, state_k);

            % Move to Next State
            new_state_k = get_state(action, state_k);

            % Terminal Condition
            if alpha < 0.005
                break
            end
            
            % Update Q-value for Current State
            Q(state_k, action) = Q(state_k, action) + alpha * (reward(state_k, action) + ...
                               gamma * max(Q(new_state_k, :)) - Q(state_k, action));            
                           
            % Update Parameters for Next Step
            state_k = new_state_k;
            k       = k + 1;
        end
        trial = trial + 1;
        if state_k == 100
            if (Q - Q_old < 0.001)
                Q_op(:, :, run) = Q;
                break
            end
        end    
    end   
    toc;
    disp('*************************************');
    time(run) = toc;
end

% Check if Optimal Policy Exist
[opt_rew, reach_goal, opt_allact, opt_path] = get_policy(reward, Q_op);
[~, opt_policy_idx] = max(opt_rew); 

% Pick Policy with Max Rewards
opt_policy  = opt_allact(:, opt_policy_idx);
opt_path    = opt_path(:, opt_policy_idx);
opt_max_rew = opt_rew(opt_policy_idx);

% Average Execution Time(s) for Successful Runs
goal_idx = find(reach_goal ~= 0);
ave_time = mean(time(goal_idx));

%% Result Display

if all(reach_goal == 0)
    disp('*************************************');
    disp('No Optimal Policy.');
else 
    disp('********** Results Display **********');
    disp(['Max Reward Obtained        : ' num2str(opt_max_rew)]);
    disp(['Average Execution Time     : ' num2str(ave_time) ' seconds']);
    disp(['Number of Goal Reached Run : ' num2str(sum(reach_goal)) ' runs']);
    opt_policy
end 

%% Plot
% 10X10 Grid with Arrows Showing Action Selected for Each State
figure(1);
grid on; grid minor; hold on; axis ij;
[t_1,s_1] = title({'Action Selected by Optimal Policy for Each State for Exponential Epsilon Decay';...
    ['gamma = ', num2str(gamma)]});
ax = gca;
ax.GridAlpha = 0; ax.LineWidth = 0.6;
ax.MinorGridColor = 'b'; ax.MinorGridAlpha = 1; ax.MinorGridLineStyle = '-';
ax.XLim = [0,11]; ax.YLim = [0,11];
ax.XAxis.MinorTickValues = 0:0.5:11; ax.YAxis.MinorTickValues = 0:0.5:11; 

for col = 1:10
    for row = 1:10
        pol_reshape = reshape(opt_policy, [10,10]);
        if pol_reshape(row, col) == 1
            plot(col, row, '^', 'MarkerSize',8, 'MarkerEdgeColor','red');
        elseif pol_reshape(row, col) == 2
            plot(col, row, '>', 'MarkerSize',8, 'MarkerEdgeColor','red');
        elseif pol_reshape(row, col) == 3
            plot(col, row, 'v', 'MarkerSize',8, 'MarkerEdgeColor','red');
        else 
            plot(col, row, '<', 'MarkerSize',8, 'MarkerEdgeColor','red');
        end
    end
end

% 10X10 Grid with Arrows Showing Optimal Path Taken by Robot
figure(2);
grid on; grid minor; hold on; axis ij;
[t_2,s_2] = title({'Optimal Path Taken by Robot in Task 2';...
    ['gamma = ', num2str(gamma),' Rewards = ', num2str(opt_max_rew)]});
ax = gca;
ax.GridAlpha = 0; ax.LineWidth = 0.6;
ax.MinorGridColor = 'b'; ax.MinorGridAlpha = 1; ax.MinorGridLineStyle = '-';
ax.XLim = [0,11]; ax.YLim = [0,11];
ax.XAxis.MinorTickValues = 0:0.5:11; ax.YAxis.MinorTickValues = 0:0.5:11; 

for col = 1:10
    for row = 1:10
        path_reshape = reshape(opt_path, [10,10]);
        plot(10, 10, 'p', 'MarkerSize',10, 'MarkerEdgeColor','red', 'MarkerFaceColor', 'red');
        if path_reshape(row, col) == 1
            plot(col, row, '^', 'MarkerSize',8, 'MarkerEdgeColor','red');
        elseif path_reshape(row, col) == 2
            plot(col, row, '>', 'MarkerSize',8, 'MarkerEdgeColor','red');
        elseif path_reshape(row, col) == 3
            plot(col, row, 'v', 'MarkerSize',8, 'MarkerEdgeColor','red');
        elseif path_reshape(row, col) == 4
            plot(col, row, '<', 'MarkerSize',8, 'MarkerEdgeColor','red');
        end
    end
end

%% Function
% Function to Run Various Mode
function [epsilon, alpha] = get_param(model, k)
switch(model)
    case 1 % Exponential Decay
        epsilon = exp(-0.001 * k);
        alpha   = epsilon; 
end
end

% Function to Get Action Based on Epsilon-Greedy
function action = get_action(epsilon, Q, valid_action, state)
% epsilon: explore, 1 - epislon: exploit
r = rand;
x = sum(r >= cumsum([0, 1-epsilon, epsilon]));
    
% exploit
if x == 1 
    if all(Q(state, :) == Q(state, 1))
        action   = randsample(valid_action, 1);
    else
        [~, idx] = max(Q(state, valid_action));
        action   = valid_action(idx);
    end
% explore
else    
    [~, idx] = max(Q(state, valid_action));
    valid_action(idx) = [];
    if length(valid_action) == 1
        action = valid_action;
    else
        action = randsample(valid_action, 1);
    end
end
end

% Function to Get New State from Chosen Action
function new_state = get_state(action, state)
if action == 1
    new_state = state - 1;
elseif action == 2
    new_state = state + 10;
elseif action == 3
    new_state = state + 1;
else 
    new_state = state - 10;
end    
end

% Function to Get Optimal Policy
function [rewards_list, reach_goal, policy_allaction, policy_path] = get_policy(reward, Q_op)
num_state        = size(reward, 1);
reach_goal       = zeros(10, 1);
rewards_list     = zeros(10, 1);
policy_allaction = zeros(num_state, size(Q_op, 3));
policy_path      = zeros(num_state, size(Q_op, 3));

for i = 1:size(Q_op, 3)
    Q = Q_op(:, :, i);
    state = 1;
    running_state  = [];
    rewards = 0;
    while state ~= 100
        running_state(end+1) = state;
        valid_action     = find(reward(state, :) ~= -1);
        [~, idx]         = max(Q(state, valid_action));
        action           = valid_action(idx);
        policy_path(state, i) = action;
        rewards          = rewards + reward(state, action);
        new_state        = get_state(action, state);
        state            = new_state; 
        if length(unique(running_state)) ~= length(running_state)
%             disp(['Policy ' num2str(i) ' is not Optimal.']);
%             disp('*************************************');
            break;
        end
        if state == 100
%             disp(['Policy ' num2str(i) ' is Optimal.']);
%             disp('*************************************');
            reach_goal(i)   = 1;
            rewards_list(i) = rewards;
            for j = 1:100
                [~, action_opt] = max(Q(j, :));
                policy_allaction(j, i) = action_opt;
            end
        end
    end
end
end
