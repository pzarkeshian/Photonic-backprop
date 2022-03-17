clear; clc;
tic

%% Learning Rate
epsilon = 0.0005;

%% Photon emission rate
q = 0.0005;

%% Amount of noisy photons
% np = 0.1;

%% Set Layers
Ni = 784; Nh = 500; No = 10; 
NT = 60000;

%% MNIST Load
load('MNIST_data.mat')

%% Prepare Input
rand_index = randperm(NT, NT);
x0 = training_REC(rand_index, :)'; x0(Ni+1,:) = 1;

%% Prepare Randomized Target
t = zeros(10, NT);
t_ind = sub2ind([10, NT], class_train(rand_index)'+1, 1:NT);
t(t_ind) = 1;

%% Weights
W1 = randn(Nh, Ni+1) / (Ni+1);
W2 = randn(No, Nh) / Nh;

%% Parameters for Photon Emission
num_photons = 0;
num_i = size(W1, 1) * size(W1, 2); % number of input weights
num_h = size(W2, 1) * size(W2, 2); % number of hidden weights

%% Error Storage
error = zeros(NT, 1);
store_avg = zeros(NT, 1);

%% Validation Input
val_size = size(class_test, 1);
rand_val_index = randperm(val_size, val_size);
x0_val = testing_REC(rand_val_index, :)'; x0_val(Ni+1, :) = 1;

%% Validation Target
t_val = zeros(10, val_size);
t_val_ind = sub2ind([10, val_size], class_test(rand_val_index)'+1, ...
    1:val_size);
t_val(t_val_ind) = 1;

%% Validation Error Storage
error_val = zeros(NT, 1);
store_avg_val = zeros(NT, 1);

%% Main Loop

for sample = 1:NT
    
    xi = x0(:,sample);
    
    % Feedforward
    x1 = 1 ./ (1 + exp(-(W1 * xi)));
    x2 = 1 ./ (1 + exp(-(W2 * x1)));
    
    % Derivative of Feedforward
    dx1 = x1 .* (1 - x1);
    dx2 = x2 .* (1 - x2);
    
    % Delta
    target = t(:,sample);
    delta_2 = (x2 - target) .* dx2;
    delta_1 = (W2' * delta_2) .* dx1;
    
    % Derivative of error w.r.t to weight
    dW2 = delta_2 * x1';
    dW1 = delta_1 * xi';
    
    % Binary updates
%   % Uncomment for Binary Updating of photons
%     dW2 = sign(dW2);
%     dW1 = sign(dW1);
    
    if q < 1
        % Stochastic Update of Weights
        ind_h = randperm(num_h, round(num_h * q));
        ind_i = randperm(num_i, round(num_i * q));
        W2(ind_h) = W2(ind_h) - (epsilon .* dW2(ind_h));
        W1(ind_i) = W1(ind_i) - (epsilon .* dW1(ind_i));
        
%         % Noisy Photons
%         % Uncomment if noisy photonic updates are desired
%         ind_hnp = randperm(num_h, round(num_h * np));
%         ind_inp = randperm(num_i, round(num_i * np));
%         d_noise1 = sign(rand(Nh, Ni+1) - 0.5);
%         d_noise2 = sign(rand(No, Nh) - 0.5);
%         W2(ind_hnp) = W2(ind_hnp) - (epsilon .* d_noise2(ind_hnp));
%         W1(ind_inp) = W1(ind_inp) - (epsilon .* d_noise1(ind_inp));

    else
        % Updating all Weights
        W2 = W2 - (epsilon .* dW2);
        W1 = W1 - (epsilon .* dW1);
    end
    
    % Classify the Digit
    [~, index_calc] = max(x2);
    index_target = find(target == 1);
    error(sample, 1) = index_calc ~= index_target;
    
    % Store the Moving Average Error
    index_min = max(sample - 100, 1);
    store_avg(sample, :) = mean(error(index_min:sample));
    
    % Validation Feedforward
    rand_val_index = randperm(val_size, 1);
    xi_val = x0_val(:,rand_val_index); target_val = t_val(:,rand_val_index);
    x1_val = 1 ./ (1 + exp(-(W1 * xi_val)));
    x2_val = 1 ./ (1 + exp(-(W2 * x1_val)));
    
    % Validation Error
    [~, index_val] = max(x2_val);
    index_val_t =  find(target_val == 1);
    error_val(sample, 1) = index_val ~=  index_val_t;
    store_avg_val(sample, :) = mean(error_val(index_min:sample));
    
end

toc