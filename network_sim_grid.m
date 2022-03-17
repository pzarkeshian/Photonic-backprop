function network_sim_grid

%% Grid Size
start_grid = -4; end_grid = 0; N_points = 13; % grid size
grid_trials = 10; % number of grids to generate

np = 0.1;

% Load MNIST Digit Set
load('MNIST_data', 'training_REC', 'testing_REC', 'class_train', ...
    'class_test');

% Generate learning rates epsilon and q values
epsilon_in = logspace(start_grid, end_grid, N_points);
q_in = logspace(start_grid, end_grid, N_points);

% Length of learning rates and q values
l_eps = length(epsilon_in);
l_q = length(q_in);

%% Set Layers
Ni = 784; No = 10; Nh = 500;
NT = 60000;

for loop = 3:grid_trials
    %% Storage
    store_error = zeros(60000, 1, l_q, l_eps);
    store_W1 = zeros(Nh, Ni+1, l_q, l_eps);
    store_W2 = zeros(No, Nh, l_q, l_eps);
    
    %% Select File name for Saving in INSERTFILE
    rng(loop);
    savefile = sprintf('./Data/INSERTFILE%d', loop);
    
    %% Main
    for n_epsilon = 1:l_eps
        %% Set epsilon
        epsilon = epsilon_in(n_epsilon);
        
        for n_q = 1:l_q
            %% Set q
            q = q_in(n_q);
            
            %% Weights
            W1 = randn(Nh, Ni+1) / (Ni+1);
            W2 = randn(No, Nh) / Nh;
            
            %% Prepare Input
            rand_index = randperm(NT, NT);
            x0 = training_REC(rand_index, :)'; x0(Ni+1,:) = 1;
            
            %% Prepare Randomized Target
            t = zeros(10, NT);
            t_ind = sub2ind([10, NT], class_train(rand_index)'+1, 1:NT);
            t(t_ind) = 1;
            
            %% Parameters for Photon Emission
            num_photons = 0;
            num_i = size(W1, 1) * size(W1, 2); % number of input weights
            num_h = size(W2, 1) * size(W2, 2); % number of hidden weights
            
            %% Looping through all samples
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
                
                % binary (comment out if not required)
                dW2 = sign(dW2);
                dW1 = sign(dW1);
                
                if q < 1
                    % Stochastic Update of Weights
                    ind_h = randperm(num_h, round(num_h * q));
                    ind_i = randperm(num_i, round(num_i * q));
                    W2(ind_h) = W2(ind_h) - (epsilon .* dW2(ind_h));
                    W1(ind_i) = W1(ind_i) - (epsilon .* dW1(ind_i));
                    num_photons = num_photons + size(ind_h) + size(ind_i);
                    
                    % Noisy Updates (comment out if not required)
                    ind_hnp = randperm(num_h, round(num_h * np));
                    ind_inp = randperm(num_i, round(num_i * np));
                    d_noise1 = sign(rand(Nh, Ni+1) - 0.5);
                    d_noise2 = sign(rand(No, Nh) - 0.5);
                    W2(ind_hnp) = W2(ind_hnp) - (epsilon .* d_noise2(ind_hnp));
                    W1(ind_inp) = W1(ind_inp) - (epsilon .* d_noise1(ind_inp));
                    
                else
                    % Updating all Weights
                    W2 = W2 - (epsilon .* dW2);
                    W1 = W1 - (epsilon .* dW1);
                end
                
                % Classify the Digit
                [~, index_calc] = max(x2);
                index_target = find(target == 1);
                store_error(sample, 1, n_q, n_epsilon) = index_calc ~= index_target;
                
            end
            
            % Store weight matrices
            store_W1(:,:,n_q,n_epsilon) = W1;
            store_W2(:,:,n_q,n_epsilon) = W2;
            
        end
        
    end
    
    %% Validation Input
    val_size = size(class_test, 1);
    x0_val = testing_REC';
    x0_val(Ni+1, :) = 1;
    
    %% Validation Target
    t_val = zeros(10, val_size);
    t_val_ind = sub2ind([10, val_size],class_test'+1,1:val_size);
    t_val(t_val_ind) = 1;
    
    %% Validation Error Storage
    error_val = zeros(val_size, l_q, l_eps);
    store_error_val = zeros(l_q, l_eps);
    
    %%
    for n_ee = 1:l_eps
        for n_qq = 1:l_q
            for val_samples = 1:val_size
                xi_val = x0_val(:,val_samples);
                target_val = t_val(:,val_samples);
                x1_val = 1 ./ (1 + exp(-(store_W1(:,:,n_qq,n_ee) * xi_val)));
                x2_val = 1 ./ (1 + exp(-(store_W2(:,:,n_qq,n_ee) * x1_val)));
                
                % Validation Error
                [~, index_val] = max(x2_val);
                index_val_t =  find(target_val == 1);
                error_val(val_samples,n_qq,n_ee) = index_val ~=  index_val_t;
            end
            
            % Store error
            store_error_val(n_qq,n_ee) = mean(error_val(:,n_qq,n_ee));
            
        end
    end
    
    % Save results to a file
    save(savefile, 'store_error', 'store_W1', 'store_W2', 'store_error_val')
    
end
end
