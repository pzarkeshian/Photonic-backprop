load('NoisyBinary1_RandomSign.mat');
err1 = store_error_val;
load('NoisyBinary2_RandomSign.mat');
err2 = store_error_val;
load('NoisyBinary3_RandomSign.mat');
err3 = store_error_val;
load('NoisyBinary4_RandomSign.mat');
err4 = store_error_val;
load('NoisyBinary5_RandomSign.mat');
err5 = store_error_val;
load('NoisyBinary6_RandomSign.mat');
err6 = store_error_val;
load('NoisyBinary7.mat');
err7 = store_error_val;
load('NoisyBinary8.mat');
err8 = store_error_val;
load('NoisyBinary9.mat');
err9 = store_error_val;
load('NoisyBinary10.mat');
err10 = store_error_val;

E = cat(3,err1,err2,err3,err4,err5,err6,err7,err8,err9,err10);

meanE = mean(E,3);

stdE = std(E,[],3);

start_grid = -4;
end_grid = 0;
N_points = 13;
epsilon_in = logspace(start_grid, end_grid, N_points);
q_in = logspace(start_grid, end_grid, N_points);
[Q, E] = meshgrid(q_in,epsilon_in);

figure();

%% Uncomment for STANDARD DEVIATION GRID
% contourf(Q,E,stdE,'LineStyle', 'none'); colorbar('EastOutside');
% colorbar('EastOutside'); caxis([0 0.35]);
% set(gca, 'XScale', 'log', 'YScale', 'log',...
%     'TickLabelInterpreter','latex', 'FontSize', 18);
% title('Standard Deviation for Noisy Signed Stochastic Updates (10 trials)', ...
%     'Interpreter', 'latex', 'Fontsize', 22);
% xlabel('$q$', 'Interpreter',...
%     'latex', 'Fontsize', 22);
% ylabel('Learning Rate, $\epsilon$', 'Interpreter',...
%     'latex', 'Fontsize', 22);

%% Uncomment for MEAN ERROR GRID
contourf(Q,E,meanE,'LineStyle', 'none');
colorbar('EastOutside'); caxis([0 1])
set(gca, 'XScale', 'log', 'YScale', 'log',...
    'TickLabelInterpreter','latex', 'FontSize', 18);
title('Mean Test Error Rate for Noisy Signed Stochastic Updates (10 trials)', ...
    'Interpreter', 'latex', 'Fontsize', 22);
xlabel('$q$', 'Interpreter',...
    'latex', 'Fontsize', 22);
ylabel('Learning Rate, $\epsilon$', 'Interpreter',...
    'latex', 'Fontsize', 22);
