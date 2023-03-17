% author: Sung Woo Choi
% date: 03/03/2023

clc; clear; close all;
format long;
pink = [double(0xFA)/double(0xFF), double(0x86)/double(0xC4), double(0xFA)/double(0xFF)];

%%
% randomly generate state bounds of each neuron (upper bounds are always bigger than lower bounds)
% range for input set assuming x[t] for t = 2
dim = 2;
%% case l <=0 and u > 0
% 2-dimensional convex set
xl = 1-5.*rand(dim,1);
xu = xl+5.*rand(dim,1);

%% case l >= 0
% 2-dimensional convex set
% xl = 5.*rand(dim,1);
% xu = xl+5.*rand(dim,1);
% % range for hidden set assuming h[t-1] for t = 2
% % 2-dimensional convex set
% hl = 5.*rand(dim,1);
% hu = hl; %+1.*rand(dim,1);


xl = [-3.338024164687948; 0.693344000046470];
xu = [0.538410196831187; 1.197722629826560];

X = SparseStar(xl, xu);
S = Star(xl, xu);

% create sample of input set and hidden set
num_sample = 500;
SampleX = X.sample(num_sample);

fprintf('logsig(X) \n');
figure;
nexttile;
plot(X, 'b');
hold on;
for i = 1:num_sample
    plot(SampleX(1, i), SampleX(2, i), '*k');
end
title('initial input, hidden sets');

% Apply LogSig function
LX = LogSig.sparse_multiStepLogSig_NoSplit(X, [], 'glpk');
LS = LogSig.multiStepLogSig_NoSplit(S, [], 'glpk');
SampleY = logsig(SampleX);

nexttile;
plot(LX, 'r');
hold on;
for i = 1:num_sample
    plot(SampleY(1, i), SampleY(2, i), 'ok');
end
nexttile;
plot(LS, 'r');
hold on;
for i = 1:num_sample
    plot(SampleY(1, i), SampleY(2, i), 'ok');
end
title('after LogSig')