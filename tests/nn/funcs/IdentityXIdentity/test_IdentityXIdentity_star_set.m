% author: Sung Woo Choi
% date: 02/23/2023

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
% range for hidden set assuming h[t-1] for t = 2
% 2-dimensional convex set
hl = 1-5.*rand(dim,1);
hu = hl+5.*rand(dim,1);

%% case l >= 0
% 2-dimensional convex set
% xl = 5.*rand(dim,1);
% xu = xl+5.*rand(dim,1);
% % range for hidden set assuming h[t-1] for t = 2
% % 2-dimensional convex set
% hl = 5.*rand(dim,1);
% hu = hl+5.*rand(dim,1);

X = Star(xl, xu);
H = Star(hl, hu);

% create sample of input set and hidden set
num_sample = 500;
SampleX = X.sample(num_sample);
SampleH = H.sample(num_sample);

fprintf('X * H \n');
figure;
nexttile;
plot(X, 'b');
hold on;
plot(H, 'r');
for i = 1:num_sample
    plot(SampleX(1, i), SampleX(2, i), '*k');
    plot(SampleH(1, i), SampleH(2, i), 'og');
end
title('initial input, hidden sets');


S = X.Sum(H);
Si = S;
% Apply LogsigXTansig function
SampleY = zeros(dim, num_sample);
S = IdentityXIdentity.multiStepIdentityXIdentity(S, X, H, [], 'glpk');
for i = 1:dim
    Si = IdentityXIdentity.stepIdentityXIdentity(Si, X, H, i, [], 'glpk');
    for j = 1:num_sample
        SampleY(:, j) = IdentityXIdentity.f(SampleX(:, j), SampleH(:, j));
    end
end
nexttile;
plot(S, 'r');
hold on;
for i = 1:num_sample
    plot(SampleY(1, i), SampleY(2, i), 'ok');
end
title('after IdentityXIdentity')
