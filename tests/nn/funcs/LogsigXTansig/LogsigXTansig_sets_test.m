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


% xl = [-0.032155110369042; -0.693018376386914];
% xu = [-0.032155110369042; -0.693018376386914];
% hl = [-1.870628109941787; -1.434662197148230];
% hu = [-0.559533181489582; 1.463305117617365];


% hl = [-0.032155110369042; -0.693018376386914];
% hu = [-0.032155110369042; -0.693018376386914];
% xl = [-1.870628109941787; -1.434662197148230];
% xu = [-0.559533181489582; 1.463305117617365];
% 
% xl = [-2.271763068167633; -1.197773909791377];
% xu = [-2.271763068167633; -1.197773909791377];
% hl = [-3.410892202246511; -2.475708496999521];
% hu = [-3.108529568187827; 0.630465586255969];


%% case l >= 0
% 2-dimensional convex set
% xl = 5.*rand(dim,1);
% xu = xl+5.*rand(dim,1);
% % range for hidden set assuming h[t-1] for t = 2
% % 2-dimensional convex set
% hl = 5.*rand(dim,1);
% hu = hl; %+1.*rand(dim,1);

X = Star(xl, xu);
H = Star(hl, hu);

% create sample of input set and hidden set
num_sample = 500;
SampleX = X.sample(num_sample);
SampleH = H.sample(num_sample);

fprintf('logsig(X) * tansig(H) \n');
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


% T = Star(xl, xu);
% T1 = LogSig.reach(T);
% T2 = T1.affineMap(diag(tansig(hl)), []);

% T = Star(hl, hu);
% T1 = TanSig.reach(T);
% T2 = T1.affineMap(diag(logsig(xl)), []);


S = X.Sum(H);
% Apply LogsigXTansig function
SampleY = zeros(dim, num_sample);
for i = 1:dim
    S = LogsigXTansig.stepLogsigXTansig(S, X, H, i, [], 'linprog');
    for j = 1:num_sample
        SampleY(:, j) = LogsigXTansig.f(SampleX(:, j), SampleH(:, j));
    end
end
nexttile;
hold on;
% plot(T2, 'g');
plot(S, 'r');
for i = 1:num_sample
    plot(SampleY(1, i), SampleY(2, i), 'ok');
end
title('after LogsigXTansig')

