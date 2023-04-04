clc; clear; close all;
% GRU(2, 2)
% 
% By pytorch:
% input_size = 2
% hidden_size = 2
% num_layers = 1
% sequence_length = 5
% batch_size = 1
% 
% smallest_gru_model = nn.GRU(input_size, hidden_size, num_layers)
% input = torch.randn(sequence_length, batch_size, input_size)
% h0 = torch.randn(num_layers, batch_size, hidden_size)
% output, hn = smallest_gru_model(input, h0)
format long;

input_size = 2;
hidden_size = 2;
num_layers = 1;
sequence_length = 5;
batch_size = 1;

input = zeros(sequence_length, batch_size, input_size);

input(1, :, :) = [-0.046555221080780029296875, -0.434801310300827026367188];
input(2, :, :) = [-1.386179924011230468750000,  1.396894574165344238281250];
input(3, :, :) = [ 0.215961530804634094238281, -1.277410864830017089843750];
input(4, :, :) = [-1.252190589904785156250000,  1.995421409606933593750000];
input(5, :, :) = [ 0.652253806591033935546875, -1.798372864723205566406250];

net = importONNXLayers("./the_smallest_gru.onnx");

% GRU layer in onnx
l = 4;
%GRU Layer
nI = net.Layers(l).InputSize;
nH = net.Layers(l).NumHiddenUnits;

i = 1;
% reset gate:
%   r[t] = sigmoid(Wr * x[t] + Ur * h[t-1] + br)
gru.Wr = net.Layers(l).InputWeights((i-1)*nI+1:i*nI, 1:nH); % weight matrix from intput state x[t] to reset gate r[t]
gru.Ur = net.Layers(l).RecurrentWeights((i-1)*nI+1:i*nI, 1:nH); % weight matrix from memory state h[t-1] to reset gate r[t]
gru.br = net.Layers(l).Bias((i-1)*nH+1:i*nH, 1); % bias vector for reset gate
gru.gr = net.Layers(l).Bias((i+2)*nH+1:(i+3)*nH, 1);

i = i + 1;
% update gate:
%   z[t] = sigmoid(Wz * x[t] + Uz * h[t-1] + bz)
gru.Wz = net.Layers(l).InputWeights((i-1)*nI+1:i*nI, 1:nH); % weight matrix from input state x[t] to update gate z[t]
gru.Uz = net.Layers(l).RecurrentWeights((i-1)*nI+1:i*nI, 1:nH); % weight matrix from memory state h[t-1] to update gate z[t]
gru.bz = net.Layers(l).Bias((i-1)*nH+1:i*nH, 1); % bias vector for reset gate
gru.gz = net.Layers(l).Bias((i+2)*nH+1:(i+3)*nH, 1);

i = i + 1;
% cadidate current state:
%   c[t] = tanh(Wc * x[t] + Uc * (r[t] o h[t-1]) + bc), where o is
%       Hadamard product
gru.Wc = net.Layers(l).InputWeights((i-1)*nI+1:i*nI, 1:nH); % weight matrix from input state x[t] to cadidate current state
gru.Uc = net.Layers(l).RecurrentWeights((i-1)*nI+1:i*nI, 1:nH); % weight matrix from reset gate r[t] and memory state h[t-1] to 
    % candidate current state
gru.bc = net.Layers(l).Bias((i-1)*nH+1:i*nH, 1); % bias vector for reset gate
gru.gc = net.Layers(l).Bias((i+2)*nH+1:(i+3)*nH, 1);

L = GRULayer(gru);
y = L.evaluate(input(:, : ,:));

x = input;
x = permute(x, [3 1 2]);

eps = 0.01;
n = size(x, 2);
SX = [];
X = [];
for i = 1:n
    SX = [SX SparseStar(x(:,i) - eps, x(:, i) + eps)]; % construct sequence of input sets
    X = [X Star(x(:,i) - eps, x(:, i) + eps)]; % construct sequence of input sets
    X(i).Z = [];
end

% Sample input sets and compute output of GRULayer
num_sample = 500;
SampleX = cell(1, n);
for i = 1:n
    SampleX{i} = X(i).sample(num_sample);
end

for i = 1:num_sample
    T = [SampleX{1}(:,i), SampleX{2}(:,i), SampleX{3}(:,i), SampleX{4}(:,i), SampleX{5}(:,i)];
    SampleInput{i} = permute(T, [2 3 1]);
end

for i = 1:num_sample
    SampleY{i} = L.evaluate(SampleInput{i});
end

figure('Name', 'Input')
for i = 1:n
    nexttile;
    plot(X(i));
    s = sprintf('Set %d', i);
    for j = 1:num_sample
        hold on;
        plot(SampleX{i}(1, j), SampleX{i}(2, j), '*k');
    end
    title(s);
end


reachMethod = 'approx-star';
sparse_reachMethod = 'approx-sparse-star';
option = [];
depthReduct = 0;
relaxFactor = 0;
dis_opt = [];
lp_solver = 'glpk';

disp('Output, Option 1');
% disp('Star')
% O1 = L.reach1_pytorch(X, reachMethod, option, relaxFactor, dis_opt, lp_solver);
disp('SparseStar')
S1 = L.reach1_pytorch(SX, sparse_reachMethod, option, relaxFactor, depthReduct, dis_opt, lp_solver);
depthReduct = 4;
S1d = L.reach1_pytorch(SX, sparse_reachMethod, option, relaxFactor, depthReduct, dis_opt, lp_solver);

disp('Plotting Output, Option 1')
figure('Name', 'Output, Option 1')
for t = 1:n
    nexttile;
    hold on;
    plot(S1{t}.getBox);
    
    for i = 1:num_sample
        hold on;
        plot(SampleY{i}(t, 1, 1), SampleY{i}(t, 1, 2), '*k');
    end

    s = sprintf('Set %d', t);
    title(s);
end

disp('Plotting Output with depthReduction, Option 1')
figure('Name', 'Output with depthReduction, Option 1')
for t = 1:n
    nexttile;
    hold on;
    plot(S1d{t}.getBox);
    
    for i = 1:num_sample
        hold on;
        plot(SampleY{i}(t, 1, 1), SampleY{i}(t, 1, 2), '*k');
    end

    s = sprintf('Set %d', t);
    title(s);
end

disp('SpareStar');
[lb, ub] = S1{5}.getRanges();
disp('SparseStar with depthReduction, d_max = 4');
[lbd, ubd] = S1d{5}.getRanges();

disp('lower bound differences');
lbd - lb
disp('upper bound differences');
ubd - ub

disp('SpareStar computation time');
depthReduct = 0;
timeit(@() L.reach1_pytorch(SX, sparse_reachMethod, option, relaxFactor, depthReduct, dis_opt, lp_solver))
disp('SparseStar with depthReduction computation time, d_max = 4');
depthReduct = 4;
timeit(@() L.reach1_pytorch(SX, sparse_reachMethod, option, relaxFactor, depthReduct, dis_opt, lp_solver))