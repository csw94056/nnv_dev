clc; clear; close all;

% # Hyperparameters
% input_size = 28
% hidden_size = 28
% num_layers = 1
% num_classes = 10
% sequence_length = 28
% learning_rate = 0.005
% batch_size = 64
% num_epochs = 3

% RNN_GRU(
%   (gru): GRU(28, 256, num_layers=2, batch_first=True)
%   (fc): Linear(in_features=7168, out_features=10, bias=True)
% )
% 
% gru.weight_ih_l0: torch.Size([768, 28])
% gru.weight_hh_l0: torch.Size([768, 256])
% gru.bias_ih_l0: torch.Size([768])
% gru.bias_hh_l0: torch.Size([768])
% gru.weight_ih_l1: torch.Size([768, 256])
% gru.weight_hh_l1: torch.Size([768, 256])
% gru.bias_ih_l1: torch.Size([768])
% gru.bias_hh_l1: torch.Size([768])
% fc.weight: torch.Size([10, 7168])
% fc.bias: torch.Size([10])

format long;
input_size = 28;
hidden_size = 28;
num_layers = 1;
sequence_lenght = 28;
batch_size = 64;

net = importONNXLayers("./model_mnist_gruNN_028h_01l.onnx");

load model_mnist_gruNN_028h_01l_input.csv
input_sample = model_mnist_gruNN_028h_01l_input;
x = reshape(input_sample, [1, 28, 28]);
x = permute(x, [3, 1, 2]);
% s is number of sequence
% b is number of batch size
% n is number of input size
% [s, b, n] = size(x);

eps = 0.01;
n = size(x, 1); % size: [28, 1, 28]
X = [];
S = [];
for i = 1:n
    X = [X SparseStar(x(:,i) - eps, x(:, i) + eps)]; % construct sequence of input sets
    S = [S Star(x(:,i) - eps, x(:, i) + eps)]; % construct sequence of input sets
end


% Sample input sets and compute output of GRULayer
% num_sample = 500;
% SampleX = cell(1, n);
% for i = 1:n
%     SampleX{i} = X(i).sample(num_sample);
% end
% 
% Sample_x = cell(1, num_sample);
% for i = 1:num_sample
%     T = [SampleX{1}(:,i), SampleX{2}(:,i), SampleX{3}(:,i), SampleX{4}(:,i), SampleX{5}(:,i)];
%     SampleInput{i} = permute(T, [2 3 1]);
% end




% GRU layer in onnx
gru_l = 5;
fc_l = 8;

%% GRU Layer
l = gru_l;

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

L1 = GRULayer(gru);

% output sample
% y = L1.evaluate(x);

%% FC layer
l = fc_l;
fc.W = net.Layers(l).fc_weight';
fc.b = net.Layers(l).fc_bias;

L2 = FullyConnectedLayer(fc.W, fc.b);

% Reachability Analysis
reachMethod = 'approx-sparse-star';
option = [];
relaxFactor = 0;
dis_opt = [];
lp_solver = 'estimate'; %'glpk';

disp('SparsStar time:')
tic;
Y = L1.reach1_pytorch(X, reachMethod, option, relaxFactor, dis_opt, lp_solver);
toc
disp('Star time:')
tic;
% R = L1.reach1_pytorch(S, reachMethod, option, relaxFactor, dis_opt, lp_solver);
toc
% for i = 1:num_sample
%     SampleY{i} = L1.evaluate(SampleInput{i});
% end

% [lb, ub] = O1{28}.getRanges();
% T = table();
% T.lb = lb;
% T.ub = ub;
% 
% max_ = -ones(28, 1);
% min_ = ones(28, 1);
% for i = 1:num_sample
%     SampleY{i} = permute( SampleY{i}, [3 1 2]);
% 
%     for j = 28:28
%         for k = 1:28
%             if max_ < SampleY{i}(:, j);
%                 max_(k, j) =  SampleY{i}(:, j);
%             end
%             if min_ > SampleY{i}(:, j);
%                 min_(k, j) =  SampleY{i}(:, j);
%             end
%         end
%     end
% end

% disp('Plotting Output, Option 1')
% figure('Name', 'Output, Option 1')
% for t = 1:n
%     nexttile;
%     hold on;
%     plot(O1{t}.getBox);
%     
%     for i = 1:num_sample
%         hold on;
%         plot(SampleY{i}(1, t), SampleY{i}(2, t), '*k');
%     end
% 
%     s = sprintf('Set %d', t);
%     title(s);
% end



n = length(O1);
O1_concate = O1{1};
for i = 2:n
    O1_concate = O1_concate.concatenate(O1{i});
end
O2 = L2.reach(O1_concate, reachMethod, option, relaxFactor, dis_opt, lp_solver);



