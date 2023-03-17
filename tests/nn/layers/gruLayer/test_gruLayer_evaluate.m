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

% input = zeros(batch_size, input_size, sequence_length);
% 
% input(:, :, 1) = [-0.0465552211, -0.4348013103];
% input(:, :, 2) = [-1.3861799240,  1.3968945742];
% input(:, :, 3) = [-1.3861799240,  1.3968945742];
% input(:, :, 4) = [-1.2521905899,  1.9954214096];
% input(:, :, 5) = [ 0.6522538066, -1.7983728647];

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