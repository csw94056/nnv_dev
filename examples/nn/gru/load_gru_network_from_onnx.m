net = importONNXLayers("./saved/model_mnist_gruNN_028h_01l.onnx");

%GRU Layer
nI = net.Layers(5).InputSize;
nH = net.Layers(5).NumHiddenUnits;

i = 1;
% update gate:
%   z[t] = sigmoid(Wz * x[t] + Uz * h[t-1] + bz)
Wz = net.Layers(5).InputWeights(i:i*nI, nH); % weight matrix from input state x[t] to update gate z[t]
Uz = net.Layers(5).RecurrentWeights(i:i*nI, nH); % weight matrix from memory state h[t-1] to update gate z[t]
bz = net.Layers(5).Bias(i:i*nI, nH) + net.Layers(5).Bias(i:i*nI, nH);% bias vector for update gate

i = i + 1;
% reset gate:
%   r[t] = sigmoid(Wr * x[t] + Ur * h[t-1] + br)
Wr = net.Layers(5).InputWeights(i:i*nI, nH); % weight matrix from intput state x[t] to reset gate r[t]
Ur = net.Layers(5).RecurrentWeights(i:i*nI, nH); % weight matrix from memory state h[t-1] to reset gate r[t]
br; % bias vector for reset gate

i = i + 1;
% cadidate current state:
%   c[t] = tanh(Wc * x[t] + Uc * (r[t] o h[t-1]) + bc), where o is
%       Hadamard product
Wc = net.Layers(5).InputWeights(i:i*nI, nH);; % weight matrix from input state x[t] to cadidate current state
Uc = net.Layers(5).RecurrentWeights(i:i*nI, nH); % weight matrix from reset gate r[t] and memory state h[t-1] to 
    % candidate current state
bc; % bias vector for candidate state

