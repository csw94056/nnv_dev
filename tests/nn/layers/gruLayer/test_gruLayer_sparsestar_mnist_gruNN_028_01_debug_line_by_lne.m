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
sequence_length = 28;
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
end


% Sample input sets and compute output of GRULayer
num_sample = 500;
SampleX = cell(1, n);
for i = 1:n
    SampleX{i} = X(i).sample(num_sample);
end

Sample_x =  cell(1, num_sample);
for i = 1:n
    for j = 1:num_sample
        T = [];
        for k = 1:sequence_length
            T = [T; SampleX{k}(:, i)]; 
        end
        Sample_x{i} = T;
        SampleInput{i} = permute(T, [2 3 1]);
    end
end




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
lp_solver = 'linprog'; %'glpk';

I = X;

n = length(I); % number of sequence
O = cell(1, n); % output reachable set sequence
Wz = L1.Wz;
Uz = L1.Uz;
bz = L1.bz;
gz = L1.gz;
Wr = L1.Wr;
Ur = L1.Ur;
br = L1.br;
gr = L1.gr;
Wc = L1.Wc;
Uc = L1.Uc;
bc = L1.bc;
gc = L1.gc;

WZ = []; % mapped input set: Wz = Wz*I + bz
WR = []; % mapped input set: Wr = Wr*I + br
WC = []; % mapped input set: Wc = Wc*I + bc

rF = relaxFactor;
dis = dis_opt;
lps = lp_solver;

if strcmp(option, 'parallel') % reachability analysis using star set
else
    for i = 1:n
        if isa(I(i), 'SparseStar') || isa(I(i), 'Star')
            WZ = [WZ I(i).affineMap(Wz, bz + gz)];
            WR = [WR I(i).affineMap(Wr, br + gr)];
            WC = [WC I(i).affineMap(Wc, bc)];
        else
            error('RStar is only supported for GRULayer reachability analysis');
        end
    end

    %%%%% SAMPLES
    [~, s, b] = size(Sample_x{1});
    for i = 1:num_sample
        % update gate:
        z{i} = zeros(nH, s);
        % reset gate:
        r{i} = zeros(nH, s);
        % cadidate current state:
        c{i} = zeros(nH, s);
        % output state
        h{i} = zeros(nH, s);
        
        wz{i} = zeros(nH, s);
        rtgc{i}  = zeros(nH, s);
        t_{i} = zeros(nH, s);
        nwz{i} = zeros(nH, s);
        zct{i} = zeros(nH, s);
        zct_{i} = zeros(nH, s);

        WzP{i} = Wz*Sample_x{i};
        WrP{i} = Wr*Sample_x{i};
        WcP{i} = Wc*Sample_x{i};

        wuz{i} = zeros(nH, s);
        wur{i} = zeros(nH, s);

        zht{i} = zeros(nH, s);
        zct{i} = zeros(nH, s);
        ztct{i} = zeros(nH, s);
    end

    
    %%%%% SAMPELS

    H1 = cell(1, n); % hidden state using IdentityXIdentity
    for t = 1:n
        if t == 1
            %%%%%%%%%%%%%% Option 1: IdentityXIdentity only %%%%%%%%%%%%%%
            %   z[t] = sigmoid(Wz * x[t] + bz + gz) => Zt
            %   r[t] = sigmoid(Wr * x[t] + br + gr) => Rt
            %   c[t] = tanh(Wc * x[t] + bc + r[t] o gc) 
            %        => tansig(WC + Rt o gc) 
            %        = tansig(WC + Rtgc)
            %        = tansig(T) = Ct
            %   h[t] = (1 - z[t]) o c[t]
            %        => (1 - Zt) c Ct
            %        => InZt o Ct = IdentityXIdentity(InZt, Ct)
            
            Zt = LogSig.reach(WZ(1), method, [], rF, dis, lps);
            Rt = LogSig.reach(WR(1), method, [], rF, dis, lps);
            Rtgc = Rt.affineMap(diag(gc), []);
            T = WC(1).Sum(Rtgc);
            Ct = TanSig.reach(T, method, [], rF, dis, lps);
            
            dim = Zt.dim;
            InZt = Zt.affineMap(-eye(dim), ones(dim, 1));
            H1{t} = IdentityXIdentity.reach(InZt, Ct, method, rF, dis, lps);

            %%%%% SAMPLES
            for i = 1:num_sample
                z{i}(:, t) = logsig(WzP{i}(:, t) + bz + gz);
                r{i}(:, t) = logsig(WrP{i}(:, t) + br + gr);
                c{i}(:, t) = tanh(WcP{i}(:, t) + bc + r{i}(:, t) .* gc);
                h{i}(:, t) = (1 - z{i}(:, t)) .* c{i}(:, t);

                rtgc{i}(:, t) = r{i}(:, t) .* gc;
                t_{i}(:, t) = WcP{i}(:, t) + bc + r{i}(:, t) .* gc;
                zct{i}(:, t) = z{i}(:, t) .* c{i}(:, t);
            end
            %%%%% SAMPELS
%             Zt, Rt, T, Ct, H1
            
            T = table;
            [T.lb, T.ub] = Y{i}.getRanges('linprog');

            
        else
%             Ht_1 = H1{t-1};
            if t >= 3
                [lb, ub] = H1{t-1}.getRanges(lps);
                Ht_1 = SparseStar(lb, ub);
            else
                Ht_1 = H1{t-1};
            end
            
            %%%%%%%%%%%%%% Option 1: IdentityXIdentity only %%%%%%%%%%%%%%
            % z[t] = sigmoid(Wz * x[t] + bz + Uz * h[t-1] + gz)
            %      => sigmoid(WZ + UZ) = Zt
            % r[t] = sigmoid(Wr * x[t] + br + Ur * h[t-1] + gr)
            %      => sigmoid(WR + UR) = sigmoid(WUr) = Rt
            % c[t] = tanh(Wc * x[t] + bc + r[t] o (Uc * h[t-1]  + gc))
            %      => tanh(WC + Rt o UC)
            %      = tanh(WC + IdentityXIdentity(Rt, UC))
            %      = tanh(WUc) = Ct1
            % h[t] = z[t] o h[t-1] + (1 - z[t]) o c[t]
            %      => IdentityXIdentity(Zt, H{t-1}) + IdentityXIdentity(1-Zt, Ct)
            %      = IdentityXIdentity(Zt, H{t-1}) + IdentityXIdentity(InZt, Ct)
            %      = ZtHt_1 + ZtCt

            UZ = Ht_1.affineMap(Uz, []);
            WUz = WZ(t).Sum(UZ);
            Zt = LogSig.reach(WUz, method, [], rF, dis, lps);
            
            UR = Ht_1.affineMap(Ur, []);
            WUr = WR(t).Sum(UR);
            Rt = LogSig.reach(WUr, method, [], rF, dis, lps);
            
            UC = Ht_1.affineMap(Uc, gc);
            RtUC = IdentityXIdentity.reach(Rt, UC, method, rF, dis, lps);
            WUc = WC(t).Sum(RtUC);
            Ct1 = TanSig.reach(WUc, method, [], rF, dis, lps);

            ZtHt_1 = IdentityXIdentity.reach(Zt, Ht_1, method, rF, dis, lps);
            InZt = Zt.affineMap(-eye(Zt.dim), ones(Zt.dim, 1));
            ZtCt = IdentityXIdentity.reach(InZt, Ct1, method, rF, dis, lps);
            H1{t} = ZtHt_1.Sum(ZtCt);

            %%%%%%%%%%%%%%%%%%%% Samples %%%%%%%%%%%%%%
            for i = 1:num_sample
                z{i}(:, t) = logsig(WzP{i}(:, t) + bz + Uz*h{i}(:, t-1) + gz);
                r{i}(:, t) = logsig(WrP{i}(:, t) + br + Ur*h{i}(:, t-1) + gr);
                c{i}(:, t) = tanh(WcP{i}(:, t) + bc + r{i}(:, t) .* (Uc*h{i}(:, t-1) + gc));
                h{i}(:, t) = z{i}(:, t) .* h{i}(:, t-1) + (1 - z{i}(:, t)) .* c{i}(:, t);

                wuz{i}(:, t) = WzP{i}(:, t) + bz + Uz*h{i}(:, t-1) + gz;
                wur{i}(:, t) = WrP{i}(:, t) + br + Ur*h{i}(:, t-1) + gr;

                zht{i}(:, t) = z{i}(:, t) .* h{i}(:, t-1);
                zct{i}(:, t) = z{i}(:, t) .* c{i}(:, t);
                ztct{i}(:, t) = (1 - z{i}(:, t)) .* c{i}(:, t);
            end

            ss = sprintf('GRULayer, t=%d', t);
            figure('Name', ss);

            nexttile;
            hold on;
            plot(Ht_1.getBox);
            for i = 1:num_sample
                hold on;
                plot(h{i}(1, t-1), h{i}(2, t-1), '*k');
            end
            stitle = sprintf('H1\{t\}, t = %d', t-1);
            title(stitle);

            
            nexttile;
            hold on;
            plot(WUz.getBox);
            for i = 1:num_sample
                hold on;
                plot(wuz{i}(1, t), wuz{i}(2, t), '*k');
            end
            stitle = sprintf('WUz, t = %d', t);
            title(stitle);

            nexttile;
            hold on;
            plot(WUr.getBox);
            for i = 1:num_sample
                hold on;
                plot(wur{i}(1, t), wur{i}(2, t), '*k');
            end
            stitle = sprintf('WUr, t = %d', t);
            title(stitle);


            nexttile;
            hold on;
            plot(Zt.getBox);
            for i = 1:num_sample
                hold on;
                plot(z{i}(1, t), z{i}(2, t), '*k');
            end
            stitle = sprintf('Zt, t = %d', t);
            title(stitle);

            nexttile;
            hold on;
            plot(Rt.getBox);
            for i = 1:num_sample
                hold on;
                plot(r{i}(1, t), r{i}(2, t), '*k');
            end
            stitle = sprintf('Rt, t = %d', t);
            title(stitle);
            
            nexttile;
            hold on;
            plot(Ct1.getBox);
            for i = 1:num_sample
                hold on;
                plot(c{i}(1, t), c{i}(2, t), '*k');
            end
            stitle = sprintf('Ct1, t = %d', t);
            title(stitle);

            nexttile;
            hold on;
            plot(ZtHt_1.getBox);
            for i = 1:num_sample
                hold on;
                plot(zht{i}(1, t), zht{i}(2, t), '*k');
            end
            stitle = sprintf('ZtHt\_1, t = %d', t);
            title(stitle);
            


            nexttile;
            hold on;
            plot(ZtCt.getBox);
            for i = 1:num_sample
                hold on;
                plot(ztct{i}(1, t), ztct{i}(2, t), '*k');
            end
            stitle = sprintf('ZtCt, t = %d', t);
            title(stitle);


            nexttile;
            hold on;
            plot(H1{t}.getBox);
            for i = 1:num_sample
                hold on;
                plot(h{i}(1, t), h{i}(2, t), '*k');
            end
            stitle = sprintf('H1\{t\}, t = %d', t);
            title(stitle);


%             disp(' ');
        end
    end
    O = H1;
end

% disp('SparsStar time:')
% tic;
% Y = L1.reach1_pytorch(X, reachMethod, option, relaxFactor, dis_opt, lp_solver);
% toc


SampleY = cell(1, sequence_length);
for i = 1:sequence_length
    SampleY{i} = L1.evaluate(SampleInput{i});
end


max_ = -ones(hidden_size, sequence_length);
min_ = ones(hidden_size, sequence_length);
for i = 1:sequence_length
    SampleT{i} = permute( SampleY{i}, [3 1 2]);
end
for i = 1:sequence_length %sequence
    for j = 1:hidden_size %dime
        max_(j,i) = max(SampleT{i}(j, :));
        min_(j,i) = min(SampleT{i}(j, :));
    end
end
T = cell(1, sequence_length);
for i = 1:sequence_length
    [T{i}.lb, T{i}.ub] = Y{i}.getRanges('linprog');
    T{i}.s_lb = min_(:, i);
    T{i}.s_ub = max_(:, i);
end


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



