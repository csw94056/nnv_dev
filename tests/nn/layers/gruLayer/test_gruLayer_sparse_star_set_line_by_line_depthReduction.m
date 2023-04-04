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
X = [];
for i = 1:n
    X = [X SparseStar(x(:,i) - eps, x(:, i) + eps)]; % construct sequence of input sets
end

% Sample input sets and compute output of GRULayer
num_sample = 500;
SampleX = cell(1, n);
for i = 1:n
    SampleX{i} = X(i).sample(num_sample);
end

Sample_x = cell(1, num_sample);
for i = 1:num_sample
    T = [SampleX{1}(:,i), SampleX{2}(:,i), SampleX{3}(:,i), SampleX{4}(:,i), SampleX{5}(:,i)];
    Sample_x{i} = T; 
    SampleInput{i} = permute(T, [2 3 1]);
end

for i = 1:num_sample
    SampleY{i} = L.evaluate(SampleInput{i});
end

%{
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
%}

reachMethod = 'approx-sparse-star';
option = [];
relaxFactor = 0;
dis_opt = [];
lp_solver = 'linprog';
% compute reach sets
% Y = L.reach(X, reachMethod, option, relaxFactor, dis_opt, lp_solver);

method = reachMethod;

%{
I = X;
n = length(I); % number of sequence
O = cell(1, n); % output reachable set sequence
Wz = L.Wz;
Uz = L.Uz;
bz = L.bz;
gz = L.gz;
Wr = L.Wr;
Ur = L.Ur;
br = L.br;
gr = L.gr;
Wc = L.Wc;
Uc = L.Uc;
bc = L.bc;
gc = L.gc;

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
%     H2 = cell(1, n); % hidden state using LogsigXTansig
%     H3 = cell(1, n); % hidden state using LogsigXIdentity
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

            %%%%%%%%%%%%%% Option 2: LogsigXTansig  & LogsigXIdentity only %%%%%%%%%%%%%%
            %   z[t] = logsig(Wz * x[t] + bz + gz) => logsig(WZ)
            %   r[t] = logsig(Wr * x[t] + br + gr) => logsig(WR) = Rt
            %   c[t] = tanh(Wc * x[t] + bc + r[t] o gc) 
            %        => tansig(WC + Rt o gc) 
            %        = tansig(WC + Rtgc)
            %        = tansig(T) = Ct
            %   h[t] = (1 - z[t]) o c[t]
            %        = c[t] - z[t] o c[t]
            %        => Ct - LogsigXTansig(Wz, T)
            %        => Ct - ZCt

%             ZCt2 = LogsigXTansig.reach(WZ(1), T, method, rF, dis, lps);
%             H2{t} = Ct.Sum(negative(ZCt2));
           
            %%%%%%%%%%%%%% Option 3: LogsigXIdentity only %%%%%%%%%%%%%%
            %   z[t] = sigmoid(Wz * x[t] + bz + gz)
            %   r[t] = sigmoid(Wr * x[t] + br + gr) => Rt
            %   c[t] = tanh(Wc * x[t] + bc + r[t] o gc) 
            %        => tansig(WC + Rt o gc) 
            %        = tansig(WC + Rtgc)
            %        = tansig(T) = Ct
            %   h[t] = (1 - z[t]) o c[t]
            %        => (1 - Zt) c Ct
            %        => InZt o Ct = IdentityXIdentity(InZt, Ct)
%             ZCt3 = LogsigXIdentity.reach(WZ(1), Ct, method, rF, dis, lps);
%             H3{t} = Ct.Sum(negative(ZCt3));

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

            figure('Name', 'GRULayer, t=1');
            nexttile;
            hold on;
            plot(Zt);
            for i = 1:num_sample
                hold on;
                plot(z{i}(1, t), z{i}(2, t), '*k');
            end
            title('Zt, t = 1');

            nexttile;
            hold on;
            plot(Rt);
            for i = 1:num_sample
                hold on;
                plot(r{i}(1, t), r{i}(2, t), '*k');
            end
            title('Rt, t = 1');

%             nexttile;
%             hold on;
%             plot(Rtgc);
%             for i = 1:num_sample
%                 hold on;
%                 plot(rtgc{i}(1, t), rtgc{i}(2, t), '*k');
%             end
%             title('Rtgc, t = 1');

            nexttile;
            hold on;
            plot(T);
            for i = 1:num_sample
                hold on;
                plot(t_{i}(1, t), t_{i}(2, t), '*k');
            end
            title('T, t = 1');
            
            nexttile;
            hold on;
            plot(Ct.getBox);
            for i = 1:num_sample
                hold on;
                plot(c{i}(1, t), c{i}(2, t), '*k');
            end
            title('Ct, t = 1');

%             nexttile;
%             hold on;
%             plot(ZCt2.getBox);
%             for i = 1:num_sample
%                 hold on;
%                 plot(zct{i}(1, t), zct{i}(2, t), '*k');
%             end
%             title('ZCt2, t = 1');

%             nexttile;
%             hold on;
%             plot(ZCt3.getBox);
%             for i = 1:num_sample
%                 hold on;
%                 plot(zct{i}(1, t), zct{i}(2, t), '*k');
%             end
%             title('ZCt3, t = 1');

            nexttile;
            hold on;
            plot(H1{t}.getBox);
            for i = 1:num_sample
                hold on;
                plot(h{i}(1, t), h{i}(2, t), '*k');
            end
            title('H1\{t\}, t = 1');

%             nexttile;
%             hold on;
%             plot(H2{t}.getBox);
%             for i = 1:num_sample
%                 hold on;
%                 plot(h{i}(1, t), h{i}(2, t), '*k');
%             end
%             title('H2\{t\}, t = 1');
% 
%             nexttile;
%             hold on;
%             plot(H3{t}.getBox);
%             for i = 1:num_sample
%                 hold on;
%                 plot(h{i}(1, t), h{i}(2, t), '*k');
%             end
%             title('H3\{t\}, t = 1');

        else
            Ht_1 = H1{t-1};
            
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

            %%%%%%%%%%%%%% Option 2: LogsigXTansig  & LogsigXIdentity only %%%%%%%%%%%%%%
            % z[t] = sigmoid(Wz * x[t] + bz + Uz * h[t-1] + gz)
            %      => sigmoid(WZ + UZ)
            %      = sigmoid(WUz)
            % r[t] = sigmoid(Wr * x[t] + br + Ur * h[t-1] + gr)
            %      => sigmoid(WR + UR)
            %      = sigmoid(WUr)
            % c[t] = tanh(Wc * x[t] + bc + r[t] o (Uc * h[t-1]  + gc))
            %      => tanh(WC + r[t] o UC)
            %      = tanh(WC + LogsigXIdentity(WUr, UC)
            %      = tanh(WC + WUrUc)
            %      = tanh(WUc) = Ct2
            % h[t] = z[t] o h[t-1] + (1 - z[t]) o c[t]
            %      => LogsigXIdentity(WUz, H{t-1}) + c[t] - LogsigXtansig(WUz, WUc)
            %      = ZHt_1 + Ct - ZCt
            %      = ZHt_1Ct + nZCt

%             UZ = Ht_1.affineMap(Uz, []);
%             WUz = WZ(t).Sum(UZ);
% 
%             UC = Ht_1.affineMap(Uc, gc);
%             RtUC = IdentityXIdentity.reach(Rt, UC, method, rF, dis, lps);
%             WUc = WC(t).Sum(RtUC);
% 
%             UC = Ht_1.affineMap(Uc, gc);
%             RtUC = IdentityXIdentity.reach(Rt, UC, method, rF, dis, lps);
%             WUc = WC(t).Sum(RtUC);
%             Ct = TanSig.reach(WUc, method, [], rF, dis, lps);

%             WUrUc = LogsigXIdentity.reach(WUr, UC, method, rF, dis, lps);
%             WUc = WC(t).Sum(WUrUc);
%             Ct2 = TanSig.reach(WUc, method, [], rF, dis, lps);
%             
%             ZHt_1 = LogsigXIdentity.reach(WUz, Ht_1, method, rF, dis, lps);
%             ZCt2 = LogsigXTansig.reach(WUz, WUc, method, rF, dis, lps);
%             Ct2nZCt2 = Ct2.Sum(negative(ZCt2));
%             H2{t} = ZHt_1.Sum(Ct2nZCt2);


            %%%%%%%%%%%%%% Option 3: LogsigXIdentity only %%%%%%%%%%%%%%
            % z[t] = sigmoid(Wz * x[t] + bz + Uz * h[t-1] + gz)
            %      => sigmoid(WZ + UZ)
            %      = sigmoid(WUz)
            % r[t] = sigmoid(Wr * x[t] + br + Ur * h[t-1] + gr)
            %      => sigmoid(WR + UR)
            %      = sigmoid(WUr)
            % c[t] = tanh(Wc * x[t] + bc + r[t] o (Uc * h[t-1]  + gc))
            %      => tanh(WC + r[t] o UC)
            %      = tanh(WC + LogsigXIdentity(WUr, UC)
            %      = tanh(WC + WUrUc)
            %      = tanh(WUc) = Ct
            % h[t] = z[t] o h[t-1] + (1 - z[t]) o c[t]
            %      => LogsigXIdentity(WUz, H{t-1}) + c[t] - LogsigXIdentity(WUz, WUc)
            %      = ZHt_1 + Ct - ZCt
            %      = ZHt_1Ct + nZCt
%             ZCt3 = LogsigXIdentity.reach(WUz, Ct2, method, rF, dis, lps);
%             Ct2nZCt3 = Ct2.Sum(negative(ZCt3));
%             H3{t} = ZHt_1.Sum(Ct2nZCt3);
            
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

%             nexttile;
%             hold on;
%             plot(Ct2.getBox);
%             for i = 1:num_sample
%                 hold on;
%                 plot(c{i}(1, t), c{i}(2, t), '*k');
%             end
%             stitle = sprintf('Ct2, t = %d', t);
%             title(stitle);

            nexttile;
            hold on;
            plot(ZtHt_1.getBox);
            for i = 1:num_sample
                hold on;
                plot(zht{i}(1, t), zht{i}(2, t), '*k');
            end
            stitle = sprintf('ZtHt\_1, t = %d', t);
            title(stitle);
            

%             nexttile;
%             hold on;
%             plot(ZHt_1.getBox);
%             for i = 1:num_sample
%                 hold on;
%                 plot(zht{i}(1, t), zht{i}(2, t), '*k');
%             end
%             stitle = sprintf('ZHt\_1, t = %d', t);
%             title(stitle);

%             nexttile;
%             hold on;
%             plot(ZCt2.getBox);
%             for i = 1:num_sample
%                 hold on;
%                 plot(zct{i}(1, t), zct{i}(2, t), '*k');
%             end
%             stitle = sprintf('ZCt2, t = %d', t);
%             title(stitle);
% 
%             nexttile;
%             hold on;
%             plot(ZCt3.getBox);
%             for i = 1:num_sample
%                 hold on;
%                 plot(zct{i}(1, t), zct{i}(2, t), '*k');
%             end
%             stitle = sprintf('ZCt3, t = %d', t);
%             title(stitle);


            nexttile;
            hold on;
            plot(ZtCt.getBox);
            for i = 1:num_sample
                hold on;
                plot(ztct{i}(1, t), ztct{i}(2, t), '*k');
            end
            stitle = sprintf('ZtCt, t = %d', t);
            title(stitle);

%             nexttile;
%             hold on;
%             plot(Ct2nZCt2.getBox);
%             for i = 1:num_sample
%                 hold on;
%                 plot(ztct{i}(1, t), ztct{i}(2, t), '*k');
%             end
%             stitle = sprintf('Ct2nZCt2, t = %d', t);
%             title(stitle);
% 
% 
%             nexttile;
%             hold on;
%             plot(Ct2nZCt3.getBox);
%             for i = 1:num_sample
%                 hold on;
%                 plot(ztct{i}(1, t), ztct{i}(2, t), '*k');
%             end
%             stitle = sprintf('Ct2nZCt3, t = %d', t);
%             title(stitle);



            nexttile;
            hold on;
            plot(H1{t}.getBox);
            for i = 1:num_sample
                hold on;
                plot(h{i}(1, t), h{i}(2, t), '*k');
            end
            stitle = sprintf('H1\{t\}, t = %d', t);
            title(stitle);

%             nexttile;
%             hold on;
%             plot(H2{t}.getBox);
%             for i = 1:num_sample
%                 hold on;
%                 plot(h{i}(1, t), h{i}(2, t), '*k');
%             end
%             stitle = sprintf('H2\{t\}, t = %d', t);
%             title(stitle);
% 
%             nexttile;
%             hold on;
%             plot(H3{t}.getBox);
%             for i = 1:num_sample
%                 hold on;
%                 plot(h{i}(1, t), h{i}(2, t), '*k');
%             end
%             stitle = sprintf('H3\{t\}, t = %d', t);
%             title(stitle);

%             disp(' ');
        end
    end
    O = H1;
end
%}


% % plot(h{i}(1, t), h{i}(2, t), '*k');
% figure('Name', 'Output')
% for t = 1:n
%     nexttile;
%     hold on;
%     plot(O{t}.getBox);
%     
%     for i = 1:num_sample
%         hold on;
%         plot(h{i}(1, t), h{i}(2, t), '*k');
%     end
% 
%     s = sprintf('Set %d', t);
%     title(s);
% end



for i = 1:num_sample
    SampleY{i} = permute( SampleY{i}, [3 1 2]);
end


%{
figure('Name', 'Output')
for t = 1:n
    nexttile;
    hold on;
    plot(O{t}.getBox);
    
    for i = 1:num_sample
        hold on;
        plot(SampleY{i}(1, t), SampleY{i}(2, t), '*k');
    end

    s = sprintf('Set %d', t);
    title(s);
end
%}

% reachMethod = 'approx-star';
% option = [];
% relaxFactor = 0;
% dis_opt = [];
% lp_solver = 'glpk';
disp('Output, Option 1');
O1 = L.reach1_pytorch(X, reachMethod, option, relaxFactor, dis_opt, lp_solver);
% disp('Output, Option 2');
% O2 = L.reach2_pytorch(X, reachMethod, option, relaxFactor, dis_opt, lp_solver);
% disp('Output, Option 3');
% O3 = L.reach3_pytorch(X, reachMethod, option, relaxFactor, dis_opt, lp_solver);

% O1 = O;
%{%
disp('Plotting Output, Option 1')
figure('Name', 'Output, Option 1')
for t = 1:n
    nexttile;
    hold on;
    plot(O1{t}.getBox);
%     plot(O1{t}.getRanges('glpk'));
    
    for i = 1:num_sample
        hold on;
        plot(SampleY{i}(1, t), SampleY{i}(2, t), '*k');
    end

    s = sprintf('Set %d', t);
    title(s);
end
%}

% disp('Plotting Output, Option 2')
% figure('Name', 'Output, Option 2')
% for t = 1:n
%     nexttile;
%     hold on;
%     plot(O2{t}.getBox);
%     
%     for i = 1:num_sample
%         hold on;
%         plot(SampleY{i}(1, t), SampleY{i}(2, t), '*k');
%     end
% 
%     s = sprintf('Set %d', t);
%     title(s);
% end
% 
% disp('Plotting Output, Option 3')
% figure('Name', 'Output, Option 3')
% for t = 1:n
%     nexttile;
%     hold on;
%     plot(O3{t}.getBox);
%     
%     for i = 1:num_sample
%         hold on;
%         plot(SampleY{i}(1, t), SampleY{i}(2, t), '*k');
%     end
% 
%     s = sprintf('Set %d', t);
%     title(s);
% end

fprintf('%s\n', class(O1{1}));
fprintf('%s\n', lp_solver);
for t = 1:n
    fprintf('sequence: %d\n', t);
    T = table;
%     [T.lb, T.ub] = O1{t}.getRanges(lp_solver);
    T.lb = O1{t}.getMins(1:O1{t}.dim, 'single',[], lp_solver);
    T.ub = O1{t}.getMaxs(1:O1{t}.dim, 'single',[], lp_solver);
    T
end



