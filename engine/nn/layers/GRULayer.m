classdef GRULayer
    % GRULayer is a Gate Recurrent Unit layer class for GruRNN (GRU neural
    % entwork). It contains reachability analysis method using star
    % represantation.

    % author: Sung Woo Choi
    % date: 08/16/2022
    
    properties
        % update gate:
        %   z[t] = sigmoid(Wz * x[t] + bz + Uz * h[t-1] + gz)
        Wz; % weight matrix from input state x[t] to update gate z[t]
        Uz; % weight matrix from memory state h[t-1] to update gate z[t]
        bz; % bias vector for update gate
        gz; %

        % reset gate:
        %   r[t] = sigmoid(Wr * x[t] + bz + Ur * h[t-1] + gr)
        Wr; % weight matrix from intput state x[t] to reset gate r[t]
        Ur; % weight matrix from memory state h[t-1] to reset gate r[t]
        br; % bias vector for reset gate
        gr;

        % cadidate current state:
        %   c[t] = tanh(Wc * x[t] + bc + r[t] o (Uc * h[t-1]  + gc)), where o is
        %       Hadamard product
        Wc; % weight matrix from input state x[t] to cadidate current state
        Uc; % weight matrix from reset gate r[t] and memory state h[t-1] to 
            % candidate current state
        bc; % bias vector for candidate state
        gc;

        % output state
        % h[t] = z[t] o h[t-1] + (1 - z[t]) o c[t]
        %      => sigmoid( ... ) o h[t-1] + (1 - sigmoid( ...)) o tanh( ... )
        %      => ZC + ZH

        nI; % number of input nodes
        nH; % number of nodes at output current state h[t]
%         nL; % number of recurrent layers
%             % E.g., setting num_layers=2 would mean stacking two GRUs 
%             % together to form a stacked GRU, with the second GRU taking in
%             % outputs of the first GRU and computing the final results. 
%             % Default: 1

        option = []; % parallel option, 'parallel' or []
        dis_opt = []; % display option, 'display' or []
        lp_solver = 'glpk'; % lp solver option, 'linprog' or 'glpk'
        relaxFactor = 0; % use only for approx-star method
    end
    
    methods % Constructor - evaluation

        % Constructor     
        function obj = GRULayer(varargin)
            if isstruct(varargin{1})
                % update gate
                if isfield(varargin{1}, 'Wz')
                    obj.Wz = double(varargin{1}.Wz);
                else
                    error('Input should have Wz filed');
                end
                if isfield(varargin{1}, 'Uz')
                    obj.Uz = double(varargin{1}.Uz);
                else
                    error('Input should have Uz filed');
                end
                if isfield(varargin{1}, 'bz')
                    obj.bz = double(varargin{1}.bz);
                else
                    error('Input should have bz filed');
                end
                if isfield(varargin{1}, 'gz')
                    obj.gz = double(varargin{1}.gz);
                else
                    error('Input should have gz filed');
                end
    
                % reset gate
                if isfield(varargin{1}, 'Wr')
                    obj.Wr = double(varargin{1}.Wr);
                else
                    error('Input should have Wr filed');
                end
                if isfield(varargin{1}, 'Ur')
                    obj.Ur = double(varargin{1}.Ur);
                else
                    error('Input should have Ur filed');
                end
                if isfield(varargin{1}, 'br')
                    obj.br = double(varargin{1}.br);
                else
                    error('Input should have br filed');
                end
                if isfield(varargin{1}, 'gr')
                    obj.gr = double(varargin{1}.gr);
                else
                    error('Input should have gr filed');
                end
    
                % cadidate state
                if isfield(varargin{1}, 'Wc')
                    obj.Wc = double(varargin{1}.Wc);
                else
                    error('Input should have Wc filed');
                end
                if isfield(varargin{1}, 'Uc')
                    obj.Uc = double(varargin{1}.Uc);
                else
                    error('Input should have Uh filed');
                end
                if isfield(varargin{1}, 'bc')
                    obj.bc = double(varargin{1}.bc);
                else
                    error('Input should have bh filed');
                end
                if isfield(varargin{1}, 'gc')
                    obj.gc = double(varargin{1}.gc);
                else
                    error('Input should have bz filed');
                end
            else
                error('Input should be a struct array');
            end


            if all(size(obj.Wz) ~= size(obj.Uz)) || size(obj.Wz, 1) ~= size(obj.bz, 1) || size(obj.gz, 1) ~= size(obj.bz, 1)
                error('Inconsistent dimension between weight matrix of update gate and its bias vector')
            end
            if all(size(obj.Wr) ~= size(obj.Ur)) || size(obj.Wr, 1) ~= size(obj.br, 1) || size(obj.gr, 1) ~= size(obj.br, 1)
                error('Inconsistent dimension between weight matrix of reset gate and its bias vector')
            end
            if all(size(obj.Wc) ~= size(obj.Uc)) || size(obj.Wc, 1) ~= size(obj.bc, 1) || size(obj.gc, 1) ~= size(obj.bc, 1)
                error('Inconsistent dimension between weight matrix of cadidate current state and its bias vector')
            end
        
            obj.nI = size(obj.Wz, 2);
            obj.nH = size(obj.Wz, 1);
        end

        % Evaluation method
        function y = evaluate(obj, x)   % evaluation of this layer with a specific vector
            % @x: an input (sequence, batch_size, input_size)
            % @y: an output (sequence, batch_size, input_size)
            
            % author: Sung Woo Choi
            % date: 20/26/2023
            
            % input x is equivalent to batch_first=False for pytorch
            % s is number of sequence
            % b is number of batch size
            % n is number of input size
            [s, b, n] = size(x);

            if n ~= obj.nI
                error('Inconsistent dimension of the input vector and the network input')
            end

            if s < 0
                error('Invalid input sequence')
            end

            % update gate:
            z = zeros(obj.nH, s);
            % reset gate:
            r = zeros(obj.nH, s);
            % cadidate current state:
            c = zeros(obj.nH, s);
            % output state
            h = zeros(obj.nH, s);
            
            x = permute(x, [3 1 2]);

            Wz = obj.Wz*x;
            Wr = obj.Wr*x;
            Wc = obj.Wc*x;
            for t = 1:s
                if t == 1
                    z(:, t) = logsig(Wz(:, t) + obj.bz + obj.gz);
                    r(:, t) = logsig(Wr(:, t) + obj.br + obj.gr);
                    c(:, t) = tanh(Wc(:, t) + obj.bc + r(:, t) .* obj.gc);
                    h(:, t) = (1 - z(:, t)) .* c(:, t);
                else
                    z(:, t) = logsig(Wz(:, t) + obj.bz + obj.Uz*h(:, t-1) + obj.gz);
                    r(:, t) = logsig(Wr(:, t) + obj.br + obj.Ur*h(:, t-1) + obj.gr);
                    c(:, t) = tanh(Wc(:, t) + obj.bc + r(:, t) .* (obj.Uc*h(:, t-1) + obj.gc));
                    h(:, t) = z(:, t) .* h(:, t-1) + (1 - z(:, t)) .* c(:, t);
                end
            end
            
            y = permute(h, [2 3 1]);
        end


    end


    methods % rechability analysis method
        function O = reach1_pytorch_opt(varargin)
            % with IdentityXIdentity only
            % @I: an array of inputs set sequence
            % @method: none
            % @option:  'parallel' use parallel computing
            %           '[]' or not declared -> don't use parallel
            %           computing
            % @O: a cell of output sets sequence, length(O) = length(I)

            % author: Sung Woo Choi
            % date: 02/28/2023

            switch nargin
                case 7
                    obj = varargin{1};
                    I = varargin{2};
                    method = varargin{3};
                    obj.option = varargin{4};
                    obj.relaxFactor = varargin{5}; % only use for approx-star method
                    obj.dis_opt = varargin{6};
                    obj.lp_solver = varargin{7};
                case 6
                    obj = varargin{1};
                    I = varargin{2};
                    method = varargin{3};
                    obj.option = varargin{4};
                    obj.relaxFactor = varargin{5}; % only use for approx-star method
                    obj.dis_opt = varargin{6};
                case 5
                    obj = varargin{1};
                    I = varargin{2};
                    method = varargin{3};
                    obj.option = varargin{4};
                    obj.relaxFactor = varargin{5}; % only use for approx-star method
                case 4
                    obj = varargin{1};
                    I = varargin{2};
                    method = varargin{3};
                    obj.option = varargin{4};
                case 3
                    obj = varargin{1};
                    I = varargin{2};
                    method = varargin{3};
                case 2 
                    obj = varargin{1};
                    I = varargin{2};
                    method = 'approx-star';
                otherwise
                    error('Invalid number of input arguments (should be 1, 2, 3, 4, 5, or 6)');
            end

            if ~strcmp(method, 'rstar') && ...
                ~strcmp(method, 'approx-star') && ...
                ~strcmp(method, 'approx-sparse-star') && ...
                ~strcmp(method, 'relax-star') && ...
                ~strcmp(method, 'abs-dom')
                error('Unknown reachability analysis method');
            end

            n = length(I); % number of sequence
            O = cell(1, n); % output reachable set sequence
            Wz = obj.Wz;
            Uz = obj.Uz;
            bz = obj.bz;
            gz = obj.gz;
            Wr = obj.Wr;
            Ur = obj.Ur;
            br = obj.br;
            gr = obj.gr;
            Wc = obj.Wc;
            Uc = obj.Uc;
            bc = obj.bc;
            gc = obj.gc;

            WZ = []; % mapped input set: Wz = Wz*I + bz
            WR = []; % mapped input set: Wr = Wr*I + br
            WC = []; % mapped input set: Wc = Wc*I + bc

            rF = obj.relaxFactor;
            dis = obj.dis_opt;
            lps = obj.lp_solver;

            if strcmp(obj.option, 'parallel') % reachability analysis using star set
            else
                for i = 1:n
                    if isa(I(i), 'SparseStar') || isa(I(i), 'Star')
                        WZ = [WZ I(i).affineMap(Wz, bz + gz)];
                        WR = [WR I(i).affineMap(Wr, br + gr)];
                        WC = [WC I(i).affineMap(Wc, bc)];
                    else
                        error('Star and SparseStar are only supported for GRULayer reachability analysis');
                    end
                end

                H1 = cell(1, n);
                for t = 1:n
                    if t == 1
                        %   z[t] = sigmoid(Wz * x[t] + bz + gz) => Zt
                        %   r[t] = sigmoid(Wr * x[t] + br + gr) => Rt
                        %   c[t] = tanh(Wc * x[t] + bc + r[t] o gc) 
                        %        => tansig(WC + Rt o gc) 
                        %        = tansig(WC + Rtgc)
                        %        = tansig(T) = Ct
                        %   h[t] = (1 - z[t]) o c[t]
                        %        => (1 - Zt) o Ct
                        %        = InZt o Ct
                        
                        Zt = LogSig.reach(WZ(1), method, [], rF, dis, lps);
                        Rt = LogSig.reach(WR(1), method, [], rF, dis, lps);
                        Rtgc = Rt.affineMap(diag(gc), []);
                        T = WC(1).Sum(Rtgc);
                        Ct = TanSig.reach(T, method, [], rF, dis, lps);

                        InZt = Zt.affineMap(-eye(Zt.dim), ones(Zt.dim, 1));
                        H1{t} = IdentityXIdentity.reach(InZt, Ct, method, rF, dis, lps);
                    else
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
                        %      = ZtHt_1 + IdentityXIdentity(InZt, Ct)
                        %      = ZtHt_1 + ZtCt

                        if t >= 4
                            [lb, ub] = H1{t-1}.getRanges(lps);
                            Ht_1 = SparseStar(lb, ub);
                        else
                            Ht_1 = H1{t-1};
                        end
            
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

                        t
                    end
                end
                O = H1;
            end

        end


        function O = reach1_pytorch(varargin)
            % with IdentityXIdentity only
            % @I: an array of inputs set sequence
            % @method: none
            % @option:  'parallel' use parallel computing
            %           '[]' or not declared -> don't use parallel
            %           computing
            % @O: a cell of output sets sequence, length(O) = length(I)

            % author: Sung Woo Choi
            % date: 02/28/2023

            switch nargin
                case 7
                    obj = varargin{1};
                    I = varargin{2};
                    method = varargin{3};
                    obj.option = varargin{4};
                    obj.relaxFactor = varargin{5}; % only use for approx-star method
                    obj.dis_opt = varargin{6};
                    obj.lp_solver = varargin{7};
                case 6
                    obj = varargin{1};
                    I = varargin{2};
                    method = varargin{3};
                    obj.option = varargin{4};
                    obj.relaxFactor = varargin{5}; % only use for approx-star method
                    obj.dis_opt = varargin{6};
                case 5
                    obj = varargin{1};
                    I = varargin{2};
                    method = varargin{3};
                    obj.option = varargin{4};
                    obj.relaxFactor = varargin{5}; % only use for approx-star method
                case 4
                    obj = varargin{1};
                    I = varargin{2};
                    method = varargin{3};
                    obj.option = varargin{4};
                case 3
                    obj = varargin{1};
                    I = varargin{2};
                    method = varargin{3};
                case 2 
                    obj = varargin{1};
                    I = varargin{2};
                    method = 'approx-star';
                otherwise
                    error('Invalid number of input arguments (should be 1, 2, 3, 4, 5, or 6)');
            end

            if ~strcmp(method, 'rstar') && ...
                ~strcmp(method, 'approx-star') && ...
                ~strcmp(method, 'approx-sparse-star') && ...
                ~strcmp(method, 'relax-star') && ...
                ~strcmp(method, 'abs-dom')
                error('Unknown reachability analysis method');
            end

            n = length(I); % number of sequence
            O = cell(1, n); % output reachable set sequence
            Wz = obj.Wz;
            Uz = obj.Uz;
            bz = obj.bz;
            gz = obj.gz;
            Wr = obj.Wr;
            Ur = obj.Ur;
            br = obj.br;
            gr = obj.gr;
            Wc = obj.Wc;
            Uc = obj.Uc;
            bc = obj.bc;
            gc = obj.gc;

            WZ = []; % mapped input set: Wz = Wz*I + bz
            WR = []; % mapped input set: Wr = Wr*I + br
            WC = []; % mapped input set: Wc = Wc*I + bc

            rF = obj.relaxFactor;
            dis = obj.dis_opt;
            lps = obj.lp_solver;

            if strcmp(obj.option, 'parallel') % reachability analysis using star set
            else
                for i = 1:n
                    if isa(I(i), 'SparseStar') || isa(I(i), 'Star')
                        WZ = [WZ I(i).affineMap(Wz, bz + gz)];
                        WR = [WR I(i).affineMap(Wr, br + gr)];
                        WC = [WC I(i).affineMap(Wc, bc)];
                    else
                        error('Star and SparseStar are only supported for GRULayer reachability analysis');
                    end
                end

                H1 = cell(1, n);
                for t = 1:n
                    if t == 1
                        %   z[t] = sigmoid(Wz * x[t] + bz + gz) => Zt
                        %   r[t] = sigmoid(Wr * x[t] + br + gr) => Rt
                        %   c[t] = tanh(Wc * x[t] + bc + r[t] o gc) 
                        %        => tansig(WC + Rt o gc) 
                        %        = tansig(WC + Rtgc)
                        %        = tansig(T) = Ct
                        %   h[t] = (1 - z[t]) o c[t]
                        %        => (1 - Zt) o Ct
                        %        = InZt o Ct
                        
                        Zt = LogSig.reach(WZ(1), method, [], rF, dis, lps);
                        Rt = LogSig.reach(WR(1), method, [], rF, dis, lps);
                        Rtgc = Rt.affineMap(diag(gc), []);
                        T = WC(1).Sum(Rtgc);
                        Ct = TanSig.reach(T, method, [], rF, dis, lps);

                        InZt = Zt.affineMap(-eye(Zt.dim), ones(Zt.dim, 1));
                        H1{t} = IdentityXIdentity.reach(InZt, Ct, method, rF, dis, lps);
                    else
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
                        %      = ZtHt_1 + IdentityXIdentity(InZt, Ct)
                        %      = ZtHt_1 + ZtCt

                        Ht_1 = H1{t-1};
            
                        UZ = Ht_1.affineMap(Uz, []);
%                         WUz = WZ(t).Sum(UZ);
                        WUz = UZ.Sum(WZ(t));
                        Zt = LogSig.reach(WUz, method, [], rF, dis, lps);
                        
                        UR = Ht_1.affineMap(Ur, []);
%                         WUr = WR(t).Sum(UR);
                        WUr = UR.Sum(WR(t));
                        Rt = LogSig.reach(WUr, method, [], rF, dis, lps);
                        
                        UC = Ht_1.affineMap(Uc, gc);
                        RtUC = IdentityXIdentity.reach(Rt, UC, method, rF, dis, lps);
%                         WUc = WC(t).Sum(RtUC);
                        WUc = RtUC.Sum(WC(t));
                        Ct1 = TanSig.reach(WUc, method, [], rF, dis, lps);
            
                        ZtHt_1 = IdentityXIdentity.reach(Zt, Ht_1, method, rF, dis, lps);
                        InZt = Zt.affineMap(-eye(Zt.dim), ones(Zt.dim, 1));
                        ZtCt = IdentityXIdentity.reach(InZt, Ct1, method, rF, dis, lps);
                        H1{t} = ZtHt_1.Sum(ZtCt);
                        
                        t
%                         if t == 4
%                             break;
%                         end
                    end
                end
                O = H1;
            end

        end

        function O = reach1(varargin)
            % with IdentityXIdentity only

        end

        function O = reach2_pytorch(varargin)
            % with IdentityXIdentity only
            % @I: an array of inputs set sequence
            % @method: none
            % @option:  'parallel' use parallel computing
            %           '[]' or not declared -> don't use parallel
            %           computing
            % @O: a cell of output sets sequence, length(O) = length(I)

            % author: Sung Woo Choi
            % date: 02/28/2023

            switch nargin
                case 7
                    obj = varargin{1};
                    I = varargin{2};
                    method = varargin{3};
                    obj.option = varargin{4};
                    obj.relaxFactor = varargin{5}; % only use for approx-star method
                    obj.dis_opt = varargin{6};
                    obj.lp_solver = varargin{7};
                case 6
                    obj = varargin{1};
                    I = varargin{2};
                    method = varargin{3};
                    obj.option = varargin{4};
                    obj.relaxFactor = varargin{5}; % only use for approx-star method
                    obj.dis_opt = varargin{6};
                case 5
                    obj = varargin{1};
                    I = varargin{2};
                    method = varargin{3};
                    obj.option = varargin{4};
                    obj.relaxFactor = varargin{5}; % only use for approx-star method
                case 4
                    obj = varargin{1};
                    I = varargin{2};
                    method = varargin{3};
                    obj.option = varargin{4};
                case 3
                    obj = varargin{1};
                    I = varargin{2};
                    method = varargin{3};
                case 2 
                    obj = varargin{1};
                    I = varargin{2};
                    method = 'approx-star';
                otherwise
                    error('Invalid number of input arguments (should be 1, 2, 3, 4, 5, or 6)');
            end

            if ~strcmp(method, 'rstar') && ...
                ~strcmp(method, 'approx-star') && ...
                ~strcmp(method, 'approx-sparse-star') && ...
                ~strcmp(method, 'relax-star') && ...
                ~strcmp(method, 'abs-dom')
                error('Unknown reachability analysis method');
            end

            n = length(I); % number of sequence
            O = cell(1, n); % output reachable set sequence
            Wz = obj.Wz;
            Uz = obj.Uz;
            bz = obj.bz;
            gz = obj.gz;
            Wr = obj.Wr;
            Ur = obj.Ur;
            br = obj.br;
            gr = obj.gr;
            Wc = obj.Wc;
            Uc = obj.Uc;
            bc = obj.bc;
            gc = obj.gc;

            WZ = []; % mapped input set: Wz = Wz*I + bz
            WR = []; % mapped input set: Wr = Wr*I + br
            WC = []; % mapped input set: Wc = Wc*I + bc

            rF = obj.relaxFactor;
            dis = obj.dis_opt;
            lps = obj.lp_solver;

            if strcmp(obj.option, 'parallel') % reachability analysis using star set
            else
                for i = 1:n
                    if isa(I(i), 'SparseStar') || isa(I(i), 'Star')
                        WZ = [WZ I(i).affineMap(Wz, bz + gz)];
                        WR = [WR I(i).affineMap(Wr, br + gr)];
                        WC = [WC I(i).affineMap(Wc, bc)];
                    else
                        error('Star and SparseStar are only supported for GRULayer reachability analysis');
                    end
                end

                H2 = cell(1, n);
                for t = 1:n
                    if t == 1
                        %   z[t] = logsig(Wz * x[t] + bz + gz) => logsig(WZ)
                        %   r[t] = logsig(Wr * x[t] + br + gr) => logsig(WR) = Rt
                        %   c[t] = tanh(Wc * x[t] + bc + r[t] o gc) 
                        %        => tansig(WC + Rt o gc) 
                        %        = tansig(WC + Rtgc) = tansig(T) = Ct2
                        %   h[t] = (1 - z[t]) o c[t]
                        %        = c[t] - z[t] o c[t]
                        %        => Ct - logsigXtansig(Wz, T)
                        %        = Ct - ZCt
            
                        Rt = LogSig.reach(WR(1), method, [], rF, dis, lps);
                        Rtgc = Rt.affineMap(diag(gc), []);
                        T = WC(1).Sum(Rtgc);
                        Ct = TanSig.reach(T, method, [], rF, dis, lps);
                        ZCt2 = LogsigXTansig.reach(WZ(1), T, method, rF, dis, lps);
                        H2{t} = Ct.Sum(negative(ZCt2));
                    else
                        % z[t] = sigmoid(Wz * x[t] + bz + Uz * h[t-1] + gz)
                        %      => sigmoid(WZ + UZ) = sigmoid(WUz)
                        % r[t] = sigmoid(Wr * x[t] + br + Ur * h[t-1] + gr)
                        %      => sigmoid(WR + UR) = sigmoid(WUr)
                        % c[t] = tanh(Wc * x[t] + bc + r[t] o (Uc * h[t-1]  + gc))
                        %      => tanh(WC + r[t] o UC)
                        %      = tanh(WC + LogsigXIdentity(WUr, UC)
                        %      = tanh(WC + WUrUc) = tanh(WUc) = Ct2
                        % h[t] = z[t] o h[t-1] + (1 - z[t]) o c[t]
                        %      => LogsigXIdentity(WUz, H{t-1}) + c[t] - LogsigXtansig(WUz, WUc)
                        %      = ZHt_1 + Ct2 - ZCt2
                        %      = ZHt_1 + Ct2nZCt2
                        Ht_1 = H2{t-1};
            
                        UZ = Ht_1.affineMap(Uz, []);
                        WUz = WZ(t).Sum(UZ);

                        UR = Ht_1.affineMap(Ur, []);
                        WUr = WR(t).Sum(UR);
            
                        UC = Ht_1.affineMap(Uc, gc);
%                         RtUC = IdentityXIdentity.reach(Rt, UC, method, rF, dis, lps);
%                         WUc = WC(t).Sum(RtUC);
%                         Ct1 = TanSig.reach(WUc, method, [], rF, dis, lps);
                        WUrUc = LogsigXIdentity.reach(WUr, UC, method, rF, dis, lps);
                        WUc = WC(t).Sum(WUrUc);
                        Ct2 = TanSig.reach(WUc, method, [], rF, dis, lps);
                                    
                        ZHt_1 = LogsigXIdentity.reach(WUz, Ht_1, method, rF, dis, lps);
                        ZCt2 = LogsigXTansig.reach(WUz, WUc, method, rF, dis, lps);
                        Ct2nZCt2 = Ct2.Sum(negative(ZCt2));
                        H2{t} = ZHt_1.Sum(Ct2nZCt2);

                    end
                end
                O = H2;
            end

        end

        function O = reach2(varargin)
            % with LogsigXTansig and LogsigXIdentity

        end

        function O = reach3_pytorch(varargin)
            % with LogsigXTansig and LogsigXIdentity
            % @I: an array of inputs set sequence
            % @method: none
            % @option:  'parallel' use parallel computing
            %           '[]' or not declared -> don't use parallel
            %           computing
            % @O: a cell of output sets sequence, length(O) = length(I)

            % author: Sung Woo Choi
            % date: 02/28/2023

            switch nargin
                case 7
                    obj = varargin{1};
                    I = varargin{2};
                    method = varargin{3};
                    obj.option = varargin{4};
                    obj.relaxFactor = varargin{5}; % only use for approx-star method
                    obj.dis_opt = varargin{6};
                    obj.lp_solver = varargin{7};
                case 6
                    obj = varargin{1};
                    I = varargin{2};
                    method = varargin{3};
                    obj.option = varargin{4};
                    obj.relaxFactor = varargin{5}; % only use for approx-star method
                    obj.dis_opt = varargin{6};
                case 5
                    obj = varargin{1};
                    I = varargin{2};
                    method = varargin{3};
                    obj.option = varargin{4};
                    obj.relaxFactor = varargin{5}; % only use for approx-star method
                case 4
                    obj = varargin{1};
                    I = varargin{2};
                    method = varargin{3};
                    obj.option = varargin{4};
                case 3
                    obj = varargin{1};
                    I = varargin{2};
                    method = varargin{3};
                case 2 
                    obj = varargin{1};
                    I = varargin{2};
                    method = 'approx-star';
                otherwise
                    error('Invalid number of input arguments (should be 1, 2, 3, 4, 5, or 6)');
            end

            if ~strcmp(method, 'rstar') && ...
                ~strcmp(method, 'approx-star') && ...
                ~strcmp(method, 'approx-sparse-star') && ...
                ~strcmp(method, 'relax-star') && ...
                ~strcmp(method, 'abs-dom')
                error('Unknown reachability analysis method');
            end

            n = length(I); % number of sequence
            O = cell(1, n); % output reachable set sequence
            Wz = obj.Wz;
            Uz = obj.Uz;
            bz = obj.bz;
            gz = obj.gz;
            Wr = obj.Wr;
            Ur = obj.Ur;
            br = obj.br;
            gr = obj.gr;
            Wc = obj.Wc;
            Uc = obj.Uc;
            bc = obj.bc;
            gc = obj.gc;

            WZ = []; % mapped input set: Wz = Wz*I + bz
            WR = []; % mapped input set: Wr = Wr*I + br
            WC = []; % mapped input set: Wc = Wc*I + bc

            rF = obj.relaxFactor;
            dis = obj.dis_opt;
            lps = obj.lp_solver;

            if strcmp(obj.option, 'parallel') % reachability analysis using star set
            else
                for i = 1:n
                    if isa(I(i), 'SparseStar') || isa(I(i), 'Star')
                        WZ = [WZ I(i).affineMap(Wz, bz + gz)];
                        WR = [WR I(i).affineMap(Wr, br + gr)];
                        WC = [WC I(i).affineMap(Wc, bc)];
                    else
                        error('SparseStar and Star are only supported for GRULayer reachability analysis');
                    end
                end

                H3 = cell(1, n);
                for t = 1:n
                    if t == 1
                        %   z[t] = logsig(Wz * x[t] + bz + gz) => logsig(WZ)
                        %   r[t] = logsig(Wr * x[t] + br + gr) => logsig(WR) = Rt
                        %   c[t] = tanh(Wc * x[t] + bc + r[t] o gc) 
                        %        => tansig(WC + Rt o gc) 
                        %        = tansig(WC + Rtgc) = tansig(T) = Ct
                        %   h[t] = (1 - z[t]) o c[t]
                        %        = c[t] - z[t] o c[t]
                        %        => Ct - logsigXtansig(Wz, T)
                        %        = Ct - ZCt3
            
                        Rt = LogSig.reach(WR(1), method, [], rF, dis, lps);
                        Rtgc = Rt.affineMap(diag(gc), []);
                        T = WC(1).Sum(Rtgc);
                        Ct = TanSig.reach(T, method, [], rF, dis, lps);
                        ZCt3 = LogsigXIdentity.reach(WZ(1), Ct, method, rF, dis, lps);
                        H3{t} = Ct.Sum(negative(ZCt3));
                    else
                        % z[t] = sigmoid(Wz * x[t] + bz + Uz * h[t-1] + gz)
                        %      => sigmoid(WZ + UZ) = sigmoid(WUz)
                        % r[t] = sigmoid(Wr * x[t] + br + Ur * h[t-1] + gr)
                        %      => sigmoid(WR + UR) = sigmoid(WUr)
                        % c[t] = tanh(Wc * x[t] + bc + r[t] o (Uc * h[t-1]  + gc))
                        %      => tanh(WC + r[t] o UC)
                        %      = tanh(WC + LogsigXIdentity(WUr, UC)
                        %      = tanh(WC + WUrUc) = tanh(WUc) = Ct2
                        % h[t] = z[t] o h[t-1] + (1 - z[t]) o c[t]
                        %      => LogsigXIdentity(WUz, H{t-1}) + c[t] - LogsigXIdentity(WUz, WUc)
                        %      = ZHt_1 + Ct2 - ZCt3
                        %      = ZHt_1 + Ct2nZCt
                        Ht_1 = H3{t-1};
            
                        UZ = Ht_1.affineMap(Uz, []);
                        WUz = WZ(t).Sum(UZ);

                        UR = Ht_1.affineMap(Ur, []);
                        WUr = WR(t).Sum(UR);
            
                        UC = Ht_1.affineMap(Uc, gc);
%                         RtUC = IdentityXIdentity.reach(Rt, UC, method, rF, dis, lps);
%                         WUc = WC(t).Sum(RtUC);
%                         Ct1 = TanSig.reach(WUc, method, [], rF, dis, lps);
                        WUrUc = LogsigXIdentity.reach(WUr, UC, method, rF, dis, lps);
                        WUc = WC(t).Sum(WUrUc);
                        Ct2 = TanSig.reach(WUc, method, [], rF, dis, lps);
                                    
                        ZHt_1 = LogsigXIdentity.reach(WUz, Ht_1, method, rF, dis, lps);
                        ZCt3 = LogsigXIdentity.reach(WUz, Ct2, method, rF, dis, lps);
                        Ct2nZCt3 = Ct2.Sum(negative(ZCt3));
                        H3{t} = ZHt_1.Sum(Ct2nZCt3);

                    end
                end
                O = H3;
            end
        end

        function O = reach3(varargin)
            % with LogsigXTansig and LogsigXIdentity

        end

        function O = reachXX(varargin)
            % @I: an array of inputs set sequence
            % @method: none
            % @option:  'parallel' use parallel computing
            %           '[]' or not declared -> don't use parallel
            %           computing
            % @O: a cell of output sets sequence, length(O) = length(I)

            % author: Sung Woo Choi
            % date: 08/22/2022

            switch nargin
                case 7
                    obj = varargin{1};
                    I = varargin{2};
                    method = varargin{3};
                    obj.option = varargin{4};
                    obj.relaxFactor = varargin{5}; % only use for approx-star method
                    obj.dis_opt = varargin{6};
                    obj.lp_solver = varargin{7};
                case 6
                    obj = varargin{1};
                    I = varargin{2};
                    method = varargin{3};
                    obj.option = varargin{4};
                    obj.relaxFactor = varargin{5}; % only use for approx-star method
                    obj.dis_opt = varargin{6};
                case 5
                    obj = varargin{1};
                    I = varargin{2};
                    method = varargin{3};
                    obj.option = varargin{4};
                    obj.relaxFactor = varargin{5}; % only use for approx-star method
                case 4
                    obj = varargin{1};
                    I = varargin{2};
                    method = varargin{3};
                    obj.option = varargin{4};
                case 3
                    obj = varargin{1};
                    I = varargin{2};
                    method = varargin{3};
                case 2 
                    obj = varargin{1};
                    I = varargin{2};
                    method = 'approx-star';
                otherwise
                    error('Invalid number of input arguments (should be 1, 2, 3, 4, 5, or 6)');
            end

            if ~strcmp(method, 'rstar') && ...
                ~strcmp(method, 'approx-star') && ...
                ~strcmp(method, 'relax-star') && ...
                ~strcmp(method, 'abs-dom')
                error('Unknown reachability analysis method');
            end

            n = length(I); % number of sequence
            O = cell(1, n); % output reachable set sequence
            Wz = obj.Wz;
            Uz = obj.Uz;
            bz = obj.bz;
            gz = obj.gz;
            Wr = obj.Wr;
            Ur = obj.Ur;
            br = obj.br;
            gr = obj.gr;
            Wc = obj.Wc;
            Uc = obj.Uc;
            bc = obj.bc;
            gc = obj.gc;

            WZ = []; % mapped input set: Wz = Wz*I + bz
            WR = []; % mapped input set: Wr = Wr*I + br
            WC = []; % mapped input set: Wc = Wc*I + bc

            rF = obj.relaxFactor;
            dis = obj.dis_opt;
            lps = obj.lp_solver;
            
            if strcmp(obj.option, 'parallel') % reachability analysis using star set
            else
                for i = 1:n
                    if isa(I(i), 'SparseStar') || isa(I(i), 'Star')
                        WZ = [WZ I(i).affineMap(Wz, bz + gz)];
                        WR = [WR I(i).affineMap(Wr, br + br)];
                        WC = [WC I(i).affineMap(Wc, bc + Uc*gc)];
                    else
                        error('Star and SparseStar are only supported for GRULayer reachability analysis');
                    end
                end

                H = cell(1, n);
                for t = 1:n
                    if t == 1
                        %   z[t] = sigmoid(Wz * x[t] + bz + gz) => logsig(WZ)
                        %   r[t] = sigmoid(Wr * x[t] + br + gr) => logsig(WR)
                        %   c[t] = tanh(Wc * x[t] + gc + Uc * gc) => tansig(WC)
                        %   h[t] = (1 - z[t]) o c[t]
                        %        = c[t] - z[t] o c[t]
                        %        => tansig(WC) + logsigXtansig(-Wz, WC)
                        T = TanSig.reach(WC(1), method, [], rF, dis, lps);
                        ZCt = LogsigXTansig.reach(negative(WZ(1)), WC(1), method, rF, dis, lps);
                        H{t} = T.Sum(ZCt);
                    else
                        % z[t] = sigmoid(Wz * x[t] + bz + Uz * h[t-1] + gz)
                        % r[t] = sigmoid(Wr * x[t] + br + Ur * h[t-1] + gr)
                        % c[t] = tanh(Wc * x[t] + bc + Uc * (r[t] o h[t-1]  + gc))
                        %      = tanh(Wc * x[t] + bc + Uc * gc + Uc * (r[t] o h[t-1]))
                        % h[t] = z[t] o h[t-1] + (1 - z[t]) o c[t]

                        Ht_1 = H{t-1};
                        
                        % UZ = Uz * h[t-1]
                        UZ = Ht_1.affineMap(Uz, []);
                        % WZ = Wz * x[t] + bz + gz
                        % z[t] = sigmoid(Wz * x[t] + bz + Uz * h[t-1] + gz)
                        % Zt   = WZ + UZ
                        Zt = WZ(t).Sum(UZ);
                        
                        % UR = Ur * h[t-1]
                        UR = Ht_1.affineMap(Ur, []);
                        % WR = Wr * x[t] + br + gr
                        % r[t] = sigmoid((Wr * x[t] + br + Ur * h[t-1] + gr)
                        % Rt   = WR + UR
                        Rt = WR(t).Sum(UR);
                        
                        % c[t] = tanh(Wc * x[t] + bc + Uc * (r[t] o h[t-1]  + gc))
                        %      = tanh(Wc * x[t] + bc + Uc * gc + Uc * (r[t] o h[t-1]))
                        % RHt: r[t] o h[t-1]
                        RHt_1 = IdentityXIdentity.reach(Rt, Ht_1, method, rF, dis, lps);
                        % UC = Uc * (r[t] o h[t-1])
                        UC = RHt_1.affineMap(Uc, []);
                        % WC = Wc * x[t] + bc + Uc * gc
                        % Ct = WC + UC
                        Ct = WC(t).Sum(UC);

                        % ZCt = sigmoid(Zt) * tanh(Ct)
                        ZCt_1 = LogsigXTansig.reach(Zt, Ct, method, rF, dis, lps);
                        % ZHt = sigmoid(Zt) o Ht_1
                        ZHt_1 = LogsigXIdentity.reach(Zt, Ht_1, method, rF, dis, lps);
                        % h[t] = z[t] o h[t-1] + (1 - z[t]) o c[t]

                        % h[t] = z[t] o h[t-1] + (1 - z[t]) o c[t]
                        %      => sigmoid(Zt) o h[t-1] + (1 - sigmoid(Zt)) o tanh(Ct)
                        %      => ZHt_1 + tanh(Ct) - sigmoid(Zt) o tanh(ct)
                        %      => ZHt_1 + T - ZCt_1
                        %      => ZHt_1 + TZ
                        T = TanSig.reach(Ct, method, rF, dis, lps);
                        TZ = T.Sum(negative(ZCt_1));
                        H{t} = ZHt_1.Sum(TZ);
                    end
                end
                O = H;
            end
        end


    end



    methods(Static)

        function L = rand(nH, nI)
            % @nH: number of hidden units
            % @nI: number of inputs (sequence, batch_size, input_size)
            
            % author: Sung Woo Choi
            % date: 02/26/2023

            if nH <= 0 || nI <= 0
                error('Invalid numbero hidden units or inputs');
                
            end
            
            gru.Wz = rand(nH, nI);
            gru.Uz = rand(nH, nH);
            gru.bz = -rand(nH, 1);
            gru.gz = -rand(nH, 1);

            gru.Wr = rand(nH, nI);
            gru.Ur = rand(nH, nH);
            gru.br = -rand(nH, 1);
            gru.gr = -rand(nH, 1);

            gru.Wc = rand(nH, nI);
            gru.Uc = rand(nH, nH);
            gru.bc = -rand(nH, 1);
            gru.gc = -rand(nH, 1);

            L = GRULayer(gru);
        end

    end

    
end

