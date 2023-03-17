classdef LayerS
    % LAYERS is a class that contains only reachability analysis method using
    % stars, this gonna replace Layer class in the future, i.e., delete
    % polyhedron-based reachability analysis method.
    % author: Dung Tran
    % date: 27/2/2019
    
    properties
        W; % weights_mat 
        b; % bias vector 
        f; % activation function;
        N; % number of neurons
        gamma = 0; % used only for leakyReLU layer
        
        option = []; % parallel option, 'parallel' or []
        dis_opt = []; % display option, 'display' or []
        lp_solver = 'glpk'; % lp solver option, 'linprog' or 'glpk'
        relaxFactor = 0; % use only for approx-star method
    end
    
    methods % constructor - evaluation - sampling
        % Constructor
        function obj = LayerS(varargin)
            
            switch nargin
                case 3
                    W = varargin{1};
                    b = varargin{2};
                    f = varargin{3};
                    gamma = 0;
                case 4
                    W = varargin{1};
                    b = varargin{2};
                    f = varargin{3};
                    gamma = varargin{4};
                otherwise
                    error('Invalid number of input arguments, should be 3 or 4');
            end
            
            if size(W, 1) == size(b, 1)
                obj.W = double(W);
                obj.b = double(b);
                obj.N = size(W, 1);
            else
                error('Inconsistent dimensions between Weights matrix and bias vector');
            end
            
            if gamma >= 1
                error('Invalid parameter for leakyReLU, gamma should be <= 1');
            end
                
            obj.f = f;
            obj.gamma = gamma;
               
        end
        
        % Evaluation method
        function y = evaluate(obj, x)  % evaluation of this layer with a specific vector
            if size(x, 1) ~= size(obj.W, 2)
                error('Inconsistent input vector')
            end
            y1 = obj.W * x;
            if size(x, 2) ~= 1
                n = size(x, 2);
                for i=1:n
                    y1(:,i) = y1(:,i) + obj.b;
                end
            else
                y1 = y1 + obj.b;
            end           

            if strcmp(obj.f, 'poslin')
                y = poslin(y1);
            elseif strcmp(obj.f, 'purelin')
                y = y1;
            elseif strcmp(obj.f, 'satlin')
                y = satlin(y1);
            elseif strcmp(obj.f, 'satlins')
                y = satlins(y1);
            elseif strcmp(obj.f, 'tansig')
                y = tansig(y1);
            elseif strcmp(obj.f, 'logsig')
                y = logsig(y1);
            elseif strcmp(obj.f, 'softmax')
                if isa(y1, 'dlarray')
                    y = softmax(y1, 'DataFormat', 'C');
                else
                    y = softmax(y1);
                end
            elseif strcmp(obj.f, 'leakyrelu')
                y = y1;
                y(find(y < 0)) = obj.gamma*y(find(y<0));
            elseif strcmp(obj.f, 'sign')
                y = sign(y1);
            else
                error('Unknown or unsupported activation function');
            end 

        end
        
        % evaluate the value of the layer output with a set of vertices
        function Y = sample(obj, V)
  
            n = size(V, 2);
            Y = [];
            for j=1:n
                y = obj.evaluate(V(:, j));
                Y = [Y y];
            end
        end
        
    end
    
    
    methods % reachability analysis method
        
        function S = reach(varargin)
            % @I: an array of inputs
            % @method: 'exact-star' or 'approx-star' or 'approx-zono' or 'abs-dom', i.e., abstract domain (support
            % later) or 'face-latice', i.e., face latice (support later)
            % @option:  'parallel' use parallel computing
            %           '[]' or not declared -> don't use parallel
            %           computing
            
            % author: Dung Tran
            % date: 27/2/2019
            % update: 7/10/2020 add the relaxed approx-star method for poslin, logsig and tansig
            %         7/18/2020: add dis_play option + lp_solver option
             
            % parse inputs 
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
                otherwise
                    error('Invalid number of input arguments (should be 2, 3, 4, 5, or 6)');
            end
            
            if ~strcmp(method, 'exact-star') && ~strcmp(method, 'approx-star') && ~strcmp(method, 'approx-star-fast') && ~contains(method, 'relax-star') && ~strcmp(method, 'approx-zono') && ~strcmp(method, 'abs-dom') && ~strcmp(method, 'exact-polyhedron') && ~strcmp(method, 'approx-star-split') && ~strcmp(method,'approx-star-no-split')
                error('Unknown reachability analysis method');
            end
            
            if strcmp(method, 'exact-star') && (~strcmp(obj.f, 'purelin') && ~strcmp(obj.f, 'leakyrelu') && ~strcmp(obj.f, 'sign') && ~strcmp(obj.f, 'poslin') && ~strcmp(obj.f, 'satlin') && ~strcmp(obj.f, 'satlins'))
                method = 'approx-star';
                fprintf('\nThe current layer has %s activation function -> cannot compute exact reachable set for the current layer, we use approx-star method instead', obj.f);
            end
            
            n = length(I);
            S = [];
            W1 = obj.W;
            b1 = obj.b;
            f1 = obj.f;
            gamma1 = obj.gamma;
            
            if strcmp(obj.option, 'parallel') % reachability analysis using star set
                
                rF = obj.relaxFactor;
                dis = obj.dis_opt;
                lps = obj.lp_solver;
                parfor i=1:n
                    
                    % affine mapping y = Wx + b;
                    if ~isa(I(i), 'Polyhedron')
                        I1 = I(i).affineMap(W1, b1);
                    else
                        I1 = W1*I(i) + b1;
                    end
                    % apply activation function: y' = ReLU(y) or y' = 

                    if strcmp(f1, 'purelin')
                        S = [S I1];
                    elseif strcmp(f1, 'poslin')
                        S = [S PosLin.reach(I1, method, [], rF, dis, lps)];
                    elseif strcmp(f1, 'sign')
                        S = [S Sign.reach(I1, method, [], rF, dis, lps)];
                    elseif strcmp(f1, 'satlin')
                        S = [S SatLin.reach(I1, method, [], dis, lps)];
                    elseif strcmp(f1, 'satlins')
                        S = [S SatLins.reach(I1, method)];
                    elseif strcmp(f1, 'leakyrelu')
                        S = [S LeakyReLU.reach(I1, gamma1, method, [], rF, dis, lps)];
                    elseif strcmp(f1, 'logsig')
                        S = [S LogSig.reach(I1, method,[], rF, dis, lps)];
                    elseif strcmp(f1, 'tansig')
                        S = [S TanSig.reach(I1, method, [], rF, dis, lps)];
                    elseif strcmp(f1, 'softmax')
                        fprintf("\nSoftmax reachability is neglected in verification");
                        S = [S I1];
                    else
                        error('Unsupported activation function, currently support purelin, poslin(ReLU), satlin, logsig, tansig');
                    end
                 
                end
                
                
            else
                
                for i=1:n
                    % affine mapping y = Wx + b;                     
                    if isa(I(i), 'Polyhedron')
                        I1 = W1*I(i) + b1;
                    else                        
                        I1 = I(i).affineMap(W1, b1);
                    end
                        
                    if strcmp(f1, 'purelin')
                        S = [S I1];
                    elseif strcmp(f1, 'poslin')
                        S = [S PosLin.reach(I1, method, [], obj.relaxFactor, obj.dis_opt, obj.lp_solver)];
                    elseif strcmp(f1, 'sign')
                        S = [S Sign.reach(I1, method, [], obj.relaxFactor, obj.dis_opt, obj.lp_solver)];
                    elseif strcmp(f1, 'satlin')
                        S = [S SatLin.reach(I1, method, [], obj.dis_opt, obj.lp_solver)];
                    elseif strcmp(f1, 'satlins')
                        S = [S SatLins.reach(I1, method)];
                    elseif strcmp(f1, 'leakyrelu')
                        S = [S LeakyReLU.reach(I1, obj.gamma, method, [], obj.relaxFactor, obj.dis_opt, obj.lp_solver)];
                    elseif strcmp(f1, 'logsig')
                        S = [S LogSig.reach(I1, method, [], obj.relaxFactor, obj.dis_opt, obj.lp_solver)];
                    elseif strcmp(f1, 'tansig')
                        S = [S TanSig.reach(I1, method, [], obj.relaxFactor, obj.dis_opt, obj.lp_solver)];
                    elseif strcmp(f1, 'softmax')
                        fprintf("\nSoftmax reachability is neglected in verification");
                        S = [S I1];
                    else
                        error('Unsupported activation function, currently support purelin, poslin(ReLU), satlin, satlins, logsig, tansig, softmax');
                    end

                end
                
           
                
            end
                
            
            
        end
        
    end
    
    
    methods % flattening a layer into a sequence of operation
        
        
        function Ops = flatten(obj, reachMethod)
            % @reachMethod: reachability method
            % @Ops: an array of operations for the reachability of
            % the layer
            
            % author: Dung Tran
            % date: 1/18/2020
            
            O1 = Operation('AffineMap', obj.W, obj.b);
            
            
            if strcmp(obj.f, 'poslin')
                
                if strcmp(reachMethod, 'exact-star')
                    O2(obj.N) = Operation;
                    for i=1:obj.N
                        O2(i) = Operation('PosLin_stepExactReach', i);
                    end
                elseif strcmp(reachMethod, 'approx-star')
                    O2 = Operation('PosLin_approxReachStar');
                elseif strcmp(reachMethod, 'approx-zono')
                    O2 = Operation('PosLin_approxReachZono');
                elseif strcmp(reachMethod, 'abs-dom')
                    O2 = Operation('PosLin_approxReachAbsDom');
                end
                
            elseif strcmp(obj.f, 'satlin')
                
                if strcmp(reachMethod, 'exact-star')
                    O2(obj.N) = Operation;
                    for i=1:obj.N
                        O2(i) = Operation('SatLin_stepExactReach', i);
                    end
                elseif strcmp(reachMethod, 'approx-star')
                    O2 = Operation('SatLin_approxReachStar');
                elseif strcmp(reachMethod, 'approx-zono')
                    O2 = Operation('SatLin_approxReachZono');
                elseif strcmp(reachMethod, 'abs-dom')
                    O2 = Operation('SatLin_approxReachAbsDom');
                end
                
            elseif strcmp(obj.f, 'satlins')
                
                if strcmp(reachMethod, 'exact-star')
                    O2(obj.N) = Operation;
                    for i=1:obj.N
                        O2(i) = Operation('SatLins_stepExactReach', i);
                    end
                elseif strcmp(reachMethod, 'approx-star')
                    O2 = Operation('SatLins_approxReachStar');
                elseif strcmp(reachMethod, 'approx-zono')
                    O2 = Operation('SatLins_approxReachZono');
                elseif strcmp(reachMethod, 'abs-dom')
                    O2 = Operation('SatLins_approxReachAbsDom');
                end
                
            elseif strcmp(obj.f, 'purelin')
                
                O2 = [];
                
            elseif strcmp(obj.f, 'logsig')
                
                if strcmp(reachMethod, 'exact-star')
                    error('\nCannot do exact analysis for layer with logsig activation function, use approximate method');
                elseif strcmp(reachMethod, 'approx-star')
                    O2 = Operation('LogSig_approxReachStar');
                elseif strcmp(reachMethod, 'approx-zono')
                    O2 = Operation('LogSig_approxReachZono');
                elseif strcmp(reachMethod, 'abs-dom')
                    O2 = Operation('LogSig_approxReachAbsDom');
                end
                
            elseif strcmp(obj.f, 'tansig')
                
                
                if strcmp(reachMethod, 'exact-star')
                    error('\nCannot do exact analysis for layer with logsig activation function, use approximate method');
                elseif strcmp(reachMethod, 'approx-star')
                    O2 = Operation('TanSig_approxReachStar');
                elseif strcmp(reachMethod, 'approx-zono')
                    O2 = Operation('TanSig_approxReachZono');
                elseif strcmp(reachMethod, 'abs-dom')
                    O2 = Operation('TanSig_approxReachAbsDom');
                end
                
            end
                           
            Ops = [O1 O2];
            
        end
        
    end
    
    
    
end

