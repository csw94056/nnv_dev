classdef SparseStar
    % SparseStar set class
    %
    properties
        A = []; % affine matrix
        C = []; % constraint matrix
        d = []; % constraint vector
        dim = 0; % dimension of star set
        nVar = 0; % number of variables in the constraints

        pred_lb = []; % lower bound vector of predicate variable
        pred_ub = []; % upper bound vector of predicate variable
        
        state_lb = []; % lower bound of state variables
        state_ub = []; % upper bound of state variables
        
        lps = []; %'linprog'; % options: 'linprog', 'glpk', or 'estimate'
        par = []; %'single'; % ottions: 'singple', or 'parallel'
    end


    methods
        % constructor
        function obj = SparseStar(varargin)
            % @V: bassic matrix
            % @C: constraint matrix
            % @d: constraint vector
            
            switch nargin
                
                case 7
                    A = varargin{1};
                    C = varargin{2};
                    d = varargin{3};
                    pred_lb = varargin{4};
                    pred_ub = varargin{5};
                    state_lb = varargin{6};
                    state_ub = varargin{7};
                    
                    [nA, mA] = size(A);
                    [nC, mC] = size(C);
                    [nd, md] = size(d);
                    [n1, m1] = size(pred_lb);
                    [n2, m2] = size(pred_ub);
                    [n3, m3] = size(state_lb);
                    [n4, m4] = size(state_ub);
                    
%                     if mA ~= mC + 1
%                         error('Inconsistency between basic matrix and constraint matrix');
%                     end

                    if nC ~= nd
                        error('Inconsistency between constraint matrix and constraint vector');
                    end

                    if md ~= 1
                        error('constraint vector should have one column');
                    end
                    
                    if m1 ~=1 || m2 ~=1 
                        error('predicate lower- or upper-bounds vector should have one column');
                    end
                    
                    if n1 ~= n2 || n1 ~= mC
                        error('Inconsistency between number of predicate variables and predicate lower- or upper-bounds vector');
                    end
                    
                    if n3 ~= nA || n4 ~= nA
                        error('Inconsistent dimension between lower bound and upper bound vector of state variables and matrix V');
                    end
                    
                    if m3 ~= 1 || m4 ~= 1
                        error('Invalid lower bound or upper bound vector of state variables');
                    end

                    obj.A = A;
                    obj.C = C;
                    obj.d = d;
                    
                    obj.dim = nA;
                    obj.nVar = mC;
                    obj.pred_lb = pred_lb;
                    obj.pred_ub = pred_ub;
                    obj.state_lb = state_lb;
                    obj.state_ub = state_ub;

                
                case 5
                    A = varargin{1};
                    C = varargin{2};
                    d = varargin{3};
                    pred_lb = varargin{4};
                    pred_ub = varargin{5};
                    
                    [nA, mA] = size(A);
                    [nC, mC] = size(C);
                    [nd, md] = size(d);
                    [n1, m1] = size(pred_lb);
                    [n2, m2] = size(pred_ub);

%                     if mA ~= mC + 1
%                         error('Inconsistency between basic matrix and constraint matrix');
%                     end

                    if nC ~= nd
                        error('Inconsistency between constraint matrix and constraint vector');
                    end

                    if md ~= 1
                        error('constraint vector should have one column');
                    end
                    
                    if (m1 ~= 0 && m2~=0) && (m1 ~=1 || m2 ~=1) 
                        error('predicate lower- or upper-bounds vector should have one column');
                    end
                    
                    if (n1 ~=0 && n2 ~= 0) && (n1 ~= n2 || n1 ~= mC)
                        error('Inconsistency between number of predicate variables and predicate lower- or upper-bounds vector');
                    end

                    obj.A = A;
                    obj.C = C;
                    obj.d = d;
                    obj.dim = nA;
                    obj.nVar = mC;
                    obj.pred_lb = pred_lb;
                    obj.pred_ub = pred_ub;
                
                case 3
                    A = varargin{1};
                    C = varargin{2};
                    d = varargin{3};
                    [nA, mA] = size(A);
                    [nC, mC] = size(C);
                    [nd, md] = size(d);
                    
                    
                    if mA ~= mC + 1
                        error('Inconsistency between basic matrix and constraint matrix');
                    end

                    if nC ~= nd
                        error('Inconsistency between constraint matrix and constraint vector');
                    end

                    if md ~= 1
                        error('constraint vector should have one column');
                    end

                    obj.A = A;
                    obj.C = C;
                    obj.d = d;
                    
                    obj.dim = nA;
                    obj.nVar = mC;
                                                            
                case 2
                    % construct star from lower bound and upper bound
                    % vector
                    lb = varargin{1};
                    ub = varargin{2};

                    [n1, m1] = size(lb);
                    [n2, m2] = size(ub);
                    
                    if m1 ~= 1 || m2 ~= 1
                        error('lb and ub should be a vector');
                    end
                    
                    if n1 ~= n2
                        error('Inconsistent dimensions between lb and ub');
                    end

                    dim = n1;
                    center = 0.5 * (lb + ub);
                    vec = 0.5 * (ub - lb);

                    if norm(vec) == 0
                        vec = zeros(dim, 1);
                    end
                    
                    %2*n/n*(n+1) > 0.69299
                    if 2 > 0.69299*(n1+1)
                        obj.A = [sparse(center), spdiags(vec, 0, dim, dim)];
                    else
                        obj.A = [center, diag(vec)];
                    end
                    obj.C = zeros(1, dim); % initiate an obvious constraint
                    obj.d = 0;
                    obj.dim = dim;
                    obj.nVar = dim;
                    obj.state_lb = -ones(dim, 1);
                    obj.state_ub = ones(dim, 1);
                    obj.pred_lb = obj.state_lb;
                    obj.pred_ub = obj.state_ub;
                 
                case 1 % accept a polyhedron as an input and transform to a star
                    I = varargin{1};
                    if ~isa(I, 'Polyhedron')
                        error('Input set is not a polyhedron');
                    end
                    
                    c = sparse(I.Dim, 1);
                    V1 = speye(I.Dim);
                    V = [c V1];
                    if isempty(I.Ae)    
                        obj = SparseStar(V, I.A, I.b);
                    else
                        A1 = [I.Ae; -I.Ae];
                        b1 = [I.be; -I.be];
                        obj = Star(V, [I.A; A1], [I.b; b1]);
                    end
                    [lb, ub] = obj.getRanges;
                    obj.pred_lb = lb;
                    obj.pred_ub = ub;
                
                case 0
                    % create empty Star (for preallocation an array of star)
                    obj.A = [];
                    obj.C = [];
                    obj.d = [];
                    obj.dim = 0;
                    obj.nVar = 0;

                otherwise
                    error('Invalid number of input arguments (should be 0 or 2 or 3 or 5)');
            end

        end

        % affine mapping of star set S = Wx + b;
        function S = affineMap(obj, W, b)
            % @W: mapping matrix
            % @b: mapping vector
            % @S: new SparseStar set
            
            if size(W, 2) ~= obj.dim
                error('Inconsistency between the affine mapping matrix and dimension of the SparseStar set');
            end
            
            if ~isempty(b)
                if size(b, 1) ~= size(W, 1)
                    error('Inconsistency between the mapping vec and mapping matrix');
                end

                if size(b, 2) ~= 1
                    error('Mapping vector should have one column');
                end

                newA = W * obj.A;
                newA(:, 1) = newA(:, 1) + b;
            else
                newA = W * obj.A;
            end
            
            S = SparseStar(newA, obj.C, obj.d, obj.pred_lb, obj.pred_ub);
        end

        function f = V(obj)
            mA = size(obj.A, 2);
            % check sparcity
            if nnz(obj.A) > 0.5*obj.dim*obj.nVar
                % sparse
                f = [sparse(obj.A(:, 1)), sparse(obj.dim, obj.nVar+1-mA), sparse(obj.A(:, 2:mA))];
            else
                % not sparse
                f = [full(obj.A(:, 1)), zeros(obj.dim, obj.nVar+1-mA), full(obj.A(:, 2:mA))];
            end
                
        end

        function f = X(obj)
            mA = size(obj.A, 2);
            nPred = mA-1;
            % check sparcity
            if nnz(obj.A(:, 2:mA)) > 0.5*obj.dim*nPred
                % sparse
                f = [sparse(obj.dim, obj.nVar+1-mA), sparse(obj.A(:, 2:mA))];
            else
                % not sparse
                f = [zeros(obj.dim, obj.nVar+1-mA), full(obj.A(:, 2:mA))];
            end
        end

        function f = c(obj)
            % check sparcity
            if nnz(obj.A(:, 1)) > 0.5*obj.dim
                % sparse
                f = sparse(obj.A(:, 1));
            else
                % not sparse
                f = full(obj.A(:, 1));
            end
        end

        % check is empty set
        function bool = isEmptySet(obj)
            % author: Sung Woo Choi
            % date: 03/07/2023
            if nnz(obj.C) == 0
                C = sparse(1, obj.dim); % initiate an obvious constraint
                d = 0;
            else
                C = obj.C;
                d = obj.d;
            end

            if isempty(obj.lps)
                lps = 'linprog';
            else
                lps = obj.lps;
            end

            f = sparse(zeros(1, obj.nVar));
            if strcmp(lps, 'linprog')
                options = optimoptions(@linprog, 'Display','none'); 
                options.OptimalityTolerance = 1e-10; % set tolerance
%                 [~, ~, exitflag, ~] = linprog(f, C, d, [], [], obj.pred_lb, obj.pred_ub, options);
                [~, ~, exitflag, ~] = linprog(full(f), full(C), full(d), [], [], obj.pred_lb, obj.pred_ub, options);
                if exitflag == 1
                    bool = 0;
                elseif exitflag == -2 || exitflag == -9 || exitflag == -5 || exitflag == 0
                    bool = 1;
                else
                    error('Error, exitflag = %d', exitflag);
                end
            elseif strcmp(lps, 'glpk')
                [~, ~, exitflag, ~] = glpk(full(f), full(C), d, obj.pred_lb, obj.pred_ub);
                if exitflag == 5 || exitflag == 2
                    bool = 0;
                else
                    error('Cannot find an optimal solution, exitflag = %d', exitflag);
                end
                
            else
                error('Unknown lp solver, should be glpk or linprog'); 
            end
            
        end

        % check if a star set contain a point
        function bool = contains(obj, s)
            % @s: a star point
            % @bool: = 1 star set contains s, else no

            % author: Sung Woo Choi
            % date: 03/08/2023
            
            if size(s,1) ~= obj.dim
                error('Dimension mismatch');
            end
            if size(s,2) ~= 1
                error('Invalid star point');
            end
            
            mA = size(obj.A, 2);
            A = full(obj.C);
            b = obj.d;
            Ae = [zeros(obj.dim, obj.nVar+1-mA), obj.A(:, 2:obj.nVar+1)];
            be = s - obj.A(:,1);
            
            P = Polyhedron('A', A, 'b', b, 'Ae', Ae, 'be', be, 'lb', obj.pred_lb, 'ub', obj.pred_ub);
            
            bool = ~P.isEmptySet;
                     
        end


        % sampling a sparse star set
        function V = sample(obj, N)
            % @N: number of points in the samples
            % @V: a set of at most N sampled points in the star set 
            
            if N < 1
                error('Invalid number of samples');
            end
            
            [lb, ub] = obj.getRanges('linprog');
            if ~isempty(lb) && ~isempty(ub)
                
                X = cell(1, obj.dim);
                V1 = [];
                for i=1:obj.dim
                    X{1, i} = (ub(i) - lb(i)).*rand(2*N, 1) + lb(i);
                    V1 = vertcat(V1, X{1, i}');
                end
                                
                V = [];
                for i=1:2*N
                    if obj.contains(V1(:, i))
                        V = [V V1(:, i)];
                    end
                end
                
                if size(V, 2) > N               
                    V = V(:, 1:N);
                end             
                
            end
                     
        end


        % New Minkowski Sum (used for Recurrent Layer reachability)
        function S = Sum(obj, X)
            % @X: another star with same dimension
            % @S: new star
            
            if ~isa(X, 'SparseStar')
                error('Input matrix is not a Star');
            else
                if X.dim ~= obj.dim
                    error('Input star and current star have different dimensions');
                end
            end

            mOA = size(obj.A, 2);
            mXA = size(X.A, 2);
            
            mZ = obj.nVar + X.nVar + 2 - mOA - mXA;
            
            A3 = [obj.A(:, 2:mOA), X.A(:, 2:mXA)];
            [n3, m3] = size(A3);
            new_c = obj.A(:, 1) + X.A(:, 1);
            
            if nnz(A3) < 0.5 * (mZ*obj.dim + n3*m3)
                new_A = [sparse(new_c), sparse(A3)];
            else
                new_A = [new_c, A3];
            end

            mOC = obj.nVar+1-mOA;
            mOX = X.nVar+1-mXA;
            
%             if mOC > 0
%                 % obj has no zero predicate
%                 OC1 = obj.C(:, 1:mOC);
%                 OC2 = obj.C(:, mOC+1:obj.nVar);
%                 Od = obj.d;
%             else
%                 OC1 = [];
%                 OC2 = [];
%                 Od = [];
%             end
%             
%             if mOX > 0
%                 % X has no zero predicate
%                 XC1 = X.C(:, 1:mOX);
%                 XC2 = X.C(:, mOX+1:X.nVar);
%                 Xd = X.d;
%             else
%                 XC1 = [];
%                 XC2 = [];
%                 Xd = [];
%             end
%             
%             if ~isempty(OC2) && ~isempty(XC2)
%                 new_C = [blkdiag(OC1, XC1), blkdiag(OC2, XC2)];
%             elseif isempty(OC2) && ~isempty(XC2)
%                 nXC1 = size(XC1, 1);
%                 new_C = [XC1, sparse(nXC1, obj.nVar), XC2];
%             elseif ~isempty(OC2) && isempty(XC2)
%                 nOC1 = size(OC1, 1);
%                 new_C = [OC1, sparse(nOC1, X.nVar), XC2];
%             else
%                 new_C = [sparse(1, ojb.nVar), OC2, sparse(1, X.nVar)];
%             end
            
            OC1 = obj.C(:, 1:mOC);
            OC2 = obj.C(:, mOC+1:obj.nVar);
            Od = obj.d;

            XC1 = X.C(:, 1:mOX);
            XC2 = X.C(:, mOX+1:X.nVar);
            Xd = X.d;

            new_C = [blkdiag(OC1, XC1), blkdiag(OC2, XC2)];
            new_d = [Od; Xd];

            if mOC == 0
                % either obj has no C or both (obj and X) have no C
                % remove first row zero
                nC = size(new_C, 1);
                new_C = new_C(2:nC, :);
                new_d = new_d(2:nC, :);
            elseif mOX == 0
                % X has no C
                nC1 = size(OC1, 1)+1;
                new_C(nC1,:) = [];
                new_d(nC1) = [];
            end

            if ~isempty(obj.pred_lb) && ~isempty(X.pred_lb)
                mOd = obj.nVar+1-mOA;
                mXd = X.nVar+1-mXA;
                new_pred_lb = [obj.pred_lb(1:mOd); X.pred_lb(1:mXd); obj.pred_lb(mOd+1:obj.nVar); X.pred_lb(mXd+1:X.nVar)];
                new_pred_ub = [obj.pred_ub(1:mOd); X.pred_ub(1:mXd); obj.pred_ub(mOd+1:obj.nVar); X.pred_ub(mXd+1:X.nVar)];
            else
                new_pred_lb = [];
                new_pred_ub = [];
            end

            S = SparseStar(new_A, new_C, new_d, new_pred_lb, new_pred_ub);       
        end
        
        % get lower bound and upper bound vector of the state variable
        function [lb, ub] = getRange(varargin)
            % author: Sung Woo Choi
            % date: 03/07/2023

            switch nargin
                case 3
                    obj = varargin{1};
                    index = varargin{2}
                    lps = varargin{3};
                case 2
                    obj = varargin{1};
                    index = varargin{2};
                    lps = obj.lps;
                otherwise
                    error('Invalid number of inputs, should be 1, or 2');
            end

            if obj.isEmptySet      
                lb = [];
                ub = [];
                return;
            end

            if isempty(obj.lps)
                lps = 'linprog';
            end

            if strcmp(lps, 'estimate')
                [lb, ub] = obj.estimateRange(index);
            elseif strcmp(lps, 'linprog') || strcmp(lps, 'glpk')
                lb = obj.getMin(index, lps);
                ub = obj.getMax(index, lps);
            else
                error('Unknown LP solver method');
            end

        end


        % get lower bound and upper bound vector of the state variables
        function [lb, ub] = getRanges(varargin)
            % author: Sung Woo Choi
            % date: 03/07/2023

            switch nargin
                case 3
                    obj = varargin{1};
                    lps = varargin{2};
                    par_option = varargin{3};
                case 2
                    obj = varargin{1};
                    lps = varargin{2};
                    par_option = 'single';
                case 1
                    obj = varargin{1};
                    lps = obj.lps;
                    par_option = 'single';
                otherwise
                    error('Invalid number of inputs, should be 1, 2, or 3');
            end

            if obj.isEmptySet      
                lb = [];
                ub = [];
                return;
            end

            if isempty(obj.lps)
                lps = 'linprog';
            end

            if strcmp(lps, 'estimate')
                [lb, ub] = obj.estimateRanges;
            elseif strcmp(lps, 'linprog') || strcmp(lps, 'glpk')
                lb = obj.getMins(1:obj.dim, obj.par, [], lps);
                ub = obj.getMaxs(1:obj.dim, obj.par, [], lps);
            else
                error('Unknown LP solver method');
            end

        end

        % Quickly estimate minimum value of a state x[index]
        function [xmin, xmax] = estimateRange(obj, index)
            % @index: position of the state
            % range: min and max values of x[index]
            
            % author: Sung Woo Choi
            % date: 03/07/2023

            if index < 1 || index > obj.dim
                error('Invalid index');
            end 

            mA = size(obj.A, 2);
            f = obj.A(index, 2:mA);

            p = obj.nVar-mA+1+obj.dim;
            lb = obj.pred_lb(p, 1);
            ub = obj.pred_ub(p, 1);
                        
            pos_mat = max(f, 0);
            neg_mat = min(f, 0);

            xmin = obj.A(index, 1) + pos_mat*lb + neg_mat*ub;
            xmax = obj.A(index, 1) + pos_mat*ub + neg_mat*lb;
        end

        % Quickly estimate lower bound and upper bound of x
        function [xmin, xmax] = estimateRanges(obj)
            % @index: position of the state
            % range: min and max values of x[index]
            
            % author: Sung Woo Choi
            % date: 03/07/2023

            mA = size(obj.A, 2);
            f = obj.A(:, 2:mA);
            
            n = obj.nVar;
            p = n-mA+2;
            lb = obj.pred_lb(p:n, 1);
            ub = obj.pred_ub(p:n, 1);
                        
            pos_mat = max(f, 0);
            neg_mat = min(f, 0);

            xmin = obj.A(:, 1) + pos_mat*lb + neg_mat*ub;
            xmax = obj.A(:, 1) + pos_mat*ub + neg_mat*lb;
        end

        % get min
        function xmin = getMin(varargin)
            % @index: position of the state
            % xmin: min value of x[index]
            
            % author: Sung Woo Choi
            % date: 03/03/2023
            
            switch nargin
                case 2
                    obj = varargin{1};
                    index = varargin{2};
                    lp_solver = 'glpk';
                case 3
                    obj = varargin{1};
                    index = varargin{2};
                    lp_solver = varargin{3};
                otherwise
                    error('Invalid number of input arguments, should be 2 or 3');
            end
            
            if index < 1 || index > obj.dim
                error('Invalid index');
            end 
            
            mA = size(obj.A, 2);
            % check sparcity
            if nnz(obj.A(index, 2:mA)) > 0.5*obj.nVar && strcmp(lp_solver, 'linprog')
                % sparse
                f = [sparse(1, obj.nVar+1-mA), sparse(obj.A(index, 2:mA))];
            else
                % not sparse
                f = [zeros(1, obj.nVar+1-mA), full(obj.A(index, 2:mA))];
            end

            if all(f(:)==0)
                xmin = obj.A(index,1);
            else               
                if strcmp(lp_solver, 'linprog')
                    options = optimoptions(@linprog, 'Display','none');
                    options.OptimalityTolerance = 1e-10; % set tolerance
%                     [~, fval, exitflag, ~] = linprog(f, obj.C, obj.d, [], [], obj.pred_lb, obj.pred_ub, options); 
                    [~, fval, exitflag, ~] = linprog(full(f), full(obj.C), full(obj.d), [], [], obj.pred_lb, obj.pred_ub, options);
                    if exitflag == 1
                        xmin = fval + obj.A(index, 1);
                    else
                        error('Cannot find an optimal solution, exitflag = %d', exitflag);
                    end    
                elseif strcmp(lp_solver, 'glpk')
                    [~, fval, exitflag, ~] = glpk(f, full(obj.C), obj.d, obj.pred_lb, obj.pred_ub);
                    if exitflag == 5
                        xmin = fval + obj.A(index, 1);
                    else
                        error('Cannot find an optimal solution, exitflag = %d', exitflag);
                    end
                    
                else
                    error('Unknown lp solver, should be glpk or linprog'); 
                end

            end
        end

        
        % get mins
        function xmin = getMins(varargin)
            % @map: an array of indexes
            % xmin: min values of x[indexes]
            
            % author: Sung Woo Choi
            % date: 03/03/2023
            
            switch nargin
                case 5
                    obj = varargin{1};
                    map = varargin{2};
                    par_option = varargin{3};
                    dis_option = varargin{4};
                    lp_solver  = varargin{5};
                case 4
                    obj = varargin{1};
                    map = varargin{2};
                    par_option = varargin{3};
                    dis_option = varargin{4};
                    lp_solver = 'glpk';
                case 3
                    obj = varargin{1};
                    map = varargin{2};
                    par_option = varargin{3};
                    dis_option = [];
                    lp_solver = 'glpk';
                case 2
                    obj = varargin{1};
                    map = varargin{2}; 
                    par_option = 'single';
                    dis_option = [];
                    lp_solver = 'glpk';
                otherwise
                    error('Invalid number of inputs, should be 1, 2, 3, or 4');
            end
            
            mA = size(obj.A, 2);
            n = length(map);
            xmin = zeros(n, 1);
            if isempty(par_option) || strcmp(par_option, 'single') % get Maxs using single core
                reverseStr = '';
                for i = 1:n
                    xmin(i) = obj.getMin(map(i), lp_solver);
                    if strcmp(dis_option, 'display')
                        msg = sprintf('%d/%d', i, n);
                        fprintf([reverseStr, msg]);
                        reverseStr = repmat(sprintf('\b'), 1, length(msg));
                    end
                end
            elseif strcmp(par_option, 'parallel') % get Maxs using multiple cores 
                % check sparcity
                if nnz(obj.A) > 0.5*n*obj.nVar && strcmp(lp_solver, 'linprog')
                    % sparse
                    f = [sparse(n, obj.nVar+1-mA), sparse(obj.A(map, 2:mA))];
                else
                    % not sparse
                    f = [zeros(n, obj.nVar+1-mA), full(obj.A(map, 2:mA))];
                end
                options = optimoptions(@linprog,'Display','none');
                options.OptimalityTolerance = 1e-10; % set tolerance
                parfor i=1:n                    
                    if all(f(i,:)==0)
                        xmin(i) = obj.A(map(i), 1);
                    else                                   
                        if strcmp(lp_solver, 'linprog') 
%                             [~, fval, exitflag, ~] = linprog(f(i, :), obj.C, obj.d, [], [], obj.pred_lb, obj.pred_ub, options); 
                            [~, fval, exitflag, ~] = linprog(full(f(i, :)), full(obj.C), full(obj.d), [], [], obj.pred_lb, obj.pred_ub, options);
                            if exitflag == 1
                                xmin(i) = fval + obj.A(map(i), 1);
                            else
                                error('Cannot find an optimal solution, exitflag = %d', exitflag);
                            end                                   
                        elseif strcmp(lp_solver, 'glpk')
                            [~, fval, exitflag, ~] = glpk(f(i, :), full(obj.C), obj.d, obj.pred_lb, obj.pred_ub);
                            if exitflag == 5
                                xmin(i) = fval + obj.A(map(i), 1);
                            else
                                error('Cannot find an optimal solution, exitflag = %d', exitflag);
                            end      
                        else
                            error('Unknown lp solver, should be glpk or linprog');
                        end
                        

                    end
                end
  
            else
                error('Unknown parallel option');
            end
        end
                

        % get max
        function xmax = getMax(varargin)
            % @index: position of the state
            % xmax: max value of x[index]
            
            % author: Sung Woo Choi
            % date: 03/03/2023
            
             switch nargin
                case 2
                    obj = varargin{1};
                    index = varargin{2};
                    lp_solver = 'glpk';
                case 3
                    obj = varargin{1};
                    index = varargin{2};
                    lp_solver = varargin{3};
                otherwise
                    error('Invalid number of input arguments, should be 2 or 3');
            end
            
            if index < 1 || index > obj.dim
                error('Invalid index');
            end
            
            mA = size(obj.A, 2);
            % check sparcity
            if nnz(obj.A(index, 2:mA)) > 0.5*obj.nVar && strcmp(lp_solver, 'linprog')
                % sparse
                f = [sparse(1, obj.nVar+1-mA), sparse(obj.A(index, 2:mA))];
            else
                % not sparse
                f = [zeros(1, obj.nVar+1-mA), full(obj.A(index, 2:mA))];
            end

            if all(f(:)==0)
                xmax = obj.A(index,1);
            else
                if strcmp(lp_solver, 'linprog')
                    options = optimoptions(@linprog, 'Display','none');
                    options.OptimalityTolerance = 1e-10; % set tolerance
%                     [~, fval, exitflag, ~] = linprog(-f, obj.C, obj.d, [], [], obj.pred_lb, obj.pred_ub, options);
                    [~, fval, exitflag, ~] = linprog(-full(f), full(obj.C), obj.d, [], [], obj.pred_lb, obj.pred_ub, options);
                    if exitflag == 1
                        xmax = -fval + obj.A(index, 1);
                    else
                        error('Cannot find an optimal solution, exitflag = %d', exitflag);
                    end    
                elseif strcmp(lp_solver, 'glpk')
                    [~, fval, exitflag, ~] = glpk(-f, full(obj.C), obj.d, obj.pred_lb, obj.pred_ub);
                    if exitflag == 5
                        xmax = -fval + obj.A(index, 1);
                    else
                        error('Cannot find an optimal solution, exitflag = %d', exitflag);
                    end
                    
                else
                    error('Unknown lp solver, should be glpk or linprog'); 
                end   

            end
            
        end
        
        % get maxs
        function xmax = getMaxs(varargin)
            % @map: an array of indexes
            % xmax: max values of x[indexes]
            
            % author: Sung Woo Choi
            % date: 03/03/2023
            
            switch nargin
                case 5
                    obj = varargin{1};
                    map = varargin{2};
                    par_option = varargin{3};
                    dis_option = varargin{4};
                    lp_solver  = varargin{5};
                case 4
                    obj = varargin{1};
                    map = varargin{2};
                    par_option = varargin{3};
                    dis_option = varargin{4};
                    lp_solver = 'glpk';
                case 3
                    obj = varargin{1};
                    map = varargin{2};
                    par_option = varargin{3};
                    dis_option = [];
                    lp_solver = 'glpk';
                case 2
                    obj = varargin{1};
                    map = varargin{2}; 
                    par_option = 'single';
                    dis_option = [];
                    lp_solver = 'glpk';
                otherwise
                    error('Invalid number of inputs, should be 1, 2, 3, or 4');
            end
            
            mA = size(obj.A, 2);
            n = length(map);
            xmax = zeros(n, 1);
            if isempty(par_option) || strcmp(par_option, 'single') % get Maxs using single core
                reverseStr = '';
                for i = 1:n
                    xmax(i) = obj.getMax(map(i), lp_solver);
                    if strcmp(dis_option, 'display')
                        msg = sprintf('%d/%d', i, n);
                        fprintf([reverseStr, msg]);
                        reverseStr = repmat(sprintf('\b'), 1, length(msg));
                    end
                end
            elseif strcmp(par_option, 'parallel') % get Maxs using multiple cores 
                % check sparcity
                if nnz(obj.A) > 0.5*n*obj.nVar && strcmp(lp_solver, 'linprog')
                    % sparse
                    f = [sparse(n, obj.nVar+1-mA), sparse(obj.A(map, 2:mA))];
                else
                    % not sparse
                    f = [zeros(n, obj.nVar+1-mA), full(obj.A(map, 2:mA))];
                end
                options = optimoptions(@linprog, 'Display','none');
                options.OptimalityTolerance = 1e-10; % set tolerance
                parfor i=1:n                    
                    if all(f(i,:)==0)
                        xmax(i) = obj.A(map(i), 1);
                    else
                        
                        if strcmp(lp_solver, 'linprog')
%                             [~, fval, exitflag, ~] = linprog(-f(i, :), obj.C, obj.d, [], [], obj.pred_lb, obj.pred_ub, options); 
                            [~, fval, exitflag, ~] = linprog(-full(f(i, :)), full(obj.C), obj.d, [], [], obj.pred_lb, obj.pred_ub, options); 
                            if exitflag == 1
                                xmax(i) = -fval + obj.A(map(i), 1);
                            else
                                error('Cannot find an optimal solution, exitflag = %d', exitflag);
                            end                                   
                        elseif strcmp(lp_solver, 'glpk')
                            [~, fval, exitflag, ~] = glpk(-f(i, :), full(obj.C), obj.d, obj.pred_lb, obj.pred_ub);
                            if exitflag == 5
                                xmax(i) = -fval + obj.A(map(i), 1);
                            else
                                error('Cannot find an optimal solution, exitflag = %d', exitflag);
                            end      
                        else
                            error('Unknown lp solver, should be glpk or linprog');
                        end     

                    end
                    
                end

            else
                error('Unknown parallel option');
            end
     
        end

        % conver to star
        function S = toStar(obj)
            S = Star(full(obj.V), full(obj.C), obj.d, obj.pred_lb, obj.pred_ub);
        end

        
        % convert to polyhedron
        function P = toPolyhedron(obj)
            
            mA = size(obj.A, 2);
            b = obj.A(:, 1);
            W = [zeros(obj.dim, obj.nVar+1-mA), full(obj.A(:, 2:mA))];
            
            if ~isempty(obj.pred_ub)
                C1 = [eye(obj.nVar); -eye(obj.nVar)];
                d1 = [obj.pred_ub; -obj.pred_lb];
                Pa = Polyhedron('A', [obj.C;C1], 'b', [obj.d;d1]);
                P = W*Pa + b;
            else
                Pa = Polyhedron('A', [obj.C], 'b', [obj.d]);
                P = W*Pa + b;
            end
        end

        function B = getBox(obj)
            if isempty(obj.C) || isempty(obj.d) % star set is just a vector (one point)
                lb = obj.V(:, 1);
                ub = obj.V(:, 1);               
                B = Box(lb, ub);

            else % star set is a set
                
%                 if ~isempty(obj.state_lb) && ~isempty(obj.state_ub)
%                     B = Box(obj.state_lb, obj.state_ub);
                if 0
                else
                    [lb, ub] = obj.getRanges('linprog')

                    if isempty(lb) || isempty(ub)
                        B = [];
                    else
                        B = Box(lb, ub);           
                    end
                end
            end

        end


        % plot sparse star set
        function plot(varargin)
            
            switch nargin
                case 2
                    obj = varargin{1};
                    color = varargin{2};
                    map_mat = [];
                    map_vec = [];
                    approx = [];
                case 1
                    obj = varargin{1};
                    color = 'red';
                    map_mat = [];
                    map_vec = [];
                    approx = [];
                case 3
                    obj = varargin{1};
                    color = varargin{2};
                    map_mat = varargin{3};
                    map_vec = [];
                    approx = [];
                case 4
                    obj = varargin{1};
                    color = varargin{2};
                    map_mat = varargin{3};
                    map_vec = varargin{4};
                    approx = [];
                case 5
                    obj = varargin{1};
                    color = varargin{2};
                    map_mat = varargin{3};
                    map_vec = varargin{4};
                    approx = varargin{5};
                    
                otherwise
                    error('Invalid number of input arguments, should be 1, 2, 3, 4, or 5');
                
            end
            
            
            if isempty(map_mat)
                
                if obj.dim > 3
                    error('Cannot visualize the star set in > 3 dimensional space, please plot a projection of the star set');
                end
                
                if obj.nVar > 20
                                        
                    if isempty(approx)
                        try 
                            P = obj.toPolyhedron;
                            P.plot('color', color);
                        catch
                            warning('The number of predicate variables is high (%d). This can cause an error for MPT plotting');
                            warning('NNV plots an over-approximation of the star set using a box instead');
                            B = obj.getBox;
                            B.plot;
                        end                       
                    else                       
                        
                        if strcmp(approx, 'zonotope')                            
                            if isempty(obj.Z)
                                Z1 = obj.getZono;
                                Zono.plots(Z1);
                            else
                                Zono.plots(obj.Z);
                            end
                        elseif strcmp(approx, 'box')
                            B = obj.getBox;
                            B.plot;
                        else
                            error('Unknown plotting option');
                        end
                        
                    end
                    
                else
                    P = obj.toPolyhedron;
                    P.plot('color', color);
                end
            
            else
                
                [n1,n2] = size(map_mat);
                if n1 > 3 || n1 < 1
                    error('Invalid projection matrix');
                end
                if n2 ~= obj.dim
                    error('Inconsistency between projection matrix and the star set dimension');
                end
                
                if ~isempty(map_vec)
                    [m1,m2] = size(map_vec);
                    if n1~=m1
                        error('Inconsistency between projection matrix and projection vector');
                    end
                    if m2 ~= 1
                        error('Invalid projection vector');
                    end
                end
                
                S1 = obj.affineMap(map_mat, map_vec);
                S1.plot(color,[],[],approx);
            end
            
            
        end

        % generate random sparse star set
        function S = rand(dim)
            % @dim: dimension of the random star set
            % @S: the sparse star set
            
            if dim <= 0 
                error('Invalid dimension');
            end
            P = ExamplePoly.randHrep('d',dim); % random polyhedron
            S = SparseStar(P);  
        end


    end

   


end