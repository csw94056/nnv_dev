classdef SparseStarA
    % SparseStarA set class
    %   SparseStarA is defined by
    %       S = { x | x = A*V*a + c, C*a <= d, 
    %               pred_lb <= a <= pred_ub,
    %               state_lb <= x <= state_ub },
    %       where V  and C are sparse matrixes
    %
    % author: Sung Woo Choi
    % date: 03/03/2023
    
    properties
        A = []; % affine matrix
        V = []; % basis matrix
        C = []; % constraint matrix
        d = []; % constraint vector

        dim = 0;
        nVar = 0;

        pred_lb = []; % lower bound vector of predicate variable
        pred_ub = []; % upper bound vector of predicate variable
        
        state_lb = []; % lower bound of state variables
        state_ub = []; % upper bound of state variables

        lps = 'linprog';
    end

    methods
        
        % constructor
        function obj = SparseStarA(varargin)

            switch nargin

                case 8
                    A = varargin{1};
                    V = varargin{2};
                    C = varargin{3};
                    d = varargin{4};
                    pred_lb = varargin{5};
                    pred_ub = varargin{6};
                    state_lb = varargin{7};
                    state_ub = varargin{8};
                    
                    [nA, mA] = size(A);
                    [nV, mV] = size(V);
                    [nC, mC] = size(C);
                    [nd, md] = size(d);
                    [npl, mpl] = size(pred_lb);
                    [npu, mpu] = size(pred_ub);
                    [nsl, msl] = size(state_lb);
                    [nsu, msu] = size(state_ub);

                    if mA ~= nV + 1
                        error('Inconsistency between affine matrix and basis matrix');
                    end
                    
                    if mV ~= mC
                        error('Inconsistency between basic matrix and constraint matrix');
                    end

                    if nC ~= nd
                        error('Inconsistency between constraint matrix and constraint vector');
                    end

                    if md ~= 1
                        error('constraint vector should have one column');
                    end

                    if mpl ~=1 || mpu ~=1 
                        error('predicate lower- or upper-bounds vector should have one column');
                    end
                    
                    if npl ~= npu || npl ~= mC
                        error('Inconsistency between number of predicate variables and predicate lower- or upper-bounds vector');
                    end
                    
                    if nsl ~= nA || nsu ~= nA
                        error('Inconsistent dimension between lower bound and upper bound vector of state variables and affine matrix');
                    end

                    if msl ~= 1 || msu ~= 1
                        error('Invalid lower bound or upper bound vector of state variables');
                    end


                    obj.A = full(A);
                    obj.V = sparse(V);
                    obj.C = sparse(C);
                    obj.d = d;
                    
                    obj.dim = nA;
                    obj.nVar = mC;
                    obj.pred_lb = pred_lb;
                    obj.pred_ub = pred_ub;
                    obj.state_lb = state_lb;
                    obj.state_ub = state_ub;

                case 6
                    A = varargin{1};
                    V = varargin{2};
                    C = varargin{3};
                    d = varargin{4};
                    pred_lb = varargin{5};
                    pred_ub = varargin{6};
                    
                    [nA, mA] = size(A);
                    [nV, mV] = size(V);
                    [nC, mC] = size(C);
                    [nd, md] = size(d);
                    [npl, mpl] = size(pred_lb);
                    [npu, mpu] = size(pred_ub);

                    if mA ~= nV + 1
                        error('Inconsistency between affine matrix and basis matrix');
                    end
                    
                    if mV ~= mC
                        error('Inconsistency between basic matrix and constraint matrix');
                    end

                    if nC ~= nd
                        error('Inconsistency between constraint matrix and constraint vector');
                    end

                    if md ~= 1
                        error('constraint vector should have one column');
                    end

                    if mpl ~=1 || mpu ~=1 
                        error('predicate lower- or upper-bounds vector should have one column');
                    end
                    
                    if npl ~= npu || npl ~= mC
                        error('Inconsistency between number of predicate variables and predicate lower- or upper-bounds vector');
                    end
                    
                    obj.A = full(A);
                    obj.V = sparse(V);
                    obj.C = sparse(C);
                    obj.d = d;
                    
                    obj.dim = nA;
                    obj.nVar = mC;
                    obj.pred_lb = pred_lb;
                    obj.pred_ub = pred_ub;

                case 2
                    
                    % construct star from lower bound and upper bound
                    % vector
                    lb = varargin{1};
                    ub = varargin{2};
                    
                    B = Box(lb,ub);
                    S = B.toStar;
                    obj.A = S.V;
                    obj.V = sparse(eye(S.dim));
                    obj.C = sparse(zeros(1, S.nVar)); % initiate an obvious constraint
                    obj.d = zeros(1, 1);
                    obj.dim = S.dim;
                    obj.nVar = S.nVar;
                    obj.state_lb = lb;
                    obj.state_ub = ub;
                    obj.pred_lb = -ones(S.nVar, 1);
                    obj.pred_ub = ones(S.nVar, 1);

                case 0
                    % create empty Star (for preallocation an array of star)
                    obj.A = [];
                    obj.V = [];
                    obj.C = [];
                    obj.d = [];
                    obj.dim = 0;
                    obj.nVar = 0;

                otherwise
                    error('Invalid number of input arguments (should be 0 or 2 or 3 or 5)');
            end


        end
        
        % check is empty set
        function bool = isEmptySet(obj)
            % author: Sung Woo Choi
            % date: 03/03/2023


            f = zeros(1, obj.nVar);
            if strcmp(obj.lps, 'linprog')
                options = optimoptions(@linprog, 'Display','none'); 
                options.OptimalityTolerance = 1e-10; % set tolerance
                [~, ~, exitflag, ~] = linprog(f, obj.C, obj.d, [], [], obj.pred_lb, obj.pred_ub, options);
                if exitflag == 1
                    bool = 0;
                elseif exitflag == -2 || exitflag == -9 || exitflag == -5 || exitflag == 0
                    bool = 1;
                else
                    error('Error, exitflag = %d', exitflag);
                end
            elseif strcmp(obj.lps, 'glpk')
                [~, ~, exitflag, ~] = glpk(f, obj.C, obj.d, obj.pred_lb, obj.pred_ub);
                if exitflag == 5 || exitflag == 2
                    bool = 0;
                else
                    error('Cannot find an optimal solution, exitflag = %d', exitflag);
                end
                
            else
                error('Unknown lp solver, should be glpk or linprog'); 
            end
        end

        % sampling a star set
        function V = sample(obj, N)
            % @N: number of points in the samples
            % @V: a set of at most N sampled points in the star set 
            
            % author: Sung Woo Choi
            % date: 03/03/2023
            
            if N < 1
                error('Invalid number of samples');
            end
            
            n = 1:obj.dim;
            lb = obj.getMins(n, 'single', [], obj.lps);
            ub = obj.getMaxs(n, 'single', [], obj.lps);
            
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

        % affine mapping of star set S = Wx + b;
        function S = affineMap(obj, W, b)
            % @W: mapping matrix
            % @b: mapping vector
            % @S: new star set
            
            if size(W, 2) ~= obj.dim
                error('Inconsistency between the affine mapping matrix and dimension of the star set');
            end
            
            if ~isempty(b)
                if size(b, 1) ~= size(W, 1)
                    error('Inconsistency between the mapping vec and mapping matrix');
                end

                if size(b, 2) ~= 1
                    error('Mapping vector should have one column');
                end

                new_A(:, 2:obj.nVar+1) = W * obj.A(:, 2:obj.nVar+1);
                new_A(:, 1) = new_A(:, 1) + b;
            else
                new_A(:, 2:obj.nVar+1) = W * obj.A;
            end

            
            S = SparseStarA(new_A, obj.V, obj.C, obj.d, obj.pred_lb, obj.pred_ub);
                       
        end

        
        % New Minkowski Sum (used for Recurrent Layer reachability)
        function S = Sum(obj, X)
            % @X: another star with same dimension
            % @S: new star
            
            if ~isa(X, 'SparseStarA')
                error('Input matrix is not a SparseStarA');
            else
                if X.dim ~= obj.dim
                    error('Input SparseStarA and current SparseStarA have different dimensions');
                end
            end

            A1 = eye(obj.dim);
            
            m1 = size(obj.A, 2);
            m2 = size(X.A, 2);
            
            A1 = obj.A(:, 2:m1);
            A2 = X.A(:, 2:m2);

            Z1 =
            Z2 =

            new_c = obj.A(:, 1) + X.A(:, 1);
            new_A = horzcat(new_c, A1);        
            
            new_V = horzcat(V1, V2);            
            new_C = blkdiag(obj.C, X.C);        
            new_d = vertcat(obj.d, X.d);
            
            if ~isempty(obj.pred_lb) && ~isempty(X.pred_lb)
                new_pred_lb = [obj.pred_lb; X.pred_lb];
                new_pred_ub = [obj.pred_ub; X.pred_ub];
                S = SparseStarA(new_A, new_V, new_C, new_d, new_pred_lb, new_pred_ub);
            else
                S = SparseStarA(new_A, new_V, new_C, new_d); % new Star has more number of basic vectors
            end
       
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
            
            f = obj.A(index, 2:obj.nVar + 1) * obj.V;
            if all(f(:)==0)
                xmin = obj.A(index,1);
            else               
                if strcmp(lp_solver, 'linprog')
                    options = optimoptions(@linprog, 'Display','none');
                    options.OptimalityTolerance = 1e-10; % set tolerance
                    [~, fval, exitflag, ~] = linprog(f, obj.C, obj.d, [], [], obj.pred_lb, obj.pred_ub, options); 
                    if exitflag == 1
                        xmin = fval + obj.A(index, 1);
                    else
                        error('Cannot find an optimal solution, exitflag = %d', exitflag);
                    end    
                elseif strcmp(lp_solver, 'glpk')
                    [~, fval, exitflag, ~] = glpk(f, obj.C, obj.d, obj.pred_lb, obj.pred_ub);
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
                f = obj.A(map, 2:obj.nVar + 1) * obj.V;
                options = optimoptions(@linprog,'Display','none');
                options.OptimalityTolerance = 1e-10; % set tolerance
                parfor i=1:n                    
                    if all(f(i,:)==0)
                        xmin(i) = V1(i,1);
                    else
                                   
                        if strcmp(lp_solver, 'linprog')
                            [~, fval, exitflag, ~] = linprog(f(i, :), obj.C, obj.d, [], [], obj.pred_lb, obj.pred_ub, options); 
                            if exitflag == 1
                                xmin(i) = fval + obj.A(map, 1);
                            else
                                error('Cannot find an optimal solution, exitflag = %d', exitflag);
                            end                                   
                        elseif strcmp(lp_solver, 'glpk')
                            [~, fval, exitflag, ~] = glpk(f(i, :), obj.C, obj.d, obj.pred_lb, obj.pred_ub);
                            if exitflag == 5
                                xmin(i) = fval + obj.A(map, 1);
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
            
            
            f = obj.A(index, 2:obj.nVar + 1) * obj.V;
            if all(f(:)==0)
                xmax = obj.A(index,1);
            else
                if strcmp(lp_solver, 'linprog')
                    options = optimoptions(@linprog, 'Display','none');
                    options.OptimalityTolerance = 1e-10; % set tolerance
                    [~, fval, exitflag, ~] = linprog(-f, obj.C, obj.d, [], [], obj.pred_lb, obj.pred_ub, options); 
                    if exitflag == 1
                        xmax = -fval + obj.A(index, 1);
                    else
                        error('Cannot find an optimal solution, exitflag = %d', exitflag);
                    end    
                elseif strcmp(lp_solver, 'glpk')
                    [~, fval, exitflag, ~] = glpk(-f, obj.C, obj.d, obj.pred_lb, obj.pred_ub);
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
                f = obj.A(map, 2:obj.nVar + 1) * obj.V;
                options = optimoptions(@linprog, 'Display','none');
                options.OptimalityTolerance = 1e-10; % set tolerance
                parfor i=1:n                    
                    if all(f(i,:)==0)
                        xmax(i) = V1(i,1);
                    else
                        
                        if strcmp(lp_solver, 'linprog')
                            [~, fval, exitflag, ~] = linprog(-f(i, :), obj.C, obj.d, [], [], obj.pred_lb, obj.pred_ub, options); 
                            if exitflag == 1
                                xmax(i) = -fval + V1(i, 1);
                            else
                                error('Cannot find an optimal solution, exitflag = %d', exitflag);
                            end                                   
                        elseif strcmp(lp_solver, 'glpk')
                            [~, fval, exitflag, ~] = glpk(-f(i, :), obj.C, obj.d, obj.pred_lb, obj.pred_ub);
                            if exitflag == 5
                                xmax(i) = -fval + V1(i, 1);
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

            
        % convert to polyhedron
        function P = toPolyhedron(obj)
            
            b = obj.A(:, 1);        
            W = obj.A(:, 2:obj.nVar + 1) * obj.V;
            
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
        

        % plot star set
        function plot(varargin)
            % author: Dung Tran
            % date: 3/27/2019
            % update: 4/2/2020
            
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
        
        
    end



    methods(Static) % plot methods
        
        % plot an array of Star (plot exactly, this is time consuming)
        function plots(varargin)
            % @S: an array of Stars
            % @colar: color
            
            % author: Dung Tran
            % date: update in 10/2/2019
            
            switch nargin
                
                case 2
                    
                    S = varargin{1};
                    color = varargin{2};
                    
                case 1
                    S = varargin{1};
                    color = 'b';
                    
                otherwise
                    error('Invalid number of inputs, should be 1 or 2');
            end
            
            n = length(S);
            if n==1
                S(1).plot(color);
            else
                for i=1:n-1
                    S(i).plot(color);
                    hold on;
                end
                S(n).plot(color);
            end
            
            
        end


    end

end