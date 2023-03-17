classdef GruRNN < handle
    % GruRNN Class is a class for handling reachability analysis of the GRU
    % recurrent nueral networks
    % reachability analysis methods: 'approx-star'
    % Author: Sung Woo Choi
    % Date: 02/23/2023

    properties
        Name = 'net';
        Layers = []; % An array of Layers, eg, Layers = [L1 L2 ...Ln]
        nL = 0; % number of Layers
        nM = 0; % number of memory units
        nN = 0; % number of neurons (not including memory units);
        nI = 0; % number of Inputs
        
        % properties for reach set computation
        
        reachMethod = 'approx-star';    % reachable set computation scheme, default - 'star'
        reachOption = []; % parallel option, default - non-parallel computing
        relaxFactor = 0; % use only for approximate star method, 0 mean no relaxation
        numCores = 1; % number of cores (workers) using in computation
        inputSet = [];  % input set
        reachSet = [];  % reachable set for each layers
        outputSet = []; % output reach set
        reachTime = []; % computation time for each layers
        numReachSet = []; % number of reach sets over layers
        totalReachTime = 0; % total computation time       
        numSamples = 0; % default number of samples using to falsify a property
        unsafeRegion = []; % unsafe region of the network
        getCounterExs = 0; % default, not getting counterexamples
        
        Operations = []; % flatten a network into a sequence of operations
        
        dis_opt = []; % display option
        lp_solver = 'glpk'; % lp solver option, should be glpk or linprog

    end


    methods 

        % constructor
        function obj = GruRNN(varargin)
            
            switch nargin
                case 1
                    Layers = varargin{1};
                    name = 'net';
                case 2
                    Layers = varargin{2};
                    name = varargin{2};
                otherwise
                    error('Invalid number of inputs');
            end

            nL = size(Layers, 2); % number of Layer
            for i = 1:nL
                L = Layers{i};
                if ~isa(L, 'GRULayer') && ~isa(L, 'LayerS')
                    error('Element %d of Layers array is not a GRULayer object or a LayerS object', i);
                end
            end

            nMemory = 0; % number of memory units
            nNeurons = 0; % number of neurons not including memory units
            % check consistency between layers
%             if nL > 1
%                 for i = 1:nL-1
%                     if isa(Layers{i}, 'GRULayer')
%                         if i==1
%                             nInputs = Layers{i}.nI;
%                         end
%                         nMemory = nMemory + Layers{i}.nH;
%                     elseif isa(Layers{i}, 'LayerS')
%                         if i==1
%                             nInputs = size(Layers{i}.W, 1);
%                         end
%                         nMemory = nMemory + Layers{i}.N;
%                     end
%                 end
%             end

            %{
                add check for consistency
            %}


            obj.Layers = Layers;
            obj.nL = nL;    % number of layers
            obj.nI = nInputs; % number of inputs
            obj.nO = nOutputs; % number of outputs
            obj.nH = nHidden; 
            obj.nN = nNeurons;
            obj.Name = name;

        end

        % Evaluation of GruRNN
        function y = evaluate(obj, x)
            % Evaluation of this GruRNN
            % @x: input vector x
            % @y: output vector y

            y = x;
            for i = 1:obj.nL
                y = obj.Layers{i}.evaluate(y);
            end
        end

        % print information to a file
        function print(obj, file_name)

        end

        % start parallel pool for computing
        function start_pool(varargin)

            switch nargin
                case 1
                    obj = varargin{1};
                    nCores = obj.numCores;
                case 2
                    nCores = varargin{2};
                otherwise
                    error('Invalid number of input arguments');
            
            end
            
            if nCores > 1
                poolobj = gcp('nocreate'); % If no pool, do not create new one.
                if isempty(poolobj)
                    parpool('local', nCores); 
                else
                    if poolobj.NumWorkers ~= nCores
                        delete(poolobj); % delete the old poolobj
                        parpool('local', nCores); % start the new one with new number of cores
                    end                    
                end
            end

        end

    end

    methods % reachability analysis method

        function [S,t] = reach(varargin)
            % @I: a sequence of input sets, an array of star sets          
            % @method: = 'exact-star' or 'approx-star' -> compute reach set using stars
            %            'abs-dom' -> compute reach set using abstract
            %            'approx-zono' -> compute reach set using zonotope
            %            domain (support in the future)
            %            'face-latice' -> compute reach set using
            %            face-latice (support in the future)
            
            % @R: output set
            % @t: reachtime
            
            % @numOfCores: number of cores you want to run the reachable
            % set computation, @numOfCores >= 1, maximum is the number of
            % cores in your computer. We also can run this method on a
            % clusters. You need to change: parpool('local', numOfCores) to
            % parcluster(your_cluster_profile), you also need to set up
            % your local clusters with installed MPT toolbox also. see: 
            % https://www.mathworks.com/help/distcomp/discover-clusters-and-use-cluster-profiles.html

            % author: Sung Woo Choi
            % date: 03/06/2023

            % parse inputs 
            switch nargin
                
                 case 7
                    obj = varargin{1}; % GruRNN object
                    obj.inputSet = varargin{2}; % input set
                    obj.reachMethod = varargin{3}; % reachability analysis method
                    obj.numCores = varargin{4}; % number of cores used in computation
                    obj.relaxFactor = varargin{5}; 
                    obj.dis_opt = varargin{6};
                    obj.lp_solver = varargin{7};
          
                case 6
                    obj = varargin{1}; % GruRNN object
                    obj.inputSet = varargin{2}; % input set
                    obj.reachMethod = varargin{3}; % reachability analysis method
                    obj.numCores = varargin{4}; % number of cores used in computation
                    obj.relaxFactor = varargin{5}; 
                    obj.dis_opt = varargin{6}; 
                
                case 5
                    obj = varargin{1}; % GruRNN object
                    obj.inputSet = varargin{2}; % input set
                    obj.reachMethod = varargin{3}; % reachability analysis method
                    obj.numCores = varargin{4}; % number of cores used in computation
                    obj.relaxFactor = varargin{5}; % used only for approx-star method
                case 4
                    obj = varargin{1}; % GruRNN object
                    obj.inputSet = varargin{2}; % input set
                    obj.reachMethod = varargin{3}; % reachability analysis method
                    obj.numCores = varargin{4}; % number of cores used in computation
                case 3
                    obj = varargin{1};
                    obj.inputSet = varargin{2};
                    obj.reachMethod = varargin{3};
                    obj.numCores = 1;
                    
                case 2
                    obj = varargin{1};
                    if ~isstruct(varargin{2})
                        obj.inputSet = varargin{2};
                        obj.reachMethod = 'approx-star';
                        obj.numCores = 1;
                    else
                        if isfield(varargin{2}, 'inputSet')
                            obj.inputSet = varargin{2}.inputSet;
                        end
                        if isfield(varargin{2}, 'numCores')
                            obj.numCores = varargin{2}.numCores;
                        end
                        if isfield(varargin{2}, 'reachMethod')
                            obj.reachMethod = varargin{2}.reachMethod;
                        end
                        if isfield(varargin{2}, 'dis_opt')
                            obj.dis_opt = varargin{2}.dis_opt;
                        end
                        if isfield(varargin{2}, 'relaxFactor')
                            obj.relaxFactor = varargin{2}.relaxFactor;
                        end
                        if isfield(varargin{2}, 'lp_solver')
                            obj.lp_solver = varargin{2}.lp_solver;
                        end
                        
                    end   
                
                otherwise
                    error('Invalid number of input arguments (should be 1, 2, 3, 4, 5, or 6)');
            end

            % if reachability analysis method is an over-approximate
            % method, we use 1 core for computation
            if ~strcmp(obj.reachMethod, 'approx-star') && ~contains(obj.reachMethod, 'relax-star')
                error('GruRNN supports approx-star and relax-star methods');
            end

            
            if obj.numCores == 1
                obj.reachOption = []; % don't use parallel computing
            else
                obj.start_pool;       % start parallel pool in Matlab
                obj.reachOption = 'parallel';
            end
            
            obj.reachSet = cell(1, obj.nL);
            obj.numReachSet = zeros(1, obj.nL);
            obj.reachTime = [];

            % compute reachable set
            In = obj.inputSet;

            for i=1:obj.nL
                
                if strcmp(obj.dis_opt, 'display')
                    fprintf('\nComputing reach set for Layer %d ...', i);
                end
                
                st = tic;
                In = obj.Layers{i}.reach(In, obj.reachMethod, obj.reachOption, obj.relaxFactor, obj.dis_opt, obj.lp_solver);
                t1 = toc(st);
                
                obj.reachSet{1, i} = In;
                obj.numReachSet(i) = length(In);
                obj.reachTime = [obj.reachTime t1];
                
                if strcmp(obj.dis_opt, 'display')
                    fprintf('\nExact computation time: %.5f seconds', t1);               
                    fprintf('\nNumber of reachable set at the output of layer %d: %d', i, length(In));               
                end           
            end
            
            obj.outputSet = In;
            obj.totalReachTime = sum(obj.reachTime);
            S = obj.outputSet;
            t = obj.totalReachTime;
            if strcmp(obj.dis_opt, 'display')
                fprintf('\nTotal reach set computation time: %.5f seconds', obj.totalReachTime);
                fprintf('\nTotal number of output reach sets: %d', length(obj.outputSet));
            end

        end

    end

    methods % verification method, verify robustness
        
        % verify robustness of classification feedforward networks
        function [rb, vt] = verifyRBN(varargin)
            % @rb    : = 1: the network is robust
            %          = 0: the network is notrobust
            %          = 2: robustness is uncertain
            % @vt: verification time
            
            % author: Sung Woo Choi
            % date: 03/06/2023
            
            t = tic;
            switch nargin                   
                
                case 3
                    obj = varargin{1};
                    x = varargin{2}; % a sequence of input points
                    eps = varargin{3}; % adversarial bound
                    verify_option = 'normal'; 
                    obj.numCores = 1;
                    obj.relaxFactor = 0;
                    obj.reachMethod = 'approx-star';
                    
                case 4
                    obj = varargin{1};
                    x = varargin{2};
                    eps = varargin{3};
                    verify_option = varargin{4};
                    obj.numCores = 1;
                    obj.relaxFactor = 0;
                    obj.reachMethod = 'approx-star';
                                        
                case 5
                    obj = varargin{1};
                    x = varargin{2};
                    eps = varargin{3};
                    verify_option = varargin{4};
                    obj.numCores = varargin{5};
                    obj.relaxFactor = 0;
                    obj.reachMethod = 'approx-star';
                    
                case 7
                    obj = varargin{1};
                    x = varargin{2};
                    eps = varargin{3};
                    verify_option = varargin{4};
                    obj.numCores = varargin{5};
                    obj.relaxFactor = varargin{6};
                    obj.reachMethod = varargin{7};
                
                    
                otherwise
                    error('Invalid number of inputs, should be 3, 4, 5 or 7');
                     
            end
            
            n = size(x, 2);
            X = [];
            for i=1:n
                X = [X Star(x(:,i) - eps, x(:, i) + eps)]; % construct sequence of input sets
            end
                        
            % compute reach sets
            [Y, ~] = obj.reach(X, obj.reachMethod, obj.numCores, obj.relaxFactor);
                       
            % verify reach sets compare with groundtruth, i.e.,
            % non-attacked signal
            y = obj.evaluate(x);
                                    
            [~,max_id] = max(y); % find the classified output
            rb = zeros(1, n);
            for i=1:n
                max_cands = Y(i).get_max_point_candidates;
                if length(max_cands) == 1
                    if max_cands == max_id(i)
                        rb(i) = 1;
                    end
                else
                    
                    a = (max_cands == max_id(i));
                    if sum(a) == 0
                        rb(i) = 0;
                    else
                        if strcmp(verify_option, 'fast') % in fast_mode, we don't use LP to verify, use estimated ranges only
                            rb(i) = 2; 
                        elseif strcmp(verify_option, 'normal')
                            max_cands(max_cands == max_id(i)) = [];
                            m = length(max_cands);
                            for j=1:m
                                if Y(i).is_p1_larger_than_p2(max_cands(j), max_id(i))
                                    rb(i) = 2; % find a counter example?
                                    break;
                                else
                                    rb(i) = 1;
                                end
                            end
                        else
                            error('Unknown verify option, should be fast or normal');                        
                        end
                        
                    end
                    
                end
            end          
            
            vt = toc(t);
           
        end
        
    end

end