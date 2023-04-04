classdef LogSig
    % LOGSIG Class contains methods for reachability analysis of layer with
    % Sigmoid activation function.
    % Reference: https://www.mathworks.com/help/deeplearning/ref/logsig.html
    % Author: Dung Tran
    % Date: 28/2/2019
    
    properties
        
    end
    
    methods(Static)  % evaluate method and over-approximate reachability analysis with stars
        
        % evaluation
        function y = evaluate(x)
            y = logsig(x);
        end
        
        
        % main method
        function S = reach_star_approx(varargin)
            % author: Dung Tran
            % date: 3/19/2020
            
            switch nargin
                case 1
                    I = varargin{1};
                    method = 'approx-star-no-split';
                    reachOption = [];
                    relaxFactor = 0;
                    dis_opt = [];
                    lp_solver = 'glpk';
                case 2
                    I = varargin{1};
                    method = varargin{2};
                    reachOption = [];
                    relaxFactor = 0;
                    dis_opt = [];
                    lp_solver = 'glpk';
                case 3
                    I = varargin{1};
                    method = varargin{2};
                    reachOption = varargin{3};
                    relaxFactor = 0;
                    dis_opt = [];
                    lp_solver = 'glpk';
                case 4
                    I = varargin{1};
                    method = varargin{2};
                    reachOption = varargin{3};
                    relaxFactor = varargin{4};
                    dis_opt = [];
                    lp_solver = 'glpk';
                case 5
                    I = varargin{1};
                    method = varargin{2};
                    reachOption = varargin{3};
                    relaxFactor = varargin{4};
                    dis_opt = varargin{5};
                    lp_solver = 'glpk';
                case 6
                    I = varargin{1};
                    method = varargin{2};
                    reachOption = varargin{3};
                    relaxFactor = varargin{4};
                    dis_opt = varargin{5};
                    lp_solver = varargin{6};
                otherwise
                    error('Invalid number of input arguments, should be 1, 2, 3, 4, 5 or 6');
            end
            
            if ~isa(I, 'Star')
                error('Input set is not a star set');
            end
           
            if strcmp(method, 'approx-star-no-split') || strcmp(method, 'approx-star')
                if relaxFactor == 0
                    S = LogSig.reach_star_approx_no_split(I, dis_opt, lp_solver);
                else
                    S = LogSig.relaxedMultiStepLogSig_NoSplit(I, relaxFactor, dis_opt, lp_solver);
                end
            elseif strcmp(method, 'approx-star-split')
                S = LogSig.reach_star_approx_split(I);
            else
                error('Unknown reachability method');
            end
            
        end
        
        
        % reachability method with star
        function S = reach_star_approx_no_split(varargin)
            % @I: the input star set
            % @S: a star set output
            
            % author: Dung Tran
            % date: 3/19/2020
            % update:4/2/2020, 6/12/2020
            % update: 7/15/2020: add display option
            %         7/16/2020: add lp_solver option
            
            switch nargin
                case 1
                    I = varargin{1};
                    dis_opt = [];
                    lp_solver = 'glpk';
                case 2
                    I = varargin{1};
                    dis_opt = varargin{2};
                    lp_solver = 'glpk';
                case 3
                    I = varargin{1};
                    dis_opt = varargin{2};
                    lp_solver = varargin{3};
                otherwise
                    error('Invalid number of input arguments, should be 1 or 2');
            end
            S = LogSig.multiStepLogSig_NoSplit(I, dis_opt, lp_solver);      % more efficient method compared with stepLogSig_NoSplit       
%             n = I.dim;
%             S = I;
%             for i=1:n
%                 S = LogSig.stepLogSig_NoSplit(S, i); 
%             end
                        
        end
        
        % reachability method with star
        function S = reach_star_approx_split(I)
            % @I: the input star set
            % @S: an array of star set output
            
            % author: Dung Tran
            % date: 3/19/2020
            % update:4/2/2020
           
            n = I.dim;
            S = I;
            for i=1:n
                m = length(S);
                O = [];
                for j=1:m
                    O = [O LogSig.stepLogSig_Split(S(j), i)];
                end
                S = O;
            end
       
        end
        
        % stepLogSig
        function S = stepLogSig_Split(I, index)
            % @I: input star set
            % @index: index of the neuron
            
            % @l: l = min(x[index]), lower bound at neuron x[index] 
            % @u: u = min(x[index]), upper bound at neuron x[index]
            % @y_l: = logsig(l); output of logsig at lower bound
            % @y_u: = logsig(u); output of logsig at upper bound
            % @dy_l: derivative of LogSig at the lower bound
            % @dy_u: derivative of LogSig at the upper bound
            
            % @S: output star set
            
            % author: Dung Tran
            % date: 3/19/2020
            % update: 4/2/2020
            
            %[l, u] = I.Z.getRange(index);
            [l, u] = I.getRange(index);
            y_l = logsig(l);
            y_u = logsig(u);
            dy_l = logsig('dn', l);
            dy_u = logsig('dn', u);
            
            if l == u
               
                new_V = I.V;
                new_V(index, 1:I.nVar+1) = 0;
                new_V(index, 1) = y_l;
                if ~isempty(I.Z)
                    c = I.Z.c;
                    V = I.Z.V;
                    c(index) = y_l;
                    V(index, :) = 0;
                    new_Z = Zono(c, V);
                else
                    new_Z = [];
                end
                S = Star(new_V, I.C, I.d, I.predicate_lb, I.predicate_ub, new_Z); 
               
            elseif l > 0
                % y is convex when x >= 0
                % constraint 1: y <= y'(l) * (x - l) + y(l)
                % constarint 2: y <= y'(u) * (x - u) + y(u) 
                % constraint 3: y >= (y(u) - y(l)) * (x - l) / (u - l) + y(l);


                n = I.nVar + 1;
                % over-approximation constraints 
                % constraint 1: y <= y'(l) * (x - l) + y(l)
                C1 = [-dy_l*I.V(index, 2:n) 1];
                d1 = dy_l * I.V(index, 1) - dy_l*l + y_l; 
                % constraint 2: y <= y'(u) * (x - u) + y(u)
                C2 = [-dy_u*I.V(index, 2:n) 1];
                d2 = dy_u * I.V(index, 1) - dy_u*u + y_u;
                % constraint 3: y >= (y(u) - y(l)) * (x - l) / (u - l) + y(l);
                a = (y_u - y_l)/(u - l);
                C3 = [a*I.V(index, 2:n) -1];
                d3 = a*l - y_l - a*I.V(index, 1);

                m = size(I.C, 1);
                C0 = [I.C zeros(m, 1)];
                d0 = I.d;
                new_C = [C0; C1; C2; C3];
                new_d = [d0; d1; d2; d3];
                new_V = [I.V zeros(I.dim, 1)];
                new_V(index, :) = zeros(1, n+1);
                new_V(index, n+1) = 1; 

                % update predicate bound
                new_predicate_lb = [I.predicate_lb; y_l]; 
                new_predicate_ub = [I.predicate_ub; y_u];

                % update outer-zonotope
                if ~isempty(I.Z)
                    c = I.Z.c;
                    V = I.Z.V;                
                    lamda = min(dy_l, dy_u);
                    mu1 = 0.5*(y_u + y_l - lamda *(u + l));
                    mu2 = 0.5*(y_u - y_l - lamda *(u - l));
                    c(index) = lamda * c(index) + mu1;
                    V(index, :) = lamda * V(index, :); 
                    I1 = zeros(I.dim, 1);
                    I1(index) = mu2;
                    V = [V I1];
                    new_Z = Zono(c, V);
                else
                    new_Z = [];
                end

                S = Star(new_V, new_C, new_d, new_predicate_lb, new_predicate_ub, new_Z);

            elseif u <= 0
                % y is concave when x <= 0
                % constraint 1: y >= y'(l) * (x - l) + y(l)
                % constraint 2: y >= y'(u) * (x - u) + y(u)
                % constraint 3: y <= (y(u) - y(l)) * (x -l) / (u - l) + y(l);

                n = I.nVar + 1;
                % over-approximation constraints 
                % constraint 1: y >= y'(l) * (x - l) + y(l)
                C1 = [dy_l*I.V(index, 2:n) -1];
                d1 = -dy_l * I.V(index, 1) + dy_l*l - y_l; 
                % constraint 2: y >= y'(u) * (x - u) + y(u)
                C2 = [dy_u*I.V(index, 2:n) -1];
                d2 = -dy_u * I.V(index, 1) + dy_u*u - y_u;
                % constraint 3: y <= (y(u) - y(l)) * (x - l) / (u - l) + y(l);
                a = (y_u - y_l)/(u - l);
                C3 = [-a*I.V(index, 2:n) 1];
                d3 = -a*l + y_l + a*I.V(index, 1);

                m = size(I.C, 1);
                C0 = [I.C zeros(m, 1)];
                d0 = I.d;
                new_C = [C0; C1; C2; C3];
                new_d = [d0; d1; d2; d3];
                new_V = [I.V zeros(I.dim, 1)];
                new_V(index, :) = zeros(1, n+1);
                new_V(index, n+1) = 1; 

                % update predicate bound
                new_predicate_lb = [I.predicate_lb; y_l]; 
                new_predicate_ub = [I.predicate_ub; y_u];
                
                % update outer-zonotope
                if ~isempty(I.Z)
                    c = I.Z.c;
                    V = I.Z.V;                
                    lamda = min(dy_l, dy_u);
                    mu1 = 0.5*(y_u + y_l - lamda *(u + l));
                    mu2 = 0.5*(y_u - y_l - lamda *(u - l));
                    c(index) = lamda * c(index) + mu1;
                    V(index, :) = lamda * V(index, :); 
                    I1 = zeros(I.dim, 1);
                    I1(index) = mu2;
                    V = [V I1];
                    new_Z = Zono(c, V);
                else
                    new_Z = [];
                end

                S = Star(new_V, new_C, new_d, new_predicate_lb, new_predicate_ub, new_Z);

            elseif l <= 0 && u >0
                % y is concave for x in [l, 0] and convex for x
                % in [0, u]
                % split can be done here 

                % case 1: x in [l, 0]
                % y'(0) = 0.25
                % y is concave when x <= 0
                % constraint 1: y >= y'(l) * (x - l) + y(l)
                % constraint 2: y >= y'(0) * (x) + y(0)
                % constraint 3: y <= (y(0) - y(l)) * (x -l) / (0 - l) + y(l);

                n = I.nVar + 1;
                % over-approximation constraints 
                % constraint 1: y >= y'(l) * (x - l) + y(l)
                C1 = [dy_l*I.V(index, 2:n) -1];
                d1 = -dy_l * I.V(index, 1) + dy_l*l - y_l; 
                % constraint 2: y >= y'(0) * (x - 0) + y(0) = 0.25*x + 0.5
                C2 = [0.25*I.V(index, 2:n) -1];
                d2 = -0.25 * I.V(index, 1) - 0.5;
                % constraint 3: y <= (y(0) - y(l)) * (x - l) / (0 - l) + y(l);
                a = (0.5 - y_l)/(0 - l);
                C3 = [-a*I.V(index, 2:n) 1];
                d3 = -a*l + y_l + a*I.V(index, 1);

                m = size(I.C, 1);
                C0 = [I.C zeros(m, 1)];
                d0 = I.d;
                new_C = [C0; C1; C2; C3];
                new_d = [d0; d1; d2; d3];
                new_V = [I.V zeros(I.dim, 1)];
                new_V(index, :) = zeros(1, n+1);
                new_V(index, n+1) = 1; 

                % update predicate bound
                new_predicate_lb = [I.predicate_lb; y_l]; 
                new_predicate_ub = [I.predicate_ub; 0.5];
                
                % update outer-zonotope
                if ~isempty(I.Z)
                    c = I.Z.c;
                    V = I.Z.V;                
                    lamda = min(dy_l, 0.25);
                    mu1 = 0.5*(0.5 + y_l - lamda *(0 + l));
                    mu2 = 0.5*(0.5 - y_l - lamda *(0 - l));
                    c(index) = lamda * c(index) + mu1;
                    V(index, :) = lamda * V(index, :); 
                    I1 = zeros(I.dim, 1);
                    I1(index) = mu2;
                    V = [V I1];
                    new_Z = Zono(c, V);
                else
                    new_Z = [];
                end
                
                S1 = Star(new_V, new_C, new_d, new_predicate_lb, new_predicate_ub, new_Z);

                % case 2: x in [0, u] 
                % y is convex when x >= 0
                % constraint 1: y <= y'(0) * (x - 0) + y(0) = 0.25*x + 0.5
                % constarint 2: y <= y'(u) * (x - u) + y(u) 
                % constraint 3: y >= (y(u) - y(0)) * (x - 0) / (u - 0) + y(0);

                % over-approximation constraints 
                % constraint 1: y <= y'(0) * (x - 0) + y(0) = 0.25*x + 0.5
                C1 = [-0.25*I.V(index, 2:n) 1];
                d1 = 0.25 * I.V(index, 1) + 0.5; 
                % constraint 2: y <= y'(u) * (x - u) + y(u)
                C2 = [-dy_u*I.V(index, 2:n) 1];
                d2 = dy_u * I.V(index, 1) - dy_u*u + y_u;
                % constraint 3: y >= (y(u) - y(0)) * (x - 0) / (u - 0) + y(0);
                a = (y_u - 0.5)/u;
                C3 = [a*I.V(index, 2:n) -1];
                d3 = -0.5 - a*I.V(index, 1);

                new_C = [C0; C1; C2; C3];
                new_d = [d0; d1; d2; d3];
                new_V = [I.V zeros(I.dim, 1)];
                new_V(index, :) = zeros(1, n+1);
                new_V(index, n+1) = 1; 

                % update predicate bound
                new_predicate_lb = [I.predicate_lb; 0.5]; 
                new_predicate_ub = [I.predicate_ub; y_u];
                
                % update outer-zonotope
                if ~isempty(I.Z)
                    c = I.Z.c;
                    V = I.Z.V;                
                    lamda = min(dy_u, 0.25);
                    mu1 = 0.5*(y_u + 0.5 - lamda *(u + 0));
                    mu2 = 0.5*(y_u - 0.5 - lamda *(u - 0));
                    c(index) = lamda * c(index) + mu1;
                    V(index, :) = lamda * V(index, :); 
                    I1 = zeros(I.dim, 1);
                    I1(index) = mu2;
                    V = [V I1];
                    new_Z = Zono(c, V);
                else
                    new_Z = [];
                end
                
                S2 = Star(new_V, new_C, new_d, new_predicate_lb, new_predicate_ub, new_Z);

                S = [S1 S2];
            end
            
               

        end
        
        % stepLogSig
        function S = stepLogSig_NoSplit(varargin)
            % @I: input star set
            % @index: index of the neuron
            % @l: l = min(x[index]), lower bound at neuron x[index] 
            % @u: u = min(x[index]), upper bound at neuron x[index]
            % @y_l: = logsig(l); output of logsig at lower bound
            % @y_u: = logsig(u); output of logsig at upper bound
            % @dy_l: derivative of LogSig at the lower bound
            % @dy_u: derivative of LogSig at the upper bound
            
            % @S: output star set
            
            % author: Dung Tran
            % date: 3/19/2020
            
            % author: Sung Woo Choi
            % update: 03/02/2023

            switch nargin
                case 2
                    I = varargin{1};
                    index = varargin{2};
                    dis_opt = [];
                    lp_solver = 'glpk';
                case 3
                    I = varargin{1};
                    index = varargin{2};
                    dis_opt = varargin{3};
                    lp_solver = 'glpk';
                case 4
                    I = varargin{1};
                    index = varargin{2};
                    dis_opt = varargin{3};
                    lp_solver = varargin{4};
                otherwise
                    error('Invalid number of input arguments, should be 1, 2 or 3');
            end

            N = I.dim;
            if strcmp(lp_solver, 'estimate')
                if strcmp(dis_opt, 'display')
                    fprintf('\nComputing lower and upper bounds: ');
                end
                [l, u] = I.estimateRange(index);
            else
                if strcmp(dis_opt, 'display')
                    fprintf('\nComputing lower-bounds: ');
                end
                l = I.getMins(index, [], dis_opt, lp_solver);
                if strcmp(dis_opt, 'display')
                    fprintf('\nComputing upper-bounds: ');  
                end
                u = I.getMaxs(index, [], dis_opt, lp_solver);
            end

            y_l = logsig(l);
            y_u = logsig(u);
            dy_l = logsig('dn', l);
            dy_u = logsig('dn', u);
            
            n = I.nVar + 1;
            if l == u
                new_V = [I.V zeros(I.dim, 1)];
                new_V(index, :) = zeros(1, n+1);
                new_V(index, n+1) = 1; 

                if ~isempty(I.Z)
                    c = I.Z.c;
                    V = I.Z.V;
                    c(index) = y_l;
                    V(index, :) = 0;
                    new_Z = Zono(c, V);
                else
                    new_Z = [];
                end
                S = Star(new_V, I.C, I.d, I.predicate_lb, I.predicate_ub, new_Z);
            
            else
                new_V = [I.V zeros(I.dim, 1)];
                new_V(index, :) = zeros(1, n+1);
                new_V(index, n+1) = 1; 

                m = size(I.C, 1);
                C0 = [I.C zeros(m, 1)];
                d0 = I.d;

                if l > 0
                    % y is convex when x >= 0
                    % constraint 1: y <= y'(l) * (x - l) + y(l)
                    % constarint 2: y <= y'(u) * (x - u) + y(u) 
                    % constraint 3: y >= (y(u) - y(l)) * (x - l) / (u - l) + y(l);

                    % over-approximation constraints 
                    % constraint 1: y <= y'(l) * (x - l) + y(l)
                    C1 = [-dy_l*I.V(index, 2:n) 1];
                    d1 = dy_l* (I.V(index, 1) - l) + y_l; 
                    % constraint 2: y <= y'(u) * (x - u) + y(u)
                    C2 = [-dy_u*I.V(index, 2:n) 1];
                    d2 = dy_u * (I.V(index, 1) - u) + y_u;
                    % constraint 3: y >= (y(u) - y(l)) * (x - l) / (u - l) + y(l);
                    a = (y_u - y_l)/(u - l);
                    C3 = [a*I.V(index, 2:n) -1];
                    d3 = a*(l - I.V(index, 1)) - y_l;
                    
                    new_C = [C0; C1; C2; C3];
                    new_d = [d0; d1; d2; d3];

                    % update outer-zonotope
                    if ~isempty(I.Z)
                        c = I.Z.c;
                        V = I.Z.V;                
                        lamda = min(dy_l, dy_u);
                        mu1 = 0.5*(y_u + y_l - lamda *(u + l));
                        mu2 = 0.5*(y_u - y_l - lamda *(u - l));
                        c(index) = lamda * c(index) + mu1;
                        V(index, :) = lamda * V(index, :); 
                        I1 = zeros(I.dim, 1);
                        I1(index) = mu2;
                        V = [V I1];
                        new_Z = Zono(c, V);
                    else
                        new_Z = [];
                    end
                    
                elseif u <= 0
                    % y is concave when x <= 0
                    % constraint 1: y >= y'(l) * (x - l) + y(l)
                    % constraint 2: y >= y'(u) * (x - u) + y(u)
                    % constraint 3: y <= (y(u) - y(l)) * (x -l) / (u - l) + y(l);
                    
                    n = I.nVar + 1;
                    % over-approximation constraints 
                    % constraint 1: y >= y'(l) * (x - l) + y(l)
                    C1 = [dy_l*I.V(index, 2:n) -1];
                    d1 = -dy_l * (I.V(index, 1) - l) - y_l; 
                    % constraint 2: y >= y'(u) * (x - u) + y(u)
                    C2 = [dy_u*I.V(index, 2:n) -1];
                    d2 = -dy_u * (I.V(index, 1) - u) - y_u;
                    % constraint 3: y <= (y(u) - y(l)) * (x - l) / (u - l) + y(l);
                    a = (y_u - y_l)/(u - l);
                    C3 = [-a*I.V(index, 2:n) 1];
                    d3 = -a*l + y_l + a*I.V(index, 1);
                    
                    new_C = [C0; C1; C2; C3];
                    new_d = [d0; d1; d2; d3];
                    
                    % update outer-zonotope
                    if ~isempty(I.Z)
                        c = I.Z.c;
                        V = I.Z.V;                
                        lamda = min(dy_l, dy_u);
                        mu1 = 0.5*(y_u + y_l - lamda *(u + l));
                        mu2 = 0.5*(y_u - y_l - lamda *(u - l));
                        c(index) = lamda * c(index) + mu1;
                        V(index, :) = lamda * V(index, :); 
                        I1 = zeros(I.dim, 1);
                        I1(index) = mu2;
                        V = [V I1];
                        new_Z = Zono(c, V);
                    else
                        new_Z = [];
                    end
                   
                elseif l <= 0 && u >0
                    % y is concave for x in [l, 0] and convex for x
                    % in [0, u]
                    % split can be done here 
                    
                    % over-approximation constraints 
                    % constraint 1: y >= y'(l) * (x - l) + y(l)
                    % constraint 2: y <= y'(u) * (x - u) + y(u)
                    % constraint 3: y <= (y(x1) - y(l))*(x - l)/(x1 - l) + y(l);
                    % constraint 4: y >= (y(x2) - y(u)) * (x - u)/(x2 - u) + y(u)
                    
                    dmin = min(dy_l , dy_u);
                    % constraint 1: y >= y'(l) * (x - l) + y(l)
                    C1 = [dmin*I.V(index, 2:n) -1];
                    d1 = -dmin * (I.V(index, 1) - l) - y_l; 
                    % constraint 2: y <= y'(u) * (x - u) + y(u)
                    C2 = [-dmin*I.V(index, 2:n) 1];
                    d2 = dmin * (I.V(index, 1) - u) + y_u;
                    
                    gu_x = (y_u - dmin * u - 0.5) / (0.25 - dmin); % upper intersect x
                    gu_y = 0.25 * gu_x + 0.5;% upper intersect y
                    gl_x = (y_l - dmin * l - 0.5) / (0.25 - dmin); % lower intersect x
                    gl_y = 0.25 * gl_x + 0.5; % lower intersect y
    
                    m_u = (y_l - gu_y)/(l - gu_x);
                    m_l = (y_u - gl_y)/(u - gl_x);
                    
                    % constraint 3: y[index] >= m_l * x[index] - m_l*u + y_u
                    C3 = [m_l*I.V(index, 2:n), -1];
                    d3 = -m_l*(I.V(index, 1) - u) - y_u;
                                    
                    % constraint 4: y[index] <= m_u * x[index] - m_u*l + y_l
                    C4 = [-m_u*I.V(index, 2:n), 1];
                    d4 = m_u*(I.V(index, 1) - l) + y_l;

                    new_C = [C0; C1; C2; C3; C4];
                    new_d = [d0; d1; d2; d3; d4];

                    % update outer-zonotope
                    if ~isempty(I.Z)
                        c = I.Z.c;
                        V = I.Z.V;                
                        lamda = min(dy_l, dy_u);
                        mu1 = 0.5*(y_u + y_l - lamda *(u + l));
                        mu2 = 0.5*(y_u - y_l - lamda *(u - l));
                        c(index) = lamda * c(index) + mu1;
                        V(index, :) = lamda * V(index, :); 
                        I1 = zeros(I.dim, 1);
                        I1(index) = mu2;
                        V = [V I1];
                        new_Z = Zono(c, V);
                    else
                        new_Z = [];
                    end


                end

                % update predicate bound
                new_predicate_lb = [I.predicate_lb; y_l]; 
                new_predicate_ub = [I.predicate_ub; y_u];

                S = Star(new_V, new_C, new_d, new_predicate_lb, new_predicate_ub, new_Z);
                                
            end

        end
        
        
        % multistepLogSig
        function S = multiStepLogSig_NoSplit(varargin)
            % @I: input star set
            
            % @l: l = min(x[index]), lower bound at neuron x[index] 
            % @u: u = min(x[index]), upper bound at neuron x[index]
            % @yl: = logsig(l); output of logsig at lower bound
            % @yu: = logsig(u); output of logsig at upper bound
            % @dyl: derivative of LogSig at the lower bound
            % @dyu: derivative of LogSig at the upper bound
            
            % @S: output star set
            
            % author: Dung Tran
            % date: 6/12/2020
            % update: 7/15/2020: add display option 
            %         7/16/2020: add lp_solver option
			
			% author: Sung Woo Choi
            % update: 03/02/2023
            
            switch nargin
                case 1
                    I = varargin{1};
                    dis_opt = [];
                    lp_solver = 'glpk';
                case 2
                    I = varargin{1};
                    dis_opt = varargin{2};
                    lp_solver = 'glpk';
                case 3
                    I = varargin{1};
                    dis_opt = varargin{2};
                    lp_solver = varargin{3};
                otherwise
                    error('Invalid number of input arguments, should be 1, 2 or 3');
            end

            N = I.dim;
            inds = 1:N;
            if strcmp(lp_solver, 'estimate')
                if strcmp(dis_opt, 'display')
                    fprintf('\nComputing lower and upper bounds: ');
                end
                [l, u] = I.estimateRanges;
            else
                if strcmp(dis_opt, 'display')
                    fprintf('\nComputing lower-bounds: ');
                end
                l = I.getMins(inds, [], dis_opt, lp_solver);
                if strcmp(dis_opt, 'display')
                    fprintf('\nComputing upper-bounds: ');  
                end
                u = I.getMaxs(inds, [], dis_opt, lp_solver);
            end
            
            yl = logsig(l);
            yu = logsig(u);
            dyl = logsig('dn', l);
            dyu = logsig('dn', u);

            % l ~= u
            map2 = find(l ~= u);
            m = length(map2);
            V2 = zeros(N, m);
            for i=1:m
                V2(map2(i), i) = 1;
            end

            % new basis matrix
            new_V = [zeros(N, I.nVar+1) V2];
            
             % l == u
            map1 = find(l == u);
            yl1 = yl(map1(:));         
            new_V(map1, 1) = yl1;
            new_V(map1, 2:I.nVar+1+m) = 0;

            % add new constraints

            % C0, d0
            n = size(I.C, 1);
            C0 = [I.C zeros(n, m)];
            d0 = I.d;

            nv = I.nVar+1;

            % C1, d1, x >= 0
            % y is convex when x >= 0
            % constraint 1: y <= y'(l) * (x - l) + y(l)
            % constarint 2: y <= y'(u) * (x - u) + y(u) 
            % constraint 3: y >= (y(u) - y(l)) * (x - l) / (u - l) + y(l);
            map1 = find(l >= 0 & l~=u);
            if ~isempty(map1)
                a = yl(map1);
                b = yu(map1);
                da = dyl(map1);
                db = dyu(map1);
                % constraint 1: y <= y'(l) * (x - l) + y(l)
                C11 = [-da.*I.V(map1, 2:nv) V2(map1, :)];
                d11 = da.*(I.V(map1, 1)-l(map1)) + a;
                % constraint 2: y <= y'(u) * (x - u) + y(u) 
                C12 = [-db.*I.V(map1, 2:nv) V2(map1, :)];
                d12 = db.*(I.V(map1, 1) - u(map1)) + b;
                % constraint 3: y >= (y(u) - y(l)) * (x - l) / (u - l) + y(l);
                gamma = (b-a)./(u(map1)-l(map1));
                C13 = [gamma.*I.V(map1, 2:nv) -V2(map1, :)];
                d13 = -gamma.*(I.V(map1, 1)-l(map1)) - a;

                C1 = [C11; C12; C13]; 
                d1 = [d11; d12; d13];
            else
                C1 = [];
                d1 = [];                
            end
            

            % C2, d2, x <= 0 
            % y is concave when x <= 0
            % constraint 1: y >= y'(l) * (x - l) + y(l)
            % constraint 2: y >= y'(u) * (x - u) + y(u)
            % constraint 3: y <= (y(u) - y(l)) * (x -l) / (u - l) + y(l);

            map1 = find(u <= 0 & l~=u);
            if ~isempty(map1)
                a = yl(map1);
                b = yu(map1);
                da = dyl(map1);
                db = dyu(map1);

                % constraint 1: y >= y'(l) * (x - l) + y(l)
                C21 = [da.*I.V(map1, 2:nv) -V2(map1, :)];
                d21 = -da.*(I.V(map1, 1)-l(map1)) - a;
                % constraint 2: y >= y'(u) * (x - u) + y(u) 
                C22 = [db.*I.V(map1, 2:nv) -V2(map1, :)];
                d22 = -db.*(I.V(map1, 1) - u(map1)) - b;
                % constraint 3: y <= (y(u) - y(l)) * (x -l) / (u - l) + y(l);
                gamma = (b-a)./(u(map1)-l(map1));
                C23 = [-gamma.*I.V(map1, 2:nv) V2(map1, :)];
                d23 = gamma.*(I.V(map1, 1)-l(map1)) + a;

                C2 = [C21; C22; C23]; 
                d2 = [d21; d22; d23];
            else
                C2 = [];
                d2 = [];
            end
            
            % C3, d3, l< 0 & u > 0, x >0 or x < 0
            %y is concave for x in [l, 0] and convex for x
            % in [0, u]
            % split can be done here            

            map1 = find(l < 0 & u > 0);
            if ~isempty(map1)
                a = yl(map1);
                b = yu(map1);
                da = dyl(map1);
                db = dyu(map1);

                dmin = (min(da', db'))';
                % over-approximation constraints 
                % constraint 1: y >= min(y'(l), y'(u)) * (x - l) + y(l)
                % constraint 2: y <= min(y'(l), y'(u)) * (x - u) + y(u)

                % constraint 1: y >= min(y'(l), y'(u)) * (x - l) + y(l)
                C31 = [dmin.*I.V(map1, 2:nv) -V2(map1, :)];
                d31 = -dmin.*(I.V(map1, 1)-l(map1)) - a;
                % constraint 2: y <= min(y'(l), y'(u)) * (x - u) + y(u) 
                C32 = [-dmin.*I.V(map1, 2:nv) V2(map1, :)];
                d32 = dmin.*(I.V(map1, 1) - u(map1)) + b;


%                 y1 = dmin.*(-l(map1)) + a;
%                 y2 = dmin.*(-u(map1)) + b;
%                 g2 = (y2 - a)./(-l(map1));
%                 g1 = (y1 - b)./(-u(map1));
% 
%                 % constraint 3: y <= g2 * x + y2
%                 C33 = [-g2.*I.V(map1, 2:nv) V2(map1, :)];
%                 d33 = g2.*I.V(map1, 1) + y2;
% 
%                 % constraint 4: y >= g1 * x + y1
%                 C34 = [g1.*I.V(map1, 2:nv) -V2(map1, :)];
%                 d34 = -g1.*I.V(map1, 1) - y1;

                l_map = l(map1);
                u_map = u(map1);
                gu_x = (b - dmin .* u_map - 0.5) ./ (0.25 - dmin); % upper intersect x
                gu_y = 0.25.* gu_x + 0.5;% upper intersect y
                gl_x = (a - dmin .* l_map - 0.5) ./ (0.25 - dmin); % lower intersect x
                gl_y = 0.25.* gl_x + 0.5; % lower intersect y
                
                m_u = (a - gu_y)./(l_map - gu_x);
                m_l = (b - gl_y)./(u_map - gl_x);

                % constraint 3: y[index] >= m_l * x[index] - m_l*u + y_u
                C33 = [m_l.*I.V(map1, 2:nv), -V2(map1, :)];
                d33 = -m_l.*(I.V(map1, 1) - u_map) - b;
                
                % constraint 4: y[index] <= m_u * x[index] - m_u*l + y_l
                C34 = [-m_u.*I.V(map1, 2:nv), V2(map1, :)];
                d34 = m_u.*(I.V(map1, 1) - l_map) + a;

                C3 = [C31; C32; C33; C34]; 
                d3 = [d31; d32; d33; d34];
            else
                C3 = [];
                d3 = [];
            end
            
            n = size(I.C, 1);
            if n == 1 && nnz(I.C) == 0
                C0 = [];
                d0 = [];
            else
                C0 = [I.C zeros(n, m)];
                d0 = I.d;
            end

            new_C = [C0; C1; C2; C3];
            new_d = [d0; d1; d2; d3]; 

            new_pred_lb = [I.predicate_lb; yl(map2)];
            new_pred_ub = [I.predicate_ub; yu(map2)];

            S = Star(new_V, new_C, new_d, new_pred_lb, new_pred_ub);
        end
        
        
        % multistepLogSig
        function S = relaxedMultiStepLogSig_NoSplit(varargin)
            % @I: input star set
            % @relaxFactor: percentage of optimized ranges used for
            % constructing reachable set
            
            % @l: l = min(x[index]), lower bound at neuron x[index] 
            % @u: u = min(x[index]), upper bound at neuron x[index]
            % @yl: = logsig(l); output of logsig at lower bound
            % @yu: = logsig(u); output of logsig at upper bound
            % @dyl: derivative of LogSig at the lower bound
            % @dyu: derivative of LogSig at the upper bound
            
            % @S: output star set
            
            % author: Dung Tran
            % date: 6/12/2020
            % update: 7/15/2020 add display option

            % author: Sung Woo Choi
            % update: 03/02/2023
            
            switch nargin
                case 2 
                    I = varargin{1};
                    relaxFactor = varargin{2};
                    dis_opt = [];
                    lp_solver = 'glpk';
                case 3
                    I = varargin{1};
                    relaxFactor = varargin{2};
                    dis_opt = varargin{3};
                    lp_solver = 'glpk';
                case 4
                    I = varargin{1};
                    relaxFactor = varargin{2};
                    dis_opt = varargin{3};
                    lp_solver = varargin{4};
                otherwise
                    error('Invalid number of input arguments, should be 2, 3 or 4');
            end
            
            if ~isa(I, 'Star')
                error('Input is not a star');
            end
            if (relaxFactor < 0) || (relaxFactor > 1)
                error('Invalid relax factor');
            end
            
            
            [l, u] = I.estimateRanges;
            n1 = round((1-relaxFactor)*length(l));
            [~, midx] = sort(u - l, 'descend');
            
            N = I.dim;
            if strcmp(dis_opt, 'display')
                fprintf('\nComputing (1-%.3f) x %d = %d lower-bounds, i.e. relaxing %2.2f%%: ' , relaxFactor, length(l), n1, 100*relaxFactor);
            end
            l2 = I.getMins(midx(1:n1), [], dis_opt, lp_solver);
            if strcmp(dis_opt, 'display')
                fprintf('\nComputing (1-%.3f) x %d = %d upper-bounds, i.e. relaxing %2.2f%%: ' , relaxFactor, length(l), n1, 100*relaxFactor);
            end
            u2 = I.getMaxs(midx(1:n1), [], dis_opt, lp_solver);
            l(midx(1:n1)) = l2;
            u(midx(1:n1)) = u2;
            
            yl = logsig(l);
            yu = logsig(u);
            dyl = logsig('dn', l);
            dyu = logsig('dn', u);

            % l ~= u
            map2 = find(l ~= u);
            m = length(map2);
            V2 = zeros(N, m);
            for i=1:m
                V2(map2(i), i) = 1;
            end

            % new basis matrix
            new_V = [zeros(N, I.nVar+1) V2];
            
             % l == u
            map1 = find(l == u);
            yl1 = yl(map1(:));         
            new_V(map1, 1) = yl1;
            new_V(map1, 2:I.nVar+1+m) = 0;

            % add new constraints

            % C0, d0
            n = size(I.C, 1);
            C0 = [I.C zeros(n, m)];
            d0 = I.d;
            
            nv = I.nVar+1;

            % C1, d1, x >= 0
            % y is convex when x >= 0
            % constraint 1: y <= y'(l) * (x - l) + y(l)
            % constarint 2: y <= y'(u) * (x - u) + y(u) 
            % constraint 3: y >= (y(u) - y(l)) * (x - l) / (u - l) + y(l);
            map1 = find(l >= 0 & l~=u);
            if ~isempty(map1)
                a = yl(map1);
                b = yu(map1);
                da = dyl(map1);
                db = dyu(map1);
                % constraint 1: y <= y'(l) * (x - l) + y(l)
                C11 = [-da.*I.V(map1, 2:nv) V2(map1, :)];
                d11 = da.*(I.V(map1, 1)-l(map1)) + a;
                % constraint 2: y <= y'(u) * (x - u) + y(u) 
                C12 = [-db.*I.V(map1, 2:nv) V2(map1, :)];
                d12 = db.*(I.V(map1, 1) - u(map1)) + b;
                % constraint 3: y >= (y(u) - y(l)) * (x - l) / (u - l) + y(l);
                gamma = (b-a)./(u(map1)-l(map1));
                C13 = [gamma.*I.V(map1, 2:nv) -V2(map1, :)];
                d13 = -gamma.*(I.V(map1, 1)-l(map1)) - a;

                C1 = [C11; C12; C13]; 
                d1 = [d11; d12; d13];
            else
                C1 = [];
                d1 = [];                
            end
            

            % C2, d2, x <= 0 
            % y is concave when x <= 0
            % constraint 1: y >= y'(l) * (x - l) + y(l)
            % constraint 2: y >= y'(u) * (x - u) + y(u)
            % constraint 3: y <= (y(u) - y(l)) * (x -l) / (u - l) + y(l);

            map1 = find(u <= 0 & l~=u);
            if ~isempty(map1)
                a = yl(map1);
                b = yu(map1);
                da = dyl(map1);
                db = dyu(map1);

                % constraint 1: y >= y'(l) * (x - l) + y(l)
                C21 = [da.*I.V(map1, 2:nv) -V2(map1, :)];
                d21 = -da.*(I.V(map1, 1)-l(map1)) - a;
                % constraint 2: y >= y'(u) * (x - u) + y(u) 
                C22 = [db.*I.V(map1, 2:nv) -V2(map1, :)];
                d22 = -db.*(I.V(map1, 1) - u(map1)) - b;
                % constraint 3: y <= (y(u) - y(l)) * (x -l) / (u - l) + y(l);
                gamma = (b-a)./(u(map1)-l(map1));
                C23 = [-gamma.*I.V(map1, 2:nv) V2(map1, :)];
                d23 = gamma.*(I.V(map1, 1)-l(map1)) + a;

                C2 = [C21; C22; C23]; 
                d2 = [d21; d22; d23];
            else
                C2 = [];
                d2 = [];
            end
            
            % C3, d3, l< 0 & u > 0, x >0 or x < 0
            %y is concave for x in [l, 0] and convex for x
            % in [0, u]
            % split can be done here            

            map1 = find(l < 0 & u > 0);
            if ~isempty(map1)
                a = yl(map1);
                b = yu(map1);
                da = dyl(map1);
                db = dyu(map1);

                dmin = (min(da', db'))';
                % over-approximation constraints 
                % constraint 1: y >= min(y'(l), y'(u)) * (x - l) + y(l)
                % constraint 2: y <= min(y'(l), y'(u)) * (x - u) + y(u)

                % constraint 1: y >= min(y'(l), y'(u)) * (x - l) + y(l)
                C31 = [dmin.*I.V(map1, 2:nv) -V2(map1, :)];
                d31 = -dmin.*(I.V(map1, 1)-l(map1)) - a;
                % constraint 2: y <= min(y'(l), y'(u)) * (x - u) + y(u) 
                C32 = [-dmin.*I.V(map1, 2:nv) V2(map1, :)];
                d32 = dmin.*(I.V(map1, 1) - u(map1)) + b;


%                 y1 = dmin.*(-l(map1)) + a;
%                 y2 = dmin.*(-u(map1)) + b;
%                 g2 = (y2 - a)./(-l(map1));
%                 g1 = (y1 - b)./(-u(map1));
% 
%                 % constraint 3: y <= g2 * x + y2
%                 C33 = [-g2.*I.V(map1, 2:nv) V2(map1, :)];
%                 d33 = g2.*I.V(map1, 1) + y2;
% 
%                 % constraint 4: y >= g1 * x + y1
%                 C34 = [g1.*I.V(map1, 2:nv) -V2(map1, :)];
%                 d34 = -g1.*I.V(map1, 1) - y1;

                l_map = l(map1);
                u_map = u(map1);
                gu_x = (b - dmin .* u_map - 0.5) ./ (0.25 - dmin); % upper intersect x
                gu_y = 0.25.* gu_x + 0.5;% upper intersect y
                gl_x = (a - dmin .* l_map - 0.5) ./ (0.25 - dmin); % lower intersect x
                gl_y = 0.25.* gl_x + 0.5; % lower intersect y
                
                m_u = (a - gu_y)./(l_map - gu_x);
                m_l = (b - gl_y)./(u_map - gl_x);

                % constraint 3: y[index] >= m_l * x[index] - m_l*u + y_u
                C33 = [m_l.*I.V(map1, 2:nv) -V2(map1, :)];
                d33 = -m_l.*(I.V(map1, 1) - u_map) - b;
                
                % constraint 4: y[index] <= m_u * x[index] - m_u*l + y_l
                C34 = [-m_u.*I.V(map1, 2:nv) V2(map1, :)];
                d34 = m_u.*(I.V(map1, 1) - l_map) + a;

                C3 = [C31; C32; C33; C34]; 
                d3 = [d31; d32; d33; d34];
            else
                C3 = [];
                d3 = [];
            end

            n = size(I.C, 1);
            if n == 1 && nnz(I.C) == 0
                C0 = [];
                d0 = [];
            else
                C0 = [I.C zeros(n, m)];
                d0 = I.d;
            end
            
            new_C = [C0; C1; C2; C3];
            new_d = [d0; d1; d2; d3]; 

            new_pred_lb = [I.predicate_lb; yl(map2)];
            new_pred_ub = [I.predicate_ub; yu(map2)];

            S = Star(new_V, new_C, new_d, new_pred_lb, new_pred_ub);
        end
        
        
    end
    
    
    methods(Static) % over-approximate reachability analysis using Zonotope
        
        function Z = reach_zono_approx(I)
            % @I: zonotope input set
            % @Z: zonotope output set
            
            % author: Dung Tran
            % date: 5/3/2019
            
            % reference: Fast and Effective Robustness Certification,
            % Gagandeep Singh, NIPS, 2018
            
            if ~isa(I, 'Zono')
                error('Input set is not a Zonotope');
            end
            
            B = I.getBox;
            
            lb = B.lb;
            ub = B.ub;
            G = [logsig('dn', lb) logsig('dn', ub)];
            gamma_opt = min(G, [], 2);
            gamma_mat = diag(gamma_opt);
            mu1 = 0.5 * (logsig(ub) + logsig(lb) - gamma_mat * (ub + lb));
            mu2 = 0.5 * (logsig(ub) - logsig(lb) - gamma_mat * (ub - lb));
            Z1 = I.affineMap(gamma_mat, mu1);
            new_V = diag(mu2);
            V = [Z1.V new_V];
            Z = Zono(Z1.c, V);
            
        end
        
        % dealing with multiple inputs in parallel
        function S = reach_zono_approx_multipleInputs(varargin)
            % author: Dung Tran
            % date: 3/27/2020
            
            switch nargin
                case 1
                    I = varargin{1};
                    parallel = []; % no parallel computation
                case 2
                    I = varargin{1};
                    parallel = varargin{2};
                otherwise
                    error('Invalid number of input arguments, should be 1 or 2');
            end
            
            p = length(I);
            S = [];
            if isempty(parallel)
                
                for i=1:p
                    S =[S LogSig.reach_zono_approx(I(i))];
                end
                
            elseif strcmp(parallel, 'parallel')
                
                parfor i=1:p
                    S =[S, LogSig.reach_zono_approx(I(i))];
                end
                
            else
                error('Unknown parallel computation option');
            end

        end
        
    end
    
    methods(Static) % over-approximate reachability analysis using abstract domain
        
        
        function S = stepLogSig_absdom(I, index, l, u, y_l, y_u, dy_l, dy_u)
            % @I: input star set
            % @index: index of the neuron
            % @l: l = min(x[index]), lower bound at neuron x[index] 
            % @u: u = min(x[index]), upper bound at neuron x[index]
            % @y_l: = logsig(l); output of logsig at lower bound
            % @y_u: = logsig(u); output of logsig at upper bound
            % @dy_l: derivative of LogSig at the lower bound
            % @dy_u: derivative of LogSig at the upper bound
    
            % @S: output star set
    
            % author: Dung Tran
            % date: 3/27/2020
                    
            if l == u
                new_V = I.V;
                new_V(index, 1:I.nVar+1) = 0;
                new_V(index, 1) = y_l;
                S = Star(new_V, I.C, I.d, I.predicate_lb, I.predicate_ub);                
            elseif l >= 0
                % y is convex when x >= 0
                % constraint 2: y <= y'(u) * (x - u) + y(u) 
                % constraint 3: y >= (y(u) - y(l)) * (x - l) / (u - l) + y(l);
    
    
                n = I.nVar + 1;
                % over-approximation constraints 
                % constraint 2: y <= y'(u) * (x - u) + y(u)
                C2 = [-dy_u*I.V(index, 2:n) 1];
                d2 = dy_u * I.V(index, 1) - dy_u*u + y_u;
                % constraint 3: y >= (y(u) - y(l)) * (x - l) / (u - l) + y(l);
                a = (y_u - y_l)/(u - l);
                C3 = [a*I.V(index, 2:n) -1];
                d3 = a*l - y_l - a*I.V(index, 1);
    
                m = size(I.C, 1);
                C0 = [I.C zeros(m, 1)];
                d0 = I.d;
                new_C = [C0; C2; C3];
                new_d = [d0; d2; d3];
                new_V = [I.V zeros(I.dim, 1)];
                new_V(index, :) = zeros(1, n+1);
                new_V(index, n+1) = 1; 
    
                % update predicate bound
                new_predicate_lb = [I.predicate_lb; y_l]; 
                new_predicate_ub = [I.predicate_ub; y_u];
                S = Star(new_V, new_C, new_d, new_predicate_lb, new_predicate_ub);
    
    
            elseif u <= 0
                % y is concave when x <= 0
                % constraint 1: y >= y'(l) * (x - l) + y(l)
                % constraint 3: y <= (y(u) - y(l)) * (x -l) / (u - l) + y(l);
    
                n = I.nVar + 1;
                % over-approximation constraints 
                % constraint 1: y >= y'(l) * (x - l) + y(l)
                C1 = [dy_l*I.V(index, 2:n) -1];
                d1 = -dy_l * I.V(index, 1) + dy_l*l - y_l; 
                % constraint 3: y <= (y(u) - y(l)) * (x - l) / (u - l) + y(l);
                a = (y_u - y_l)/(u - l);
                C3 = [-a*I.V(index, 2:n) 1];
                d3 = -a*l + y_l + a*I.V(index, 1);
    
                m = size(I.C, 1);
                C0 = [I.C zeros(m, 1)];
                d0 = I.d;
                new_C = [C0; C1; C3];
                new_d = [d0; d1; d3];
                new_V = [I.V zeros(I.dim, 1)];
                new_V(index, :) = zeros(1, n+1);
                new_V(index, n+1) = 1; 
    
                % update predicate bound
                new_predicate_lb = [I.predicate_lb; y_l]; 
                new_predicate_ub = [I.predicate_ub; y_u];
                S = Star(new_V, new_C, new_d, new_predicate_lb, new_predicate_ub);
    
    
            elseif l <0 && u >0
                % y is concave for x in [l, 0] and convex for x
                % in [0, u]
                % split can be done here 
    
                
                % over-approximation constraints 
                % constraint 1: y >= y'(l) * (x - l) + y(l)
                % constraint 2: y <= y'(u) * (x - u) + y(u)
               
                n = I.nVar + 1;
    
                dy_min = min(dy_l, dy_u);
                % constraint 1: y >= y'_min * (x - l) + y(l)
                C1 = [dy_min*I.V(index, 2:n) -1];
                d1 = -dy_min * I.V(index, 1) + dy_min*l - y_l; 
                % constraint 2: y <= y'_min * (x - u) + y(u)
                C2 = [-dy_min*I.V(index, 2:n) 1];
                d2 = dy_min * I.V(index, 1) - dy_min*u + y_u;
                
                m = size(I.C, 1);
                C0 = [I.C zeros(m, 1)];
                d0 = I.d;
                new_C = [C0; C1; C2];
                new_d = [d0; d1; d2];
                new_V = [I.V zeros(I.dim, 1)];
                new_V(index, :) = zeros(1, n+1);
                new_V(index, n+1) = 1; 
    
                % update predicate bound
                new_predicate_lb = [I.predicate_lb; y_l]; 
                new_predicate_ub = [I.predicate_ub; y_u];
                S = Star(new_V, new_C, new_d, new_predicate_lb, new_predicate_ub);
    
            end
    
            
            
        end
        
        
        
        function S = reach_absdom_approx(I)
            % @I: star input set
            % @Z: Star output set
    
            % author: Dung Tran
            % date: 3/27/2020
    
            % reference: An abstract domain for certifying neural networks. Proceedings of the ACM on Programming Languages,
            % Gagandeep Singh, POPL, 2019
    
            if ~isa(I, 'Star')
                error('Input set is not a Star');
            end
    
            [l, u] = I.estimateRanges;  
    
            y_l = logsig(l);
            y_u = logsig(u);
            dy_l = logsig('dn', l);
            dy_u = logsig('dn', u);
    
            n = I.dim;
            S = I;
            for i=1:n
                S = LogSig.stepLogSig_absdom(S, i, l(i), u(i), y_l(i), y_u(i), dy_l(i), dy_u(i)); 
            end

        end

        % dealing with multiple inputs in parallel
        function S = reach_absdom_approx_multipleInputs(varargin)
            % author: Dung Tran
            % date: 3/27/2020

            switch nargin
                case 1
                    I = varargin{1};
                    parallel = []; % no parallel computation
                case 2
                    I = varargin{1};
                    parallel = varargin{2};
                otherwise
                    error('Invalid number of input arguments, should be 1 or 2');
            end

            p = length(I);
            S = [];
            if isempty(parallel)

                for i=1:p
                    S =[S LogSig.reach_absdom_approx(I(i))];
                end

            elseif strcmp(parallel, 'parallel')

                parfor i=1:p
                    S =[S, LogSig.reach_absdom_approx(I(i))];
                end

            else
                error('Unknown parallel computation option');
            end

        end

    
    end

    methods(Static)
        function S = reach_sparse_star_approx(varargin)
            % author: Sung Woo Choi
            % date: 03/07/2023
            
            switch nargin
                case 1
                    I = varargin{1};
                    method = 'approx-sparse-star-no-split';
                    relaxFactor = 0; % for relaxed approx-star method
                    depthReduct = 0;
                    dis_opt = [];
                    lp_solver = 'glpk';
                case 2
                    I = varargin{1};
                    method = varargin{2};
                    relaxFactor = 0; % for relaxed approx-star method
                    depthReduct = 0;
                    dis_opt = [];
                    lp_solver = 'glpk';
                case 3
                    I = varargin{1};
                    method = varargin{2};
                    relaxFactor = varargin{3};
                    depthReduct = 0;
                    dis_opt = [];
                    lp_solver = 'glpk';
                case 4
                    I = varargin{1};
                    method = varargin{2};
                    relaxFactor = varargin{3};
                    depthReduct = varargin{4};
                    dis_opt = []; % display option
                    lp_solver = 'glpk';
                case 5
                    I = varargin{1};
                    method = varargin{2};
                    relaxFactor = varargin{3};
                    depthReduct = varargin{4};
                    dis_opt = varargin{5}; % display option
                    lp_solver = 'glpk'; 
                case 6
                    I = varargin{1};
                    method = varargin{2};
                    relaxFactor = varargin{3};
                    depthReduct = varargin{4};
                    dis_opt = varargin{5}; % display option
                    lp_solver = varargin{6}; 
                otherwise
                    error('Invalid number of input arguments, should be 1, 2, 3, 4, 5, or 6');
            end
            
            if ~isa(I, 'SparseStar')
                error('Input set is not a star set');
            end
           
            if strcmp(method, 'approx-sparse-star-no-split') || strcmp(method, 'approx-sparse-star')
                S = LogSig.sparse_multiStepLogSig_NoSplit(I, relaxFactor, depthReduct, dis_opt, lp_solver);
            else
                error('Unknown reachability method');
            end
            
        end

        % multiStepTanSig at one
        function S = sparse_multiStepLogSig_NoSplit(varargin)
            % @I: input SparseStar set
            
            % @l: l = min(x[index]), lower bound at neuron x[index] 
            % @u: u = min(x[index]), upper bound at neuron x[index]
            % @yl: = logsig(l); output of logsig at lower bound
            % @yu: = logsig(u); output of logsig at upper bound
            % @dyl: derivative of LogSig at the lower bound
            % @dyu: derivative of LogSig at the upper bound
            
            % @S: output SparseStar set
            % author: Sung Woo Choi
            % update: 03/08/2023
            
            switch nargin
                case 1
                    I = varargin{1};
                    relaxFactor = 0;
                    depthReduct = 0;
                    dis_opt = [];
                    lp_solver = 'glpk';
                case 2
                    I = varargin{1};
                    relaxFactor = varargin{2};
                    depthReduct = 0;
                    dis_opt = [];
                    lp_solver = 'glpk';
                case 3
                    I = varargin{1};
                    relaxFactor = varargin{2};
                    depthReduct = varargin{3};
                    dis_opt = [];
                    lp_solver = 'glpk';
                case 4
                    I = varargin{1};
                    relaxFactor = varargin{2};
                    depthReduct = varargin{3};
                    dis_opt = varargin{4};
                    lp_solver = 'glpk';
                case 5
                    I = varargin{1};
                    relaxFactor = varargin{2};
                    depthReduct = varargin{3};
                    dis_opt = varargin{4};
                    lp_solver = varargin{5};
                otherwise
                    error('Invalid number of input arguments, should be 1, 2, 3, 4, or 5');
            end
            
            if ~isa(I, 'SparseStar')
                error('Input is not a SparseStar');
            end
            if (relaxFactor < 0) || (relaxFactor > 1)
                error('Invalid relax factor');
            end

            N = I.dim;
            inds = 1:N;
            if relaxFactor > 0
                [l, u] = I.estimateRanges;
                n1 = round((1-relaxFactor)*length(l));
                [~, midx] = sort(u - l, 'descend');
                
                if strcmp(dis_opt, 'display')
                    fprintf('\nComputing (1-%.3f) x %d = %d lower-bounds, i.e. relaxing %2.2f%%: ' , relaxFactor, length(l), n1, 100*relaxFactor);
                end
                l2 = I.getMins(midx(1:n1), [], dis_opt, lp_solver);
                if strcmp(dis_opt, 'display')
                    fprintf('\nComputing (1-%.3f) x %d = %d upper-bounds, i.e. relaxing %2.2f%%: ' , relaxFactor, length(l), n1, 100*relaxFactor);
                end
                u2 = I.getMaxs(midx(1:n1), [], dis_opt, lp_solver);
                l(midx(1:n1)) = l2;
                u(midx(1:n1)) = u2;
            elseif strcmp(lp_solver, 'estimate')
                if strcmp(dis_opt, 'display')
                    fprintf('\nComputing lower and upper bounds: ');
                end
                [l, u] = I.estimateRanges;
            else
                if strcmp(dis_opt, 'display')
                    fprintf('\nComputing lower-bounds: ');
                end
                l = I.getMins(inds, [], dis_opt, lp_solver);
                if strcmp(dis_opt, 'display')
                    fprintf('\nComputing upper-bounds: ');  
                end
                u = I.getMaxs(inds, [], dis_opt, lp_solver);
            end

%             map = find(u > 0.99999);
%             u(map) = 1;
%             map = find(l < -0.99999);
%             l(map) = -1;

            yl = logsig(l);
            yu = logsig(u);
            dyl = logsig('dn', l);
            dyu = logsig('dn', u);

            % l ~= u
            map2 = find(l ~= u);
            m = length(map2);
            isAsparse = m > 3;
            if isAsparse
                A2 = sparse(N, m);
            else
                A2 = zeros(N, m);
            end
            for i=1:m
                A2(map2(i), i) = 1;
            end

            % new basis matrix
            if  m > 3
                new_A = [sparse(N, 1), A2];
            else
                new_A = [zeros(N, 1), A2];
            end
            
             % l == u
            map1 = find(l == u);
            yl1 = yl(map1(:));         
            new_A(map1, 1) = yl1;
            new_A(map1, 2:m+1) = 0;

            % add new constraints

            [nA, mA] = size(I.A);
            mZ = I.nVar+1-mA;
            isSparse = (nnz(I.A(:, 2:mA)) + N)  < 0.5 * (mZ*I.dim + nA*(mA-1) + N*m);
            if isSparse
                A2 = sparse(A2);
            end
            
            % C1, d1, x >= 0
            % y is convex when x >= 0
            % constraint 1: y <= y'(l) * (x - l) + y(l)
            % constarint 2: y <= y'(u) * (x - u) + y(u) 
            % constraint 3: y >= (y(u) - y(l)) * (x - l) / (u - l) + y(l);
            map1 = find(l >= 0 & l~=u);
            if ~isempty(map1)
                a = yl(map1);
                b = yu(map1);
                da = dyl(map1);
                db = dyu(map1);
                if isSparse
                    Z = sparse(length(map1), mZ);
                else
                    Z = zeros(length(map1), mZ);
                end
                % constraint 1: y <= y'(l) * (x - l) + y(l)
                C11 = [Z, -da.*I.A(map1, 2:mA), A2(map1, :)];
                d11 = da.*(I.A(map1, 1)-l(map1)) + a;
                % constraint 2: y <= y'(u) * (x - u) + y(u) 
                C12 = [Z, -db.*I.A(map1, 2:mA), A2(map1, :)];
                d12 = db.*(I.A(map1, 1) - u(map1)) + b;
                % constraint 3: y >= (y(u) - y(l)) * (x - l) / (u - l) + y(l);
                gamma = (b-a)./(u(map1)-l(map1));
                C13 = [Z, gamma.*I.A(map1, 2:mA), -A2(map1, :)];
                d13 = -gamma.*(I.A(map1, 1)-l(map1)) - a;

                C1 = [C11; C12; C13]; 
                d1 = [d11; d12; d13];
            else
                C1 = [];
                d1 = [];                
            end
            

            % C2, d2, x <= 0 
            % y is concave when x <= 0
            % constraint 1: y >= y'(l) * (x - l) + y(l)
            % constraint 2: y >= y'(u) * (x - u) + y(u)
            % constraint 3: y <= (y(u) - y(l)) * (x -l) / (u - l) + y(l);

            map1 = find(u <= 0 & l~=u);
            if ~isempty(map1)
                a = yl(map1);
                b = yu(map1);
                da = dyl(map1);
                db = dyu(map1);
                if isSparse
                    Z = sparse(length(map1), mZ);
                else
                    Z = zeros(length(map1), mZ);
                end
                % constraint 1: y >= y'(l) * (x - l) + y(l)
                C21 = [Z, da.*I.A(map1, 2:mA), -A2(map1, :)];
                d21 = -da.*(I.A(map1, 1)-l(map1)) - a;
                % constraint 2: y >= y'(u) * (x - u) + y(u) 
                C22 = [Z, db.*I.A(map1, 2:mA), -A2(map1, :)];
                d22 = -db.*(I.A(map1, 1) - u(map1)) - b;
                % constraint 3: y <= (y(u) - y(l)) * (x -l) / (u - l) + y(l);
                gamma = (b-a)./(u(map1)-l(map1));
                C23 = [Z, -gamma.*I.A(map1, 2:mA), A2(map1, :)];
                d23 = gamma.*(I.A(map1, 1)-l(map1)) + a;

                C2 = [C21; C22; C23]; 
                d2 = [d21; d22; d23];
            else
                C2 = [];
                d2 = [];
            end
            
            % C3, d3, l< 0 & u > 0, x >0 or x < 0
            %y is concave for x in [l, 0] and convex for x
            % in [0, u]
            % split can be done here            

            map1 = find(l < 0 & u > 0);
            if ~isempty(map1)
                a = yl(map1);
                b = yu(map1);
                da = dyl(map1);
                db = dyu(map1);

                dmin = (min(da', db'))';
                if isSparse
                    Z = sparse(length(map1), mZ);
                else
                    Z = zeros(length(map1), mZ);
                end
                % over-approximation constraints 
                % constraint 1: y >= min(y'(l), y'(u)) * (x - l) + y(l)
                % constraint 2: y <= min(y'(l), y'(u)) * (x - u) + y(u)
                
                % constraint 1: y >= min(y'(l), y'(u)) * (x - l) + y(l)
                C31 = [Z, dmin.*I.A(map1, 2:mA), -A2(map1, :)];
                d31 = -dmin.*(I.A(map1, 1)-l(map1)) - a;
                % constraint 2: y <= min(y'(l), y'(u)) * (x - u) + y(u) 
                C32 = [Z, -dmin.*I.A(map1, 2:mA), A2(map1, :)];
                d32 = dmin.*(I.A(map1, 1) - u(map1)) + b;

                l_map = l(map1);
                u_map = u(map1);
                gu_x = (b - dmin .* u_map - 0.5) ./ (0.25 - dmin); % upper intersect x
                gu_y = 0.25 .* gu_x + 0.5;% upper intersect y
                gl_x = (a - dmin .* l_map - 0.5) ./ (0.25 - dmin); % lower intersect x
                gl_y = 0.25 .* gl_x + 0.5; % lower intersect y
                
                m_u = (a - gu_y)./(l_map - gu_x);
                m_l = (b - gl_y)./(u_map - gl_x);

                % constraint 3: y[index] >= m_l * x[index] - m_l*u + y_u
                C33 = [Z, m_l.*I.A(map1, 2:mA), -A2(map1, :)];
                d33 = -m_l.*(I.A(map1, 1) - u_map) - b;
                
                % constraint 4: y[index] <= m_u * x[index] - m_u*l + y_l
                C34 = [Z, -m_u.*I.A(map1, 2:mA), A2(map1, :)];
                d34 = m_u.*(I.A(map1, 1) - l_map) + a;

                C3 = [C31; C32; C33; C34]; 
                d3 = [d31; d32; d33; d34];
            else
                C3 = [];
                d3 = [];
            end
            
            % C0, d0
            n = size(I.C, 1);
            if n == 1 && nnz(I.C) == 0
                C0 = [];
                d0 = [];
            else
                C0 = [I.C zeros(n, m)];
                d0 = I.d;
            end
            
            new_C = [C0; C1; C2; C3];
            new_d = [d0; d1; d2; d3]; 

            new_pred_lb = [I.pred_lb; yl(map2)];
            new_pred_ub = [I.pred_ub; yu(map2)];
            pd1 = I.pred_depth + 1;
            new_pred_depth = [pd1; zeros(m, 1)];

            S = SparseStar(new_A, new_C, new_d, new_pred_lb, new_pred_ub, new_pred_depth);
            if depthReduct > 0
                S = S.depthReduction(depthReduct);
            end
        end

    end
    

methods(Static) % main reach method

    % main function for reachability analysis
    function R = reach(varargin)
        % @I: an array of star input sets
        % @method: 'approx-star' or 'approx-zono' or 'abs-dom' 
        % @option: = 'parallel' or [] using parallel computation or not

        % author: Dung Tran
        % date: 3/27/2019
        % update: 4/2/2020
        % update: 6/26/2020: add relaxed approx-star method
        % update 7/15/2020: add display option
        %        7/16/2020: add lp_solver option
            
        switch nargin
            case 7
                I = varargin{1};
                method = varargin{2};
                reachOption = varargin{3};
                relaxFactor = varargin{4}; % used for aprox-star only
                depthReduct = varargin{5};
                dis_opt = varargin{6}; % display option
                lp_solver = varargin{7};
            case 6
                I = varargin{1};
                method = varargin{2};
                reachOption = varargin{3};
                relaxFactor = varargin{4}; % used for aprox-star only
                depthReduct = 0;
                dis_opt = varargin{5}; % display option
                lp_solver = varargin{6};
            case 5
                I = varargin{1};
                method = varargin{2};
                reachOption = varargin{3};
                relaxFactor = varargin{4}; % used for aprox-star only
                depthReduct = 0;
                dis_opt = varargin{5}; % display option
                lp_solver = 'glpk';
            case 4
                I = varargin{1};
                method = varargin{2};
                reachOption = varargin{3};
                relaxFactor = varargin{4}; % used for aprox-star only
                depthReduct = 0;
                dis_opt = [];
                lp_solver = 'glpk';
            case 3
                I = varargin{1};
                method = varargin{2};
                reachOption = varargin{3};
                relaxFactor = 0; % used for aprox-star only
                depthReduct = 0;
                dis_opt = [];
                lp_solver = 'glpk';
            case 2
                I = varargin{1};
                method = varargin{2};
                reachOption = [];
                relaxFactor = 0; % used for aprox-star only
                depthReduct = 0;
                dis_opt = [];
                lp_solver = 'glpk';
            case 1
                I = varargin{1};
                method = 'approx-star';
                reachOption = [];
                relaxFactor = 0; % for relaxed approx-star method
                depthReduct = 0;
                dis_opt = [];
                lp_solver = 'glpk';
            otherwise
                error('Invalid number of input arguments (should be between 1 and 7');
        end

        

        if strcmp(method, 'approx-star') || strcmp(method, 'approx-star-no-split') || strcmp(method, 'approx-star-split') 

            R = LogSig.reach_star_approx(I, method, reachOption, relaxFactor, dis_opt, lp_solver);

        elseif strcmp(method, 'approx-sparse-star')
            
            R = LogSig.reach_sparse_star_approx(I, method, relaxFactor, depthReduct, dis_opt, lp_solver);

        elseif strcmp(method, 'approx-zono')  % over-approximate analysis using zonotope

            R = LogSig.reach_zono_approx(I);

        elseif strcmp(method, 'abs-dom')  % over-approximate analysis using abstract-domain

            R = LogSig.reach_absdom_approx(I);

        else
            error('Unknown or unsupported reachability method for layer with LogSig activation function');
        end
            


    end
    
end

    
end

