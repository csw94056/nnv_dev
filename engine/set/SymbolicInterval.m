classdef SymbolicInterval
    % Symbolic Interval class
    properties
        Dl = {};
        Du = {};
        gl = {};
        gu = {};
        lb = [];
        ub = [];
        l0 = [];
        u0 = [];
        dim = 0;
    end

    methods
        function obj = SymbolicInterval(varargin)
            gpuAvailable = parallel.gpu.GPUDevice.isAvailable;

            switch nargin
                case 8
                    Dl = varargin{1};
                    Du = varargin{2};
                    gl = varargin{3};
                    gu = varargin{4};
                    lb = varargin{5};
                    ub = varargin{6};
                    l0 = varargin{7};
                    u0 = varargin{8};

                    % Dl, Du, gl, gu need to be cell array
                    if size(Dl,1) ~= 1
                        Dl = {Dl};
                        Du = {Du};
                        gu = {gu};
                        gl = {gl};
                    end

                    len = length(Dl);
                    if all(size(Dl{len}) ~= size(Du{len})) || all(size(gl{len}) ~= size(gu{len})) || size(Dl{len},1) ~= size(gl{len}, 1)
                        error('Inconsistency between Dl, Du, gl, gu');
                    end

                    if ~all(size(l0) == size(u0))
                        error('Inconsistency between initial lower bound and upper bound');
                    end

                    if gpuAvailable
                        obj.Dl = gpuArray(Dl);
                        obj.Du = gpuArray(Du);
                        obj.gl = gpuArray(gl);
                        obj.gu = gpuArray(gu);
                        obj.lb = gpuArray(lb);
                        obj.ub = gpuArray(ub);
                        obj.l0 = gpuArray(l0);
                        obj.u0 = gpuArray(u0);
                    else
                        obj.Dl = Dl;
                        obj.Du = Du;
                        obj.gl = gl;
                        obj.gu = gu;
                        obj.lb = lb;
                        obj.ub = ub;
                        obj.l0 = l0;
                        obj.u0 = u0;
                    end
                    
                    obj.dim = size(Dl{len}, 1);

                case 2
                    l0 = varargin{1};
                    u0 = varargin{2};
                    if ~all(size(l0) == size(u0))
                        error('Inconsistency between initial lower bound and upper bound');
                    end

                    nl = size(l0, 1);

                    if gpuAvailable
                        obj.Dl = {gpuArray(diag(ones(nl,1)))};
                        obj.Du = {gpuArray(diag(ones(nl,1)))};
                        obj.gl = {gpuArray(zeros(nl,1))};
                        obj.gu = {gpuArray(zeros(nl,1))};
                        obj.lb = {gpuArray(l0)};
                        obj.ub = {gpuArray(u0)};
                        obj.l0 = gpuArray(l0);
                        obj.u0 = gpuArray(u0);
                    else
                        obj.Dl = {diag(ones(nl,1))};
                        obj.Du = {diag(ones(nl,1))};
                        obj.gl = {zeros(nl,1)};
                        obj.gu = {zeros(nl,1)};
                        obj.lb = {l0};
                        obj.ub = {u0};
                        obj.l0 = l0;
                        obj.u0 = u0;
                    end
                    obj.dim = size(l0, 1);

                otherwise
                    error('Invalid number of input arguments')
            end
        end

        function SI = affineMap(varargin)
            switch nargin
                case 3
                    obj = varargin{1};
                    W = varargin{2};
                    b = varargin{3};
                case 2
                    obj = varargin{1};
                    W = varargin{2};
                    b = zeros(size(W,1),1);
            end

            [nW, mW] = size(W);
            [nb, mb] = size(b);

            if mb > 1
                error('bias vector must be one column');
            end
            
            if nW ~= nb && nb~=0
                error('Inconsistency between the affine mapping matrix and the bias vector');
            end

            if nb == 0
                b = zeros(nW, 1);
            end

            len = length(obj.Dl);
            if size(obj.Dl{len},1) ~= nW
                error('Inconsistency between the affine mapping matrix and dimension of the Symbolic Interval');
            end

            Dl = obj.Dl;
            Du = obj.Du;
            gl = obj.gl;
            gu = obj.gu;
            lb = obj.lb;
            ub = obj.ub;
            
            Dl{len+1} = W;
            Du{len+1} = W;
            gl{len+1} = b;
            gu{len+1} = b;
            [lb{len+1}, ub{len+1}] = getRanges(obj);

            SI = SymbolicInterval(Dl, Du, gl, gu, lb, ub, obj.l0, obj.u0);
        end
        
        function [lb, ub] = getRanges(obj)
            lb = lb_backSub(obj);
            ub = ub_backSub(obj);
        end

        function [Dl21, gl21, Du21, gu21] = composition(obj, Dl1, gl1, Du1, gu1, Dl2, gl2, Du2, gu2)
            Dl21 = max(0, Dl2)*Dl1 + min(0, Dl2)*Du1;
            gl21 = gl2 + max(0, Dl2)*gl1 + min(0, Dl2)*gu1;
        
            Du21 = max(0, Du2)*Du1 + min(0, Du2)*Dl1;
            gu21 = gu2 + max(0, Du2)*gu1 + min(0, Du2)*gl1;
        end

        % lower bound back-substitution
        function lb = lb_backSub(obj)
            n = length(obj.Dl);
            [nD, mD] = size(obj.Dl{n});
            Q = obj.Dl{n};
            vl = obj.gl{n};
            vu = zeros(nD, 1);

            for len = n-1:-1:1
                max_Q = max(0, Q);
                min_Q = min(Q, 0);

                vl = max_Q * obj.gl{len} + vl;
                vu = min_Q * obj.gu{len} + vu;

                Q = max_Q * obj.Dl{len} + min_Q * obj.Du{len};
            end
           
            max_Q = max(0, Q);
            min_Q = min(Q, 0);

            lb = max_Q * obj.l0 + vl + min_Q * obj.u0 + vu;
        end

        % upper bound back-substituion
        function ub = ub_backSub(obj)
            n = length(obj.Du);
            [nD, mD] = size(obj.Du{n});
            Q = obj.Du{n};
            vu = obj.gu{n};
            vl = zeros(nD, 1);
            
            for len = n-1:-1:1
                max_Q = max(0, Q);
                min_Q = min(Q, 0);

                vl = min_Q * obj.gl{len} + vl;
                vu = max_Q * obj.gu{len} + vu;
                
                Q = min_Q * obj.Dl{len} + max_Q * obj.Du{len};
            end
            
            max_Q = max(0, Q);
            min_Q = min(Q, 0);

            ub = min_Q * obj.l0 + vl + max_Q * obj.u0 + vu;
        end

%         % Minkowski Sum
%         function obj = Sum(obj, X)
%             if ~isa(X, 'SymbolicInterval')
%                 error('Input matrix is not a SymbolicInterval');
%             else
%                 if obj.dim ~= X.dim
%                     error('Input SymbolicInterval and current SymbolicInterval have different dimensions');
%                 end
%             end
% 
%             len1 = length(obj.Dl);
%             len2 = length(X.Dl);
% 
% 
%         end
        
        function SI = TanH(obj)
            [l, u] = obj.getRanges();
            
            len = length(obj.Dl);
            Dl = obj.Dl;
            Du = obj.Du;
            gl = obj.gl;
            gu = obj.gu;
            lb = obj.lb;
            ub = obj.ub;
            lb{len} = l;
            ub{len} = u;

            yl = tansig(l);
            yu = tansig(u);
            dyl = tansig('dn', l);
            dyu = tansig('dn', u);

            n = length(l);
            al = zeros(n, 1);
            au = zeros(n, 1);

            map1 = find(l >= 0);
            al(map1) = (yu(map1) - yl(map1)) ./ (u(map1) - l(map1));
            au(map1) = dyu(map1);

            map2 = find(u <= 0);
            al(map2) = dyl(map2);
            au(map2) = (yu(map2) - yl(map2)) ./ (u(map2) - l(map2));

            map3 = find(l<0 & u>0);
            lambda = min(dyl(map3), dyu(map3));
            al(map3) = lambda;
            au(map3) = lambda;         
            
            Dl_A = diag(al);
            Du_A = diag(au);
            gl_A = yl - al.*l;
            gu_A = yu - au.*u;

            [Dl{len}, gl{len}, Du{len}, gu{len}] = obj.composition(Dl{len}, gl{len}, Du{len}, gu{len}, Dl_A, gl_A, Du_A, gu_A);
            SI = SymbolicInterval(Dl, Du, gl, gu, lb, ub, obj.l0, obj.u0);
        end

        function SI = Sigmoid(obj)
            l = lb_backSub(obj);
            u = ub_backSub(obj);
            
            yl = logsig(l);
            yu = logsig(u);
            dyl = logsig('dn', l);
            dyu = logsig('dn', u);
            
            n = length(l);
            al = zeros(n, 1);
            au = zeros(n, 1);

            map1 = find(l >= 0);
            al(map1) = (yu(map1) - yl(map1)) ./ (u(map1) - l(map1));
            au(map1) = dyu(map1);

            map2 = find(u <= 0);
            al(map2) = dyl(map2);
            au(map2) = (yu(map2) - yl(map2)) ./ (u(map2) - l(map2));

            map3 = find(l<0 & u>0);
            lambda = min(dyl(map3), dyu(map3));
            al(map3) = lambda;
            au(map3) = lambda;

            len = length(obj.Dl);
            Dl = obj.Dl;
            Du = obj.Du;
            gl = obj.gl;
            gu = obj.gu;
            
            Dl_A = diag(al);
            Du_A = diag(au);
            gl_A = yl - al.*l;
            gu_A = yu - au.*u;

            [Dl{len}, gl{len}, Du{len}, gu{len}] = obj.composition(Dl{len}, gl{len}, Du{len}, gu{len}, Dl_A, gl_A, Du_A, gu_A);
            SI = SymbolicInterval(Dl, Du, gl, gu, obj.l0, obj.u0);
        end

        function SI = ReLU(obj)
            [l, u] = obj.getRanges();

            len = length(obj.Dl);
            Dl = obj.Dl;
            Du = obj.Du;
            gl = obj.gl;
            gu = obj.gu;
            lb = obj.lb;
            ub = obj.ub;
            
            n = length(l);
            Dl_A = zeros(n, 1);
            Du_A = zeros(n, 1);
            gl_A = zeros(n, 1);
            gu_A = zeros(n, 1);
            l_A = zeros(n, 1);
            u_A = zeros(n, 1);
                
            map1 = find(l >= 0);
            Dl_A(map1) = 1;
            Du_A(map1) = 1;
            l_A(map1) = l(map1);
            u_A(map1) = u(map1);

            map2 = find(l < 0 & u > 0);
            Du_A(map2) = u(map2) ./ (u(map2) - l(map2));
            gu_A(map2) = -u(map2) .* l(map2) ./ (u(map2) - l(map2));
            l_A(map2) = l(map2);
            u_A(map2) = u(map2);

            map3 = map2(find(u(map2) > -l(map2)));
            Dl_A(map3) = 1;

            Dl_A = diag(Dl_A);
            Du_A = diag(Du_A);

            lb{len} = l_A;
            ub{len} = u_A;

            [Dl{len}, gl{len}, Du{len}, gu{len}] = obj.composition(Dl{len}, gl{len}, Du{len}, gu{len}, Dl_A, gl_A, Du_A, gu_A);
            SI = SymbolicInterval(Dl, Du, gl, gu, lb, ub, obj.l0, obj.u0);
        end

        function square(obj)
            l = lb_backSub(obj);
            u = ub_backSub(obj);

            n = length(l);
            Dl_A = zeros(n, 1);
            gl_A = zeros(n, 1);

            map1 = find(l >= 0);
            Dl_A(map1) = 2*l;
            gl_A(map1) = -l*l;

            map2 = find(u <= 0);
            Dl_A(map2) = 2*u;
            gl_A(map2) = -u*u;

            Dl_A = diag(Dl_A);
            Du_A = diag(u + l);
            gu_A = -u*l;

            len = length(obj.Dl);
            Dl = obj.Dl;
            Du = obj.Du;
            gl = obj.gl;
            gu = obj.gu;
            
            [Dl{len}, gl{len}, Du{len}, gu{len}] = obj.composition(Dl{len}, gl{len}, Du{len}, gu{len}, Dl_A, gl_A, Du_A, gu_A);
            SI = SymbolicInterval(Dl, Du, gl, gu, obj.l0, obj.u0);
        end


%         function log(obj)
% 
%         end


        function P = toPolyhedron(obj)
            [lb, ub] = obj.getRanges();
            P = Polyhedron('lb', lb, 'ub', ub);
        end

        function S = toStar(obj)
            len = length(obj.Dl);
            
            pred_lb = obj.lb{1};
            pred_ub = obj.ub{1};
            Cu = obj.Dl{1};
            Cl = obj.Du{1};
            
            dl = obj.l0; % obj.gl{1};
            du = obj.u0; % obj.gu{1};

            for i = 2:len
                dim = length(obj.gl{i});
                
%                 if i == 2
%                     Cl = [[Cl, zeros(dim, dim)]; [obj.Dl{i}, diag(ones(dim,1))]];
%                     Cu = [[Cu, zeros(dim, dim)]; [-obj.Du{i}, diag(ones(dim,1))]];
% 
%                     dl = [dl; obj.gl{i}];
%                     du = [du; obj.gu{i}];
%                 else
                    [n, m] = size(obj.Dl{i});

%                     Cl = [[Cl, zeros(dim, dim)]; [ obj.Dl{i}, diag(ones(n,1))] ];
%                     Cu = [[Cu, zeros(dim, dim)]; [-obj.Du{i}, diag(ones(n,1))] ];


                    Cl = [[Cl, zeros(dim, dim)]; [zeros(n,m), -obj.Dl{i}, diag(ones(dim,1))]];
                    Cu = [[Cu, zeros(dim, dim)]; [zeros(n,m), -obj.Du{i}, diag(ones(dim,1))]];

                    

% 
%                     Cl = blkdiag(Cl, [obj.Dl{i}, diag(ones(n,1))]);
%                     Cu = blkdiag(Cu, [-obj.Du{i}, diag(ones(n,1))]);

                    dl = [dl; obj.gl{i}];
                    du = [du; obj.gu{i}];                    
%                 end
                pred_lb = [pred_lb; obj.lb{i}];
                pred_ub = [pred_ub; obj.ub{i}];
            end
            
            C = [Cu; -Cl]
            d = [du; -dl]
            dim = obj.dim;
            
            [n, m] = size(C);

            pred_lb
            pred_ub
            dim


            M = [zeros(dim, m-dim) diag(ones(dim, 1))];
            V = [zeros(dim, 1), M]
            S = Star(V, C, d, pred_lb, pred_ub);
        end
        
        function plot(varargin)
            switch nargin
                case 2
                    obj = varargin{1};
                    color = varargin{2};
                case 1
                    obj = varargin{1};
                    color = 'red';
                otherwise
                    error('Invalid number of input arguments')
            end
            P = obj.toPolyhedron;
            P.plot('color', color);
        end

    end
end