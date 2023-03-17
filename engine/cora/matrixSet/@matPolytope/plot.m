function plot(varargin)
% plot - Plots 2-dimensional projection of a matrix polytope
%
% Syntax:  
%    plot(obj,dimensions)
%
% Inputs:
%    obj - matrix polytope
%    dimensions - dimensions that should be projected (optional) 
%    linespec - plot style (optional) 
%
% Outputs:
%    none
%
% Example: 
%
% Other m-files required: none
% Subfunctions: none
% MAT-files required: none
%
% See also: none

% Author:       Matthias Althoff
% Written:      22-June-2010
% Last update:  ---
% Last revision:---

%------------- BEGIN CODE --------------

%plot vertices
if nargin==1
    matP=varargin{1};
    dims=[1 2];
    style='b-';
elseif nargin==2
    matP=varargin{1};
    dims=varargin{2};
    style='b-';
elseif nargin==3
    matP=varargin{1};
    dims=varargin{2};
    style=varargin{3};
end

%convert vertices
for i=1:matP.verts
    vec=mat2vec(matP.vertex{i});
    V(i,:)=vec(dims);
end
%keep only unique vertices
V = unique(V, 'rows');

%obtain vertices object
V=vertices(V');

%plot vertices
plot(V,style);

%------------- END OF CODE --------------
