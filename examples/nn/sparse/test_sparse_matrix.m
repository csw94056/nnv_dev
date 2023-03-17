clc; clear; close all;

% https://www.mathworks.com/help/matlab/ref/full.html

% If X is an m-by-n matrix with nz nonzero elements, then full(X) requires 
% space to store m*n elements. On the other hand, sparse(X) requires space 
% to store nz elements and (nz+n+1) integers.

% The density of a matrix (nnz(X)/numel(X)) determines whether it is more 
% efficient to store the matrix as sparse or full. The exact crossover point depends on the matrix class, as well as the platform. For example, in 32-bit MATLAB®, a double sparse matrix with less than about 2/3 density requires less space than the same matrix in full storage. In 64-bit MATLAB, however, double matrices with fewer than half of their elements nonzero are more efficient to store as sparse matrices.

% R = sprandn(m,n,density) creates a random m-by-n sparse matrix with 
% approximately density*m*n normally distributed nonzero entries for 
% density in the interval [0,1].
z = 10;
% A = sprand(z,z,0.005);
% Af = full(A);
% 
% Bf = rand(z,z);
% B = sparse(Bf);

E = sprand(z,z,0.69299); % 69~70
% F = sprand(z,z,0.55);
Ef = full(E);
% Ff = full(F);


whos

%   Name         Size                 Bytes  Class     Attributes
% 
%   A         1000x1000               87768  double    sparse    
%   Af        1000x1000             8000000  double              
%   B         1000x1000            16008008  double    sparse    
%   Bf        1000x1000             8000000  double              
%   E         1000x1000             8058504  double    sparse    
%   Ef        1000x1000             8000000  double              
%   F         1000x1000             6777480  double    sparse    
%   Ff        1000x1000             8000000  double      

% timeit(@() issparse(E))
% timeit(@() sparse(E))
% timeit(@() zeros(1, 10000))
% timeit(@() sparse(zeros(1, 10000)))

