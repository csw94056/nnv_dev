clc; clear; close all;

% https://www.mathworks.com/help/matlab/ref/full.html

% If X is an m-by-n matrix with nz nonzero elements, then full(X) requires 
% space to store m*n elements. On the other hand, sparse(X) requires space 
% to store nz elements and (nz+n+1) integers.

% The density of a matrix (nnz(X)/numel(X)) determines whether it is more 
% efficient to store the matrix as sparse or full. The exact crossover point depends on the matrix class, as well as the platform. For example, in 32-bit MATLAB®, a double sparse matrix with less than about 2/3 density requires less space than the same matrix in full storage. In 64-bit MATLAB, however, double matrices with fewer than half of their elements nonzero are more efficient to store as sparse matrices.

% R = sprandn(m,n,density) creates a random m-by-n sparse matrix with approximately density*m*n normally distributed nonzero entries for density in the interval [0,1].
A = sprand(1000,1000,0.005);
B = sprand(1000,1000,0.005);
Af = full(A);
Bf = full(B);

Cf = rand(1000,1000);
Df = rand(1000,1000);
C = sparse(Cf);
D = sparse(Df);

E = sprand(1000,1000,0.70); % 69~70
F = sprand(1000,1000,0.55);
Ef = full(E);
Ff = full(F);


whos

%   Name         Size                 Bytes  Class     Attributes
% 
%   A         1000x1000               87800  double    sparse    
%   Af        1000x1000             8000000  double              
%   B         1000x1000               87800  double    sparse    
%   Bf        1000x1000             8000000  double              
%   C         1000x1000            16008008  double    sparse    
%   Cf        1000x1000             8000000  double              
%   D         1000x1000            16008008  double    sparse    
%   Df        1000x1000             8000000  double              
%   E         1000x1000             8066696  double    sparse    
%   Ef        1000x1000             8000000  double              
%   F         1000x1000             6779768  double    sparse    
%   Ff        1000x1000             8000000  double              



timeit(@() full(A))
timeit(@() full(B))
timeit(@() sparse(Cf))
timeit(@() sparse(Df))


timeit(@() Af*Bf)

timeit(@() A*B)

C = Af*B;
timeit(@() Af*B)

whos C


