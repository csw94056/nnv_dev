close all; clear; clc; 

dim = 100;
lb = -ones(dim, 1);
ub = ones(dim, 1);
disp = (dim <= 3);

SS = SparseStar(lb, ub);
S = Star(lb, ub);

tSS = timeit(@() SparseStar(lb, ub));
tS = timeit(@() Star(lb, ub));

fprintf('dimension of sets: %d', dim);
fprintf('time for SparseStar(lb, ub): %f\n', tSS);
fprintf('time for Star(lb, ub): %f\n', tS);

if disp
    figure;
    nexttile;
    plot(S, 'c');
    title('Star input')
    nexttile;
    plot(SS, 'r');
    title('SparseStar input')
end

W = rand(dim, dim);
b = rand(dim, 1);

SSa = SS.affineMap(W, b);
Sa = S.affineMap(W, b);

tSSa = timeit(@() SS.affineMap(W, b));
tSa = timeit(@() S.affineMap(W, b));

fprintf('time for SparseStar.affineMap(W, b): %f\n', tSSa);
fprintf('time for Star.affineMap(W, b): %f\n', tSa);

if disp
    nexttile;
    plot(Sa, 'c');
    title('Star affineMap')
    nexttile;
    plot(SSa, 'r');
    title('SparseStar affineMap');
end

whos S Sa SS SSa