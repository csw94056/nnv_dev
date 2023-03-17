close all; clear; clc;
dim = 2;
lb = -rand(dim,1);
ub = rand(dim,1);

S = Star(lb, ub);
S.Z = [];
SS = SparseStar(lb, ub);

W = 1 - 2*rand(dim);
b = 1 - 2*rand(dim,1);

Sa = S.affineMap(W, b);
SSa = SS.affineMap(W, b);


TSa = LogSig.reach(Sa, 'approx-star');
TSSa = LogSig.reach(SSa, 'approx-sparse-star');


figure('Name', 'test LogSig of SparseStar');
nexttile;
plot(S, 'r');
plot(SS, 'c');
title('input');


nexttile;
plot(Sa, 'r');
plot(SSa, 'c');
title('after affineMap');


nexttile;
plot(TSa, 'r');
plot(TSSa, 'c');
title('after LogSig');

whos S SS Sa SSa TSa TSSa