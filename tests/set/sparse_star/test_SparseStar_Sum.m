% close all; clear; clc;
format long;

option = [];
relaxFactor = 0;
dis_opt = [];
lp_solver = 'estimate';

dim = 2;
lb = -rand(dim,1);
ub = rand(dim,1);

S = Star(lb, ub);
S.Z = [];
SS = SparseStar(lb, ub);

W1 = 1 - 2*rand(dim);
b1 = 1 - 2*rand(dim,1);

S1 = S.affineMap(W1, b1);
SS1 = SS.affineMap(W1, b1);


disp('after AffineMap');
tS1 = table;
[tS1.lb, tS1.ub] = S1.estimateBounds()
tSS1 = table;
[tSS1.lb, tSS1.ub] = SS1.estimateRanges()


TS1 = TanSig.reach(S1, 'approx-star', option, relaxFactor, dis_opt, lp_solver);
TSS1 = TanSig.reach(SS1, 'approx-sparse-star', option, relaxFactor, dis_opt, lp_solver);

disp('after TanSig');
tTS1 = table;
[tTS1.lb, tTS1.ub] = TS1.estimateBounds()
tTSS1 = table;
[tTSS1.lb, tTSS1.ub] = TSS1.estimateRanges()


timeit(@() TanSig.reach(S1, 'approx-star'))
timeit(@() TanSig.reach(SS1, 'approx-sparse-star'))

W2 = 1 - 2*rand(dim);
b2 = 1 - 2*rand(dim,1);

S2 = TS1.affineMap(W2,b2);
SS2 = TSS1.affineMap(W2,b2);


disp('after AffineMap');
tS2 = table;
[tS2.lb, tS2.ub] = S2.estimateBounds()
tSS2 = table;
[tSS2.lb, tSS2.ub] = SS2.estimateRanges()

TS2 = TanSig.reach(S2, 'approx-star', option, relaxFactor, dis_opt, lp_solver);
TSS2 = TanSig.reach(SS2, 'approx-sparse-star', option, relaxFactor, dis_opt, lp_solver);

disp('after TanSig');
tTS2 = table;
[tTS2.lb, tTS2.ub] = TS2.estimateBounds()
tTSS2 = table;
[tTSS2.lb, tTSS2.ub] = TSS2.estimateRanges()

MS = TS2.Sum(TS1);
MSS = TSS2.Sum(TSS1);

disp('after MinkSum');
tMS = table;
[tMS.lb, tMS.ub] = MS.estimateBounds()
tMSS = table;
[tMSS.lb, tMSS.ub] = MSS.estimateRanges()

% figure('Name', 'test TanSig of SparseStar');
% nexttile;
% plot(S, 'r');
% plot(SS, 'c');
% title('input');
% 
% 
% nexttile;
% plot(Sa, 'r');
% plot(SSa, 'c');
% title('after affineMap 1');
% 
% 
% nexttile;
% plot(S1, 'r');
% plot(SS1, 'c');
% title('after TanSig');
% 
% nexttile;
% plot(S2, 'r');
% plot(SS2, 'c');
% title('after affineMap 2');
% 
% 
nexttile;
plot(MS, 'r');
plot(MSS, 'c');
title('after MinkowskiSum');

whos S SS Sa SSa S1 SS1 S2 SS2 MS MSS