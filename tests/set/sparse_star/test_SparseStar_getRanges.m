close all; clear; clc; 
format long;

lb = [-1; -1];
ub = [1; 1;];

SS = SparseStar(lb, ub);
SS.lps = 'glpk';
S = Star(lb, ub);


W1 = rand(2, 2);
b1 = rand(2, 1);


SSa = SS.affineMap(W1, b1);
Sa = S.affineMap(W1, b1);


W2 = rand(2, 2);
b2 = rand(2, 1);

SSa2 = SS.affineMap(W2, b2);
Sa2 = S.affineMap(W2, b2);

Sa = TanSig.reach(Sa, 'approx-star');
SSa = TanSig.reach(SSa, 'approx-sparse-star');

Sa2 = TanSig.reach(Sa2, 'approx-star');
SSa2 = TanSig.reach(SSa2, 'approx-sparse-star');

% Sa = Sa.Sum(Sa2);
% SSa = SSa.Sum(SSa2);



%% Star getRanges()
tSa = table;
tSa.lb = Sa.getMins(1:Sa.dim);
tSa.ub = Sa.getMaxs(1:Sa.dim);
tSa


%% SparseStar getRanges()
Ss1 = table;
% uses getMin and getMax to achieve ranges with glpk solver
[Ss1.lb, Ss1.ub] = SSa.getRanges('glpk', 'single');
disp('SparseStar: glpk, single');
Ss1

Ss2 = table;
% uses getMins and getMax to achieve ranges with glpk solver
[Ss2.lb, Ss2.ub] = SSa.getRanges('glpk', 'parallel');
disp('SparseStar: glpk, parallel');
Ss2

Ss3 = table;
% uses getMin and getMax to achieve ranges with linprog solver
[Ss3.lb, Ss3.ub] = SSa.getRanges('linprog', 'single');
disp('SparseStar: linprog, single');
Ss3

Ss4 = table;
% uses getMins and getMax to achieve ranges with linprog solver
[Ss4.lb, Ss4.ub] = SSa.getRanges('linprog', 'parallel');
disp('SparseStar: linprog, parallel');
Ss4


Ss5 = table;
% uses getMins and getMax to achieve ranges with linprog solver
[Ss5.lb, Ss5.ub] = SSa.getRanges('estimate');
disp('SparseStar: estimage');
Ss5

TS = SSa.toStar();
disp('SparseStar to Star and getRanges');
tTS = table;
tTS.lb = TS.getMins(1:TS.dim);
tTS.ub = TS.getMaxs(1:TS.dim);
tTS
