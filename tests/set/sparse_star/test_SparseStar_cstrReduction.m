close all; clear; clc; 
format long;

dim = 2;
n_sample = 500;

W1 = rand(dim, dim);
b1 = rand(dim, 1);

W2 = rand(dim, dim);
b2 = rand(dim, 1);

W3 = rand(dim, dim);
b3 = rand(dim, 1);


S = SparseStar.rand(dim);

% samples
s = S.sample(n_sample);
s1a= W1*s + b1;
s1t = TanSig.evaluate(s1a);
s2a = W2*s1t + b2;
s2t = TanSig.evaluate(s2a);
s3a = W3*s2t + b3;
s3t = TanSig.evaluate(s3a);

figure('Name', 'ctrReduction');
S = SparseStar.rand(dim);
s = S.sample(n_sample);
nexttile;
S.plot;
sample_plot(s, n_sample);
title('Input SparseStar');

% AffineMap 1
S1a = S.affineMap(W1, b1);
nexttile;
S1a.plot;
sample_plot(s1a, n_sample);
title("AffineMap 1");

% TanSig 1
S1t = TanSig.reach(S1a, 'approx-sparse-star');
nexttile;
S1t.plot;
sample_plot(s1t, n_sample);
title("TanSig 1");

% AffineMap 2
S2a = S1t.affineMap(W2, b2);
nexttile;
S2a.plot;
sample_plot(s2a, n_sample);
title('AffineMap 2');

% TanSig 2
S2t = TanSig.reach(S2a, 'approx-sparse-star');
nexttile;
S2t.plot;
sample_plot(s2t, n_sample);
title("TanSig 2");

% AffineMap 3
S3a = S2t.affineMap(W3, b3);
nexttile;
S3a.plot;
sample_plot(s3a, n_sample);
title('AffineMap 3');

% TanSig 3
S3t = TanSig.reach(S3a, 'approx-sparse-star');
nexttile;
S3t.plot;
sample_plot(s3t, n_sample);
title("TanSig 3");

%% depthReduction 3
S3td = S3t.depthReduction(3);
nexttile
S3td.plot;
sample_plot(s3t, n_sample);
title('depthReduction(3)');

%% cstrReduction 3
S3td = S3t.cstrReduction(3);
nexttile
S3td.plot;
sample_plot(s3t, n_sample);
title('cstrReduction(3)');



function sample_plot(s, n_sample)
    hold on;
    for i = 1:n_sample
        plot(s(1, i), s(2, i), '*k');
    end
end


