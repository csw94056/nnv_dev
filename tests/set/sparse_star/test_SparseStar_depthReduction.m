close all; clear; clc; 
format long;

dim = 2;
n_sample = 500;

%% input set
z = 1;
figure('Name', 'No depthReduction');
S = SparseStar.rand(dim);
s = S.sample(n_sample);
nexttile;
S.plot;
sample_plot(s, n_sample);
title('Input SparseStar');

%%  affineMap 1
W1 = rand(dim, dim);
b1 = rand(dim, 1);

s1a= W1*s + b1;
S1a{z} = S.affineMap(W1, b1);
nexttile;
S1a{z}.plot;
sample_plot(s1a, n_sample);
title("AffineMap 1");

%% TanSig 1
s1t = TanSig.evaluate(s1a);
S1t{z} = TanSig.reach(S1a{z}, 'approx-sparse-star');
nexttile;
S1t{z}.plot;
sample_plot(s1t, n_sample);
title("TanSig 1");

%% affineMap 2
W2 = rand(dim, dim);
b2 = rand(dim, 1);

s2a = W2*s1t + b2;
S2a{z} = S1t{z}.affineMap(W2, b2);
nexttile;
S2a{z}.plot;
sample_plot(s2a, n_sample);
title('AffineMap 2');

%% TanSig 2
s2t = TanSig.evaluate(s2a);
S2t{z} = TanSig.reach(S2a{z}, 'approx-sparse-star');
nexttile;
S2t{z}.plot;
sample_plot(s2t, n_sample);
title("TanSig 2");

%% affineMap 3
W3 = rand(dim, dim);
b3 = rand(dim, 1);

s3a = W3*s2t + b3;
S3a{z} = S2t{z}.affineMap(W3, b3);
nexttile;
S3a{z}.plot;
sample_plot(s3a, n_sample);
title('AffineMap 3');

%% TanSig 3
s3t = TanSig.evaluate(s3a);
S3t{z} = TanSig.reach(S3a{z}, 'approx-sparse-star');
nexttile;
S3t{z}.plot;
sample_plot(s3t, n_sample);
title("TanSig 3");

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% input set
z = 2;
figure('Name', 'depthReduction(1)');
nexttile;
S.plot;
sample_plot(s, n_sample);
title('Input SparseStar');

%%  affineMap 1
S1a{z} = S.affineMap(W1, b1);
nexttile;
S1a{z}.plot;
sample_plot(s1a, n_sample);
title("AffineMap 1");

%% TanSig 1
S1t{z} = TanSig.reach(S1a{z}, 'approx-sparse-star');
nexttile;
S1t{z}.plot;
sample_plot(s1t, n_sample);
title("TanSig 1");

%% depthReduction 1
S1td{z} = S1t{z}.depthReduction(1);
nexttile
S1td{z}.plot;
sample_plot(s1t, n_sample);
title('depthReduction(1)');

S1t{z} = S1td{z};

%% affineMap 2
S2a{z} = S1t{z}.affineMap(W2, b2);
nexttile;
S2a{z}.plot;
sample_plot(s2a, n_sample);
title('AffineMap 2');

%% TanSig 2
S2t{z} = TanSig.reach(S2a{z}, 'approx-sparse-star');
nexttile;
S2t{z}.plot;
sample_plot(s2t, n_sample);
title("TanSig 2");

%% affineMap 3
S3a{z} = S2t{z}.affineMap(W3, b3);
nexttile;
S3a{z}.plot;
sample_plot(s3a, n_sample);
title('AffineMap 3');

%% TanSig 3
S3t{z} = TanSig.reach(S3a{z}, 'approx-sparse-star');
nexttile;
S3t{z}.plot;
sample_plot(s3t, n_sample);
title("TanSig 3");

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% input set
z = 3;
figure('Name', 'depthReduction(2)');
nexttile;
S.plot;
sample_plot(s, n_sample);
title('Input SparseStar');

%%  affineMap 1
S1a{z} = S.affineMap(W1, b1);
nexttile;
S1a{z}.plot;
sample_plot(s1a, n_sample);
title("AffineMap 1");

%% TanSig 1
S1t{z} = TanSig.reach(S1a{z}, 'approx-sparse-star');
nexttile;
S1t{z}.plot;
sample_plot(s1t, n_sample);
title("TanSig 1");

%% affineMap 2
S2a{z} = S1t{z}.affineMap(W2, b2);
nexttile;
S2a{z}.plot;
sample_plot(s2a, n_sample);
title('AffineMap 2');

%% TanSig 2
S2t{z} = TanSig.reach(S2a{z}, 'approx-sparse-star');
nexttile;
S2t{z}.plot;
sample_plot(s2t, n_sample);
title("TanSig 2");

%% depthReduction 2
S2td{z} = S2t{z}.depthReduction(2);
nexttile
S2td{z}.plot;
sample_plot(s2t, n_sample);
title('depthReduction(2)');

S2t{z} = S2td{z};

%% affineMap 3
S3a{z} = S2t{z}.affineMap(W3, b3);
nexttile;
S3a{z}.plot;
sample_plot(s3a, n_sample);
title('AffineMap 3');

%% TanSig 3
S3t{z} = TanSig.reach(S3a{z}, 'approx-sparse-star');
nexttile;
S3t{z}.plot;
sample_plot(s3t, n_sample);
title("TanSig 3");

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% input set
z = 4;

figure('Name', 'depthReduction(3)');
nexttile;
S.plot;
sample_plot(s, n_sample);
title('Input SparseStar');

%%  affineMap 1
S1a{z} = S.affineMap(W1, b1);
nexttile;
S1a{z}.plot;
sample_plot(s1a, n_sample);
title("AffineMap 1");

%% TanSig 1
S1t{z} = TanSig.reach(S1a{z}, 'approx-sparse-star');
nexttile;
S1t{z}.plot;
sample_plot(s1t, n_sample);
title("TanSig 1");

%% affineMap 2
S2a{z} = S1t{z}.affineMap(W2, b2);
nexttile;
S2a{z}.plot;
sample_plot(s2a, n_sample);
title('AffineMap 2');

%% TanSig 2
S2t{z} = TanSig.reach(S2a{z}, 'approx-sparse-star');
nexttile;
S2t{z}.plot;
sample_plot(s2t, n_sample);
title("TanSig 2");

%% affineMap 3
S3a{z} = S2t{z}.affineMap(W3, b3);
nexttile;
S3a{z}.plot;
sample_plot(s3a, n_sample);
title('AffineMap 3');

%% TanSig 3
S3t{z} = TanSig.reach(S3a{z}, 'approx-sparse-star');
nexttile;
S3t{z}.plot;
sample_plot(s3t, n_sample);
title("TanSig 3");

%% depthReduction 3
S3td{z} = S3t{z}.depthReduction(3);
nexttile
S3td{z}.plot;
sample_plot(s3t, n_sample);
title('depthReduction(3)');
S3t{z} = S3td{z};


%%
figure('Name', 'outputs');
for i = 1:4
    nexttile
    S3t{i}.plot;
    sample_plot(s3t, n_sample);
    s = sprintf('depthReduction(%d)', i-1);
    title(s);
end


function sample_plot(s, n_sample)
    hold on;
    for i = 1:n_sample
        plot(s(1, i), s(2, i), '*k');
    end
end


% figure('Name', 'outputs');
% for i = 1:4
%     nexttile
%     S3t{i}.plot;
%     hold on;
%     for j = 1:n_sample
%         plot(s3t(1, j), s3t(2, j), '*k');
%     end
%     s = sprintf('depthReduction(%d)', i-1);
%     title(s);
% end