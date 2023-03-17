clc;
clear;
close all;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('Name', 'xl_i < 0 and xu_i > 0; yl_i < 0 and yu_i > 0');
title('$xl_i < 0$ and $xu_i > 0;$ $yl_i < 0$ and $yu_i > 0$', 'interpreter', 'latex');
% plot x,y,z 3D surface of sigmoid X tanh activation function
view(3);
hold on
% xl_i < 0 and xu_i > 0
% yl_i < 0 and yu_i > 0
% predicate bounds of each neuron
xl = -1.8;
xu = 1.4;
yl = -1.3;
yu = 1.9;

a = 2.0;
% a = 0.0;
surf_x = linspace(xl - a, xu + a, 60);
surf_y = linspace(yl - a, yu + a, 60);
surf_z = logsig(surf_x') * tansig(surf_y);
% surf_z = logsig(x) * y;
surf(surf_x, surf_y, surf_z);
xlabel('x'), ylabel('y'), zlabel('z');

h = gca;
plot3(h.XLim, [0 0], [0 0], 'r', 'LineWidth',2);
plot3([0 0], h.YLim, [0 0], 'r', 'LineWidth',2);
plot3([0 0], [0 0], h.ZLim, 'r', 'LineWidth',2);

pyl = yline(yl, ':c', 'LineWidth', 2);
pyu = yline(yu, ':c', 'LineWidth', 2);
pxl = xline(xl, ':c', 'LineWidth', 2);
pxu = xline(xu, ':c', 'LineWidth', 2);


% plot sigmoid X tanh activation line
x = xl:0.01:xu;
y = yl:0.01:yu;
z = logsig(x') * tansig(y);
% pAFline = plot3(x,y,z,'b','LineWidth',4);

sxl = logsig(xl);
sxu = logsig(xu);
tyl = tansig(yl);
tyu = tansig(yu);

dsxl = logsig('dn', xl);
dsxu = logsig('dn', xu);
dtyl = tansig('dn', yl);
dtyu = tansig('dn', yu);


% Four edge points
pDerPoint = plot3(xl,yl,sxl*tyl,'*y','LineWidth',4);
pDerPoint = plot3(xu,yu,sxu*tyu,'*m','LineWidth',4);
pDerPoint = plot3(xl,yu,sxu*tyl,'*c','LineWidth',4);
pDerPoint = plot3(xu,yl,sxl*tyu,'*g','LineWidth',4);

x = linspace(xl, xu, 60);
y = linspace(yl, yu, 60);
z = logsig(x') * tansig(y);
h = surf(x,y,z);

% k = ones(size(x))' * tansig(x);
% h = surf(x, zeros(size(x)), k);
% 
% k = ones(size(x))' * logsig(x);
% h = surf(x, zeros(size(x)), k);

% set(h,'FaceColor', 'k', 'FaceAlpha',1.0,'FaceLighting','gouraud','EdgeColor','none')
% h = surf(x,yl*ones(length(x)),z);
% set(h,'FaceColor', 'b', 'FaceAlpha',1.0,'FaceLighting','gouraud','EdgeColor','none')
% h = surf(x,yu*ones(length(x)),z);
% set(h,'FaceColor', 'b', 'FaceAlpha',1.0,'FaceLighting','gouraud','EdgeColor','none')
% h = surf(xl*ones(length(y)),y,z);
% set(h,'FaceColor', 'm', 'FaceAlpha',1.0,'FaceLighting','gouraud','EdgeColor','none')
% h = surf(xu*ones(length(y)),y,z);
% set(h,'FaceColor', 'm', 'FaceAlpha',1.0,'FaceLighting','gouraud','EdgeColor','none')


d_smin = min(dsxl,dsxu);
d_tmin = min(dtyl,dtyu);
% 
% % plot constrained are
% A = [1          0           0;
%      -1         0           0;
%      0          1           0;
%      0          -1          0;
%      -d_tmin    0           1;
%      d_tmin     0           -1;
%      0          -d_smin     1;
%      0          -d_smin      -1];
% 
% b = [xu;
%      -xl;
%      yu;
%      -yl;
%      sxu*tyu-d_tmin*xu;
%      -sxu*tyl+d_tmin*xl;
%      sxu*tyu-d_smin*yu;
%      -sxu*tyl-d_smin*yu];
%      
% P = Polyhedron(A,b);
% plot(P, 'color', 'pink');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
zl = logsigXtansig(xl, yl);
zu = logsigXtansig(xu, yu);

dzl = derivative_logsigXtansig(xl, yl);
dzu = derivative_logsigXtansig(xu, yu);


dz_min = min(dzl, dzu);

% plot constrained are
A = [1          0           0;
     -1         0           0;
     0          1           0;
     0          -1          0;
     dz_min    0           1;
     -dz_min     0           -1;
     0          dz_min     1;
     0          dz_min     -1];

b = [xu;
     -xl;
     yu;
     -yl;
     sxu*tyu + dz_min*xu;
     -sxu*tyl - dz_min*xl;
     sxu*tyu + dz_min*yu;
     -sxu*tyl + dz_min*yu];
     
P = Polyhedron(A,b);
plot(P, 'color', 'pink');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('Name', 'xl_i \geq 0; yl_i \geq 0');
title('$xl_i \geq 0;$ $yl_i \geq 0$', 'interpreter', 'latex');
% plot x,y,z 3D surface of sigmoid X tanh activation function
view(3);
hold on;
% xl_i >= 0
% yl_i >= 0
% predicate bounds of each neuron
xl = 0.3;
xu = 1.5;
yl = 0.5; %-1.3;
yu = 1.7; %1.9;

a = 2.0;
% a = 0.0;
surf_x = linspace(xl - a, xu + a, 60);
surf_y = linspace(yl - a, yu + a, 60);
surf_z = logsig(surf_x') * tansig(surf_y);
% surf_z = logsig(x) * y;
surf(surf_x, surf_y, surf_z);
xlabel('x'), ylabel('y'), zlabel('z');



h = gca;
plot3(h.XLim, [0 0], [0 0], 'r', 'LineWidth',2);
plot3([0 0], h.YLim, [0 0], 'r', 'LineWidth',2);
plot3([0 0], [0 0], h.ZLim, 'r', 'LineWidth',2);


pyl = yline(yl, ':c', 'LineWidth', 2);
pyu = yline(yu, ':c', 'LineWidth', 2);
pxl = xline(xl, ':c', 'LineWidth', 2);
pxu = xline(xu, ':c', 'LineWidth', 2);


% plot sigmoid X tanh activation line
x = xl:0.01:xu;
y = yl:0.01:yu;
z = logsig(x') * tansig(y);
% pAFline = plot3(x,y,z,'b','LineWidth',4);

sxl = logsig(xl);
sxu = logsig(xu);
tyl = tansig(yl);
tyu = tansig(yu);

dsxl = logsig('dn', xl);
dsxu = logsig('dn', xu);
dtyl = tansig('dn', yl);
dtyu = tansig('dn', yu);

% Four edge points
pDerPoint = plot3(xl,yl,sxl*tyl,'*y','LineWidth',4);
pDerPoint = plot3(xu,yu,sxu*tyu,'*m','LineWidth',4);
pDerPoint = plot3(xl,yu,sxu*tyl,'*c','LineWidth',4);
pDerPoint = plot3(xu,yl,sxl*tyu,'*g','LineWidth',4);

x = linspace(xl, xu, 60);
y = linspace(yl, yu, 60);
z = logsig(x') * tansig(y);
h = surf(x,y,z);

% k = ones(size(x))' * tansig(x);
% h = surf(x, zeros(size(x)), k);
% 
% k = ones(size(x))' * logsig(x);
% h = surf(x, zeros(size(x)), k);

% set(h,'FaceColor', 'k', 'FaceAlpha',1.0,'FaceLighting','gouraud','EdgeColor','none')
% h = surf(x,yl*ones(length(x)),z);
% set(h,'FaceColor', 'b', 'FaceAlpha',1.0,'FaceLighting','gouraud','EdgeColor','none')
% h = surf(x,yu*ones(length(x)),z);
% set(h,'FaceColor', 'b', 'FaceAlpha',1.0,'FaceLighting','gouraud','EdgeColor','none')
% h = surf(xl*ones(length(y)),y,z);
% set(h,'FaceColor', 'm', 'FaceAlpha',1.0,'FaceLighting','gouraud','EdgeColor','none')
% h = surf(xu*ones(length(y)),y,z);
% set(h,'FaceColor', 'm', 'FaceAlpha',1.0,'FaceLighting','gouraud','EdgeColor','none')


d_smin = min(dsxl,dsxu);
d_tmin = min(dtyl,dtyu);

% % plot constrained are
% A = [1          0           0;
%      -1         0           0;
%      0          1           0;
%      0          -1          0;
%      -d_tmin    0           1;
%      d_tmin     0           -1;
%      0          -d_smin     1;
%      0          d_smin      -1];
% 
% b = [xu;
%      -xl;
%      yu;
%      -yl;
%      sxu*tyu-d_tmin*xu;
%      -sxl*tyl+d_tmin*xl;
%      sxu*tyu-d_smin*yu;
%      -sxu*tyl+d_smin*yu];
%      
% P = Polyhedron(A,b);
% plot(P, 'color', 'pink');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
zl = logsigXtansig(xl, yl);
zu = logsigXtansig(xu, yu);

dzl = derivative_logsigXtansig(xl, yl);
dzlu = derivative_logsigXtansig(xl, yu);
dzul = derivative_logsigXtansig(xu, yl);
dzu = derivative_logsigXtansig(xu, yu);

dz_min = min([dzl, dsxl, dtyl, dzu, dsxu, dtyu, dzlu, dzul]);  %dtyu
lambda = (sxl*tyu - sxl*tyl)/(xu-xl);
beta = (sxu*tyl-sxl*tyl)/(yu-yl);
% plot constrained are
A = [1          0           0;
     -1         0           0;
     0          1           0;
     0          -1          0;
     -dz_min    0           1;
     lambda     0           -1;
     0          -dz_min     1;
     0          beta      -1];

b = [xu;
     -xl;
     yu;
     -yl;
     sxu*tyu - dz_min*xu;
     -sxl*tyl + lambda*xl;
     sxu*tyu - dz_min*yu;
     -sxu*tyl + beta*yu];
     
P = Polyhedron(A,b);
plot(P, 'color', 'pink');


% pDerPoint = plot3(xl,yl,sxl*tyl,'*y','LineWidth',4);
% pDerPoint = plot3(xu,yu,sxu*tyu,'*m','LineWidth',4);
% pDerPoint = plot3(xl,yu,sxu*tyl,'*c','LineWidth',4);
% pDerPoint = plot3(xu,yl,sxl*tyu,'*g','LineWidth',4);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('Name', '0 \leq xu_i; 0 \leq yu_i');
title('$0 \leq xu_i; 0 \leq yu_i$', 'interpreter', 'latex');
% plot x,y,z 3D surface of sigmoid X tanh activation function
view(3);
hold on;
% 0 <= xu_i
% 0 <= yu_i
% predicate bounds of each neuron
xl = -1.8;
xu = -0.3;
yl = -2.1;
yu = -0.6;

a = 2.0;
% a = 0.0;
surf_x = linspace(xl - a, xu + a, 60);
surf_y = linspace(yl - a, yu + a, 60);
surf_z = logsig(surf_x') * tansig(surf_y);
% surf_z = logsig(x) * y;
surf(surf_x, surf_y, surf_z);
xlabel('x'), ylabel('y'), zlabel('z');

h = gca;
plot3(h.XLim, [0 0], [0 0], 'r', 'LineWidth',2);
plot3([0 0], h.YLim, [0 0], 'r', 'LineWidth',2);
plot3([0 0], [0 0], h.ZLim, 'r', 'LineWidth',2);

pyl = yline(yl, ':c', 'LineWidth', 2);
pyu = yline(yu, ':c', 'LineWidth', 2);
pxl = xline(xl, ':c', 'LineWidth', 2);
pxu = xline(xu, ':c', 'LineWidth', 2);


% plot sigmoid X tanh activation line
x = xl:0.01:xu;
y = yl:0.01:yu;
z = logsig(x') * tansig(y);
% pAFline = plot3(x,y,z,'b','LineWidth',4);

sxl = logsig(xl);
sxu = logsig(xu);
tyl = tansig(yl);
tyu = tansig(yu);

dsxl = logsig('dn', xl);
dsxu = logsig('dn', xu);
dtyl = tansig('dn', yl);
dtyu = tansig('dn', yu);

% Four edge points
pDerPoint = plot3(xl,yl,sxl*tyl,'*y','LineWidth',4);
pDerPoint = plot3(xu,yu,sxu*tyu,'*m','LineWidth',4);
pDerPoint = plot3(xl,yu,sxu*tyl,'*c','LineWidth',4);
pDerPoint = plot3(xu,yl,sxl*tyu,'*g','LineWidth',4);

x = linspace(xl, xu, 60);
y = linspace(yl, yu, 60);
z = logsig(x') * tansig(y);
h = surf(x,y,z);

% set(h,'FaceColor', 'k', 'FaceAlpha',1.0,'FaceLighting','gouraud','EdgeColor','none')
% h = surf(x,yl*ones(length(x)),z);
% set(h,'FaceColor', 'b', 'FaceAlpha',1.0,'FaceLighting','gouraud','EdgeColor','none')
% h = surf(x,yu*ones(length(x)),z);
% set(h,'FaceColor', 'b', 'FaceAlpha',1.0,'FaceLighting','gouraud','EdgeColor','none')
% h = surf(xl*ones(length(y)),y,z);
% set(h,'FaceColor', 'm', 'FaceAlpha',1.0,'FaceLighting','gouraud','EdgeColor','none')
% h = surf(xu*ones(length(y)),y,z);
% set(h,'FaceColor', 'm', 'FaceAlpha',1.0,'FaceLighting','gouraud','EdgeColor','none')


d_smin = min(dsxl,dsxu);
d_tmin = min(dtyl,dtyu);

zl = logsigXtansig(xl, yl);
zu = logsigXtansig(xu, yu);

dzl = derivative_logsigXtansig(xl, yl);
dzlu = derivative_logsigXtansig(xl, yu);
dzul = derivative_logsigXtansig(xu, yl);
dzu = derivative_logsigXtansig(xu, yu);

dz_min = min([dsxl, dtyl, dzu, dsxu, dtyu, dzlu]);  %dzlu
lambda = (sxl*tyu - sxl*tyl)/(xu-xl);
beta = (sxl*tyl - sxu*tyl)/(yu-yl);

% plot constrained are
A = [1          0           0;
     -1         0           0;
     0          1           0;
     0          -1          0;
     -lambda    0           1;
     dz_min    0           -1;
     0          dz_min     1;
     0          -beta      -1];

b = [xu;
     -xl;
     yu;
     -yl;
     sxl*tyu - lambda*xu;
     -sxu*tyl + dz_min*xl;
     sxl*tyu + dz_min*yl;
     -sxu*tyl - beta*yu];
     
P = Polyhedron(A,b);
plot(P, 'color', 'pink');

% pDerPoint = plot3(xl,yl,sxl*tyl,'*y','LineWidth',4);
% pDerPoint = plot3(xu,yu,sxu*tyu,'*m','LineWidth',4);
% pDerPoint = plot3(xl,yu,sxu*tyl,'*c','LineWidth',4);
% pDerPoint = plot3(xu,yl,sxl*tyu,'*g','LineWidth',4);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('Name', 'xl_i < 0 and xu_i > 0; yl_i >= 0');
title('$xl_i < 0$ and $xu_i > 0;$ $yl_i >= 0$', 'interpreter', 'latex');
% plot x,y,z 3D surface of sigmoid X tanh activation function
view(3);
hold on
% xl_i < 0 and xu_i > 0
% yl_i >=  0
% predicate bounds of each neuron
xl = -1.2;
xu = 0.4;
yl = 0.1;
yu = 1.7;

a = 2.0;
% a = 0.0;
surf_x = linspace(xl - a, xu + a, 60);
surf_y = linspace(yl - a, yu + a, 60);
surf_z = logsig(surf_x') * tansig(surf_y);
% surf_z = logsig(x) * y;
surf(surf_x, surf_y, surf_z);
xlabel('x'), ylabel('y'), zlabel('z');

h = gca;
plot3(h.XLim, [0 0], [0 0], 'r', 'LineWidth',2);
plot3([0 0], h.YLim, [0 0], 'r', 'LineWidth',2);
plot3([0 0], [0 0], h.ZLim, 'r', 'LineWidth',2);

pyl = yline(yl, ':c', 'LineWidth', 2);
pyu = yline(yu, ':c', 'LineWidth', 2);
pxl = xline(xl, ':c', 'LineWidth', 2);
pxu = xline(xu, ':c', 'LineWidth', 2);


% plot sigmoid X tanh activation line
x = xl:0.01:xu;
y = yl:0.01:yu;
z = logsig(x') * tansig(y);
% pAFline = plot3(x,y,z,'b','LineWidth',4);

sxl = logsig(xl);
sxu = logsig(xu);
tyl = tansig(yl);
tyu = tansig(yu);

dsxl = logsig('dn', xl);
dsxu = logsig('dn', xu);
dtyl = tansig('dn', yl);
dtyu = tansig('dn', yu);


% Four edge points
pDerPoint = plot3(xl,yl,sxl*tyl,'*y','LineWidth',4);
pDerPoint = plot3(xu,yu,sxu*tyu,'*m','LineWidth',4);
pDerPoint = plot3(xl,yu,sxu*tyl,'*c','LineWidth',4);
pDerPoint = plot3(xu,yl,sxl*tyu,'*g','LineWidth',4);

x = linspace(xl, xu, 60);
y = linspace(yl, yu, 60);
z = logsig(x') * tansig(y);
h = surf(x,y,z);

% k = ones(size(x))' * tansig(x);
% h = surf(x, zeros(size(x)), k);
% 
% k = ones(size(x))' * logsig(x);
% h = surf(x, zeros(size(x)), k);

% set(h,'FaceColor', 'k', 'FaceAlpha',1.0,'FaceLighting','gouraud','EdgeColor','none')
% h = surf(x,yl*ones(length(x)),z);
% set(h,'FaceColor', 'b', 'FaceAlpha',1.0,'FaceLighting','gouraud','EdgeColor','none')
% h = surf(x,yu*ones(length(x)),z);
% set(h,'FaceColor', 'b', 'FaceAlpha',1.0,'FaceLighting','gouraud','EdgeColor','none')
% h = surf(xl*ones(length(y)),y,z);
% set(h,'FaceColor', 'm', 'FaceAlpha',1.0,'FaceLighting','gouraud','EdgeColor','none')
% h = surf(xu*ones(length(y)),y,z);
% set(h,'FaceColor', 'm', 'FaceAlpha',1.0,'FaceLighting','gouraud','EdgeColor','none')


d_smin = min(dsxl,dsxu);
d_tmin = min(dtyl,dtyu);

zl = logsigXtansig(xl, yl);
zu = logsigXtansig(xu, yu);

dzl = derivative_logsigXtansig(xl, yl);
dzu = derivative_logsigXtansig(xu, yu);


dz_min = min([dzl, dsxl, dtyl, dzu, dsxu, dtyu, dzlu]);  %dzlu
% dz_min = min(dzl, dzu);
lambda = (sxl*tyu-sxl*tyl)/(xu-xl);

% plot constrained are
A = [1          0           0;
     -1         0           0;
     0          1           0;
     0          -1          0;
     -dz_min     0           1;
     lambda    0          -1;
     0          -dz_min     1;
     0          beta      -1];

b = [xu;
     -xl;
     yu;
     -yl;
     sxu*tyu - dz_min*xu;
     -sxl*tyl + lambda*xl;
     sxu*tyu - dz_min*yu;
     -sxu*tyl + beta*yu];
     
P = Polyhedron(A,b);
plot(P, 'color', 'pink');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure('Name', 'xl_i >= 0; yl_i < 0 and yu_i > 0');
title('$xl_i >= 0; yl_i < 0$ and $yu_i > 0$', 'interpreter', 'latex');
% plot x,y,z 3D surface of sigmoid X tanh activation function
view(3);
hold on
% xl_i >= 0
% yl_i < 0 and yu_i > 0
% predicate bounds of each neuron
xl = 0.1;
xu = 1.7;
yl = -1.2;
yu = 0.4;

a = 2.0;
% a = 0.0;
surf_x = linspace(xl - a, xu + a, 60);
surf_y = linspace(yl - a, yu + a, 60);
surf_z = logsig(surf_x') * tansig(surf_y);
% surf_z = logsig(x) * y;
surf(surf_x, surf_y, surf_z);
xlabel('x'), ylabel('y'), zlabel('z');

h = gca;
plot3(h.XLim, [0 0], [0 0], 'r', 'LineWidth',2);
plot3([0 0], h.YLim, [0 0], 'r', 'LineWidth',2);
plot3([0 0], [0 0], h.ZLim, 'r', 'LineWidth',2);

pyl = yline(yl, ':c', 'LineWidth', 2);
pyu = yline(yu, ':c', 'LineWidth', 2);
pxl = xline(xl, ':c', 'LineWidth', 2);
pxu = xline(xu, ':c', 'LineWidth', 2);


% plot sigmoid X tanh activation line
x = xl:0.01:xu;
y = yl:0.01:yu;
z = logsig(x') * tansig(y);
% pAFline = plot3(x,y,z,'b','LineWidth',4);

sxl = logsig(xl);
sxu = logsig(xu);
tyl = tansig(yl);
tyu = tansig(yu);

dsxl = logsig('dn', xl);
dsxu = logsig('dn', xu);
dtyl = tansig('dn', yl);
dtyu = tansig('dn', yu);


% Four edge points
pDerPoint = plot3(xl,yl,sxl*tyl,'*y','LineWidth',4);
pDerPoint = plot3(xu,yu,sxu*tyu,'*m','LineWidth',4);
pDerPoint = plot3(xl,yu,sxu*tyl,'*c','LineWidth',4);
pDerPoint = plot3(xu,yl,sxl*tyu,'*g','LineWidth',4);

x = linspace(xl, xu, 60);
y = linspace(yl, yu, 60);
z = logsig(x') * tansig(y);
h = surf(x,y,z);

% k = ones(size(x))' * tansig(x);
% h = surf(x, zeros(size(x)), k);
% 
% k = ones(size(x))' * logsig(x);
% h = surf(x, zeros(size(x)), k);

% set(h,'FaceColor', 'k', 'FaceAlpha',1.0,'FaceLighting','gouraud','EdgeColor','none')
% h = surf(x,yl*ones(length(x)),z);
% set(h,'FaceColor', 'b', 'FaceAlpha',1.0,'FaceLighting','gouraud','EdgeColor','none')
% h = surf(x,yu*ones(length(x)),z);
% set(h,'FaceColor', 'b', 'FaceAlpha',1.0,'FaceLighting','gouraud','EdgeColor','none')
% h = surf(xl*ones(length(y)),y,z);
% set(h,'FaceColor', 'm', 'FaceAlpha',1.0,'FaceLighting','gouraud','EdgeColor','none')
% h = surf(xu*ones(length(y)),y,z);
% set(h,'FaceColor', 'm', 'FaceAlpha',1.0,'FaceLighting','gouraud','EdgeColor','none')


d_smin = min(dsxl,dsxu);
d_tmin = min(dtyl,dtyu);

zl = logsigXtansig(xl, yl);
zu = logsigXtansig(xu, yu);

dzl = derivative_logsigXtansig(xl, yl);
dzu = derivative_logsigXtansig(xu, yu);


dz_min = min([dzl, dsxl, dtyl, dzu, dsxu, dtyu, dzlu]);  %dzlu
% dz_min = min(dzl, dzu);
lambda = (sxu*tyu - sxl*tyl)/(xu - xl);
% plot constrained are
A = [1          0           0;
     -1         0           0;
     0          1           0;
     0          -1          0;
% %      dz_min     0           1;
     -lambda    0           1;
     -dz_min    0           -1;
     0          dz_min     1;
     0          dz_min      -1];

b = [xu;
     -xl;
     yu;
     -yl;
% %      sxu*tyu + dz_min*xu;
     sxu*tyu - lambda*xu;
     -sxu*tyl - dz_min*xl;
     sxu*tyu + dz_min*yu;
     -sxu*tyl + dz_min*yu];
     
P = Polyhedron(A,b);
plot(P, 'color', 'pink');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% pDerPoint = plot3(xl,yl,sxl*tyl,'*y','LineWidth',4);
% pDerPoint = plot3(xu,yu,sxu*tyu,'*m','LineWidth',4);
% pDerPoint = plot3(xl,yu,sxu*tyl,'*c','LineWidth',4);
% pDerPoint = plot3(xu,yl,sxl*tyu,'*g','LineWidth',4);

function z = logsigXtansig(x, y)
    z = logsig(x) * tansig(y);
end

function z = derivative_logsigXtansig(x, y)
    z = logsig('dn', x) * tanh(y) + logsig(x) * tansig('dn', y);
end