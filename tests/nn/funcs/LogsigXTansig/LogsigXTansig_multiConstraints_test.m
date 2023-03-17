% author: Sung Woo Choi
% date: 02/22/2023

% clc; clear; close all;
format long;
pink = [double(0xFA)/double(0xFF), double(0x86)/double(0xC4), double(0xFA)/double(0xFF)];

%%
figure;
nexttile;
% plot x,y,z 3D surface of x * y activation function
view(3);
xlabel('x');
ylabel('y'); 
zlabel('z'); 
title('logsig(x) .* tansig(y)')
hold on;

% randomly generate state bounds of each neuron (upper bounds are always bigger than lower bounds)
xl = 1-5*rand;
xu = xl+5*rand;
yl = 1-5*rand;
yu = yl+5*rand;

fprintf('logsig(x)*tansig(y) \n')
fprintf('x-coordinate ranges: \t xl = %1.12f \t xu = %1.12f\n', xl, xu);
fprintf('y-coordinate ranges: \t yl = %1.12f \t yu = %1.12f\n', yl, yu);

% plot the surface of state bounds
x = linspace(xl, xu, 60);
y = linspace(yl, yu, 60);
z = tansig(y)' * logsig(x);
h = surf(x,y,z);

% plot x, y, and z coordinates
h = gca;

% plot four corner points
pDerPoint = plot3(xl, yl, LogsigXTansig.f(xl, yl), '*y','LineWidth',4);
pDerPoint = plot3(xu, yu, LogsigXTansig.f(xu, yu), '*m','LineWidth',4);
pDerPoint = plot3(xu, yl, LogsigXTansig.f(xu, yl), '*c','LineWidth',4);
pDerPoint = plot3(xl, yu, LogsigXTansig.f(xl, yu), '*g','LineWidth',4);


[U_x, U_y, U_b, L_x, L_y, L_b, zmax, zmin] = LogsigXTansig.getMultiConstraints(xl, xu, yl, yu);

fprintf('upper-plane constrant: \t %1.12f x +\t %1.12f y <= %1.12f \n', U_x, U_y, U_b);
fprintf('lower-plane constrant: \t %1.12f x +\t %1.12f y >= %1.12f \n', L_x, L_y, L_b);


% worst case senario: outter box of logsigXtansig function
A = [ 1         0           0;
     -1         0           0;
      0         1           0;
      0        -1           0;
      0         0           1;
      0         0          -1];
b = [xu;
     -xl;
     yu;
     -yl;
     zmax;
     -zmin];
T = Polyhedron(A,b);
% P_cyan = plot(T, 'color', 'cyan');
% alpha(P_cyan, 0.23);


% plot constrained are
A = [1          0           0;
     -1         0           0;
     0          1           0;
     0         -1           0;
     U_x       U_y          1;
     -L_x      -L_y        -1];

b = [xu; 
     -xl; 
     yu; 
     -yl; 
     U_b; 
     -L_b];
     
P = Polyhedron(A,b);
P_pink = plot(P, 'color', 'pink');
alpha(P_pink, 0.65);

