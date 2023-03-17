% author: Sung Woo Choi
% date: 02/23/2023

% clc; clear; close all;
format long;
pink = [double(0xFA)/double(0xFF), double(0x86)/double(0xC4), double(0xFA)/double(0xFF)];

%%
% figure;
% nexttile;
% % plot x,y,z 3D surface of x * y activation function
% view(3);
% xlabel('x');
% ylabel('y'); 
% zlabel('z'); 
% title('x .* y')
% hold on;

% randomly generate state bounds of each neuron (upper bounds are always bigger than lower bounds)
dim = 2;
xl = 1-5*rand(dim, 1);
xu = xl+5*rand(dim, 1);
yl = 1-5*rand(dim, 1);
yu = yl+5*rand(dim, 1);

fprintf('x'' * y \n')
fprintf('x-coordinate ranges: \t xl = %1.12f \t xu = %1.12f\n', xl, xu);
fprintf('y-coordinate ranges: \t yl = %1.12f \t yu = %1.12f\n', yl, yu);

% plot the surface of state bounds
% x = linspace(xl, xu, 60);
% y = linspace(yl, yu, 60);
% z = y' * x;
% h = surf(x,y,z);

% plot x, y, and z coordinates
% h = gca;

% plot four corner points
% pDerPoint = plot3(xl, yl, IdentityXIdentity.f(xl, yl), '*y','LineWidth',4);
% pDerPoint = plot3(xu, yu, IdentityXIdentity.f(xu, yu), '*m','LineWidth',4);
% pDerPoint = plot3(xu, yl, IdentityXIdentity.f(xu, yl), '*c','LineWidth',4);
% pDerPoint = plot3(xl, yu, IdentityXIdentity.f(xl, yu), '*g','LineWidth',4);


[MU_x, MU_y, MU_b, ML_x, ML_y, ML_b, zmax, zmin] = IdentityXIdentity.getMultiConstraints(xl, xu, yl, yu);

U_x = cell(1, dim);
U_y = cell(1, dim);
U_b = cell(1, dim);
L_x = cell(1, dim);
L_y = cell(1, dim);
L_b = cell(1, dim);
for i = 1:dim
    [U_x{i}, U_y{i}, U_b{i}, L_x{i}, L_y{i}, L_b{i}, zmax, zmin] = IdentityXIdentity.getConstraints(xl(i), xu(i), yl(i), yu(i));
end

for i = 1:dim
    fprintf('upper-plane constrant: \t %1.12f x +\t %1.12f y <= %1.12f \n', U_x{i}, U_y{i}, U_b{i});
    fprintf('lower-plane constrant: \t %1.12f x +\t %1.12f y >= %1.12f \n', L_x{i}, L_y{i}, L_b{i});
end

