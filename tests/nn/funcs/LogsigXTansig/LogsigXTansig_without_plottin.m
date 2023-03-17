clc; clear; close all;
format long;
global NUM_OF_POINTS
pink = [double(0xFA)/double(0xFF), double(0x86)/double(0xC4), double(0xFA)/double(0xFF)];
global xl xu yl yu

%%
figure;
nexttile;
% plot x,y,z 3D surface of x * y activation function
view(3);
xlabel('x');
ylabel('y'); 
zlabel('z'); 
title('logsig(x)*tansig(y)')
hold on;

% randomly generate state bounds of each neuron (upper bounds are always bigger than lower bounds)
xl = 1-5*rand;
xu = xl+5*rand;
yl = 1-5*rand;
yu = yl+5*rand;



% xl = 0.109337727998311
% xu = 0.749409726599174
% yl = -3.995401973806803
% yu = -3.139796642024643

%ex display example
% xl = -2.309723759528260
% xu = 1.541703814490041
% yl = -0.751090067205525
% yu = 2.558957924590147

%ex1
% xl = -0.469703097748226
% xu = 2.821991380093152
% yl = -3.288804462454799
% yu = 1.191925739973019

%ex2
% xl = -3.727364226398228
% xu = 0.668390132546207
% yl = -0.286609224477632
% yu = 1.390917807588175


%ex3
% xl = -3.364227261429982
% xu = 1.472901496713768
% yl = -0.989011496938018
% yu = 3.002853013675074

%ex4 fix (make tigther)** ***
% xl = -3.446702644664448
% xu = 0.795492952785262
% yl = -0.011045017101146
% yu = 4.425373087471655

% ex5 fix (wrong bound must fix)**
% xl = -1.736209718678366
% xu = -0.033030877325831
% yl = -2.843890762140755
% yu = 1.906084928988629

%ex6 fix (make tigther)**
% xl = -3.582764106670566
% xu = -0.646130228419139
% yl = -0.488484297056392
% yu = 3.459323069475695

%ex7 fix (make tigther)**
% xl = -1.522187013663468
% xu = 1.984867795329038
% yl = -3.968208278448234
% yu = -0.864557379410363

%ex8 fix (wrong bound must fix / weird case)**
% xl = -1.653464396614707
% xu = -0.746715165974905
% yl = -2.097710722164520
% yu = -0.284066134373541

% ex9 fix (direct connection between global maximum and global minimum)**
% xl = -3.591912206723565
% xu = -1.105447286033761
% yl = -0.892901542107648
% yu = 0.449308938897594

%ex 10 explore
% xl = -3.5003
% xu = -1.6540
% yl = 0.4440
% yu = 4.3452

%ex 11 fix (not tight bound fix)**
% xl = -1.121547484165687
% xu = 0.229804632994639
% yl = 0.014731009522721
% yu = 4.123336934329271

%ex12 fix red dot
% xl = -0.029877577661217
% xu = 4.709788028804627
% yl = 0.589643964511371
% yu = 1.118191097419978

%ex13 must fix (wrong bounds)**
% xl = -1.987450959362897
% xu = -0.310894305836666
% yl = -0.496125116665533
% yu = 1.766837591181087

%ex14 must fix (make tighter bounds)****** must fix**********
% xl = -0.143347410527507
% xu = 0.177588026431986
% yl = -2.836647553882872
% yu = 0.519363372899805


%ex15 must fix (wrong bounds)*******
% xl = -1.273474324959539;
% xu = 0.793662120144537;
% yl = -0.088660341786502;
% yu = 0.539612595026629;

%ex16 must fix (make tigther bounds)**
% xl = 0.395000;
% xu = 4.676753;
% yl = -3.498880;
% yu = -2.409554;

%ex17 must fix (make tighter bounds)**
% xl = -3.380922;
% xu = -0.330460;
% yl = -0.017962;
% yu = 2.581622;

%ex18 must fix (make tigther bounds)***
% xl = 0.738474
% xu = 3.522627
% yl = -2.560126
% yu = -0.120573

%ex19 must fix (idea might be wrong!!) ***!!!
% xl = -3.339416;
% xu = 0.368097;
% yl = -1.239364;
% yu = 2.308831;

%ex20 not tight bounds
% xl = -1.316303;	 
% xu = -0.255487;
% yl = 0.507406;	 
% yu = 4.625279;

%ex21 just check
% xl = 0.371727 	 
% xu = 1.916300
% yl = -2.630522 	 
% yu = 1.283838

%ex22 opposite slope (must fix)******
% xl = 0.419407436103 	 
% xu = 0.707679242177
% yl = -3.898826119880 	 
% yu = -2.474707485577

%ex23 opposite slope (must fix)******
% xl = 0.090097664155 	 
% xu = 0.555042298508
% yl = -1.317446238811 	 
% yu = -1.270783678672

%ex24 (good example to check optimal point)
% xl = -2.810548546056 	 
% xu = -1.072712793812
% yl = -1.306158796957 	 
% yu = 1.890460014039

%ex25 (good example when green is optimal)*********
% xl = -0.626494962472
% xu = 3.532717589958
% yl = -3.051473953719
% yu = -0.266482647326



%ex26 wrong bounds (must fix) ************** (green case)
% xl = 0.260912929092 	 
% xu = 0.359736247736
% yl = -3.821458651498 	 
% yu = 1.030405858569

%ex27 still can make tighter bounds----
% xl = -0.281925798984 	 
% xu = 4.667401376772
% yl = -0.749040205222 	 
% yu = 0.293571664245

%ex28 error in algorithm ****---
% xl = -2.820354456182 	 
% xu = 2.119440251808
% yl = 0.373376042922 	 
% yu = 2.195761918935

%ex29
% xl = -1.394273720255 	 
% xu = 0.146314723894
% yl = -2.722222238138 	 
% yu = 1.474524149706


%ex30
% xl = -1.602579436626 	 
% xu = 2.924734312028
% yl = -1.012651177985 	 
% yu = 0.066155082877

%31 need ti fix************************
% xl = -1.533936897674 	 
% xu = 0.106501808569
% yl = -2.767620705471 	 
% yu = 1.412408417546

%ex32 wrong slope sides ***********
% xl = -3.633103012419 	 
% xu = -3.158636233485
% yl = -0.877093918233 	 
% yu = 1.852900401901


%ex33 contradiction to monotonically decreasing X NO just wrong slope implemented***************
% xl = -1.117624356114 	 
% xu = 0.250356748360 
% yl = -1.222829219333 	 
% yu = 1.914746015174


%ex34 should we care pyx tangent slope?***
% xl = -2.661828738929 	 
% xu = -1.906302344627
% yl = -3.076166383435 	 
% yu = 1.471846275871


%ex35 interesting case that needs to be fixed*********NOT tight bound
% xl = -1.613863831764 	 
% xu = -0.247092003723
% yl = -2.591895142178 	 
% yu = 1.298192639610

%ex36 wrong bounds: MUST FIX************
% xl = -3.378002404045;	 
% xu = 0.407774957806;
% yl = -0.148128572929; 	 
% yu = 1.646589523496;

%ex37 why not tight bound?
% xl = -2.366282303719; 	 
% xu = 1.837329028498;
% yl = -1.104271402173; 	 
% yu = 2.414415345778;

%ex38 
% xl = -2.432338669522; 	 
% xu = 2.346291393191;
% yl = -0.539532306651; 	 
% yu = 1.171972006097;

%ex39
% xl = -2.244480643080; 	 
% xu = 1.147396329406;
% yl = 0.705861804660; 	 
% yu = 1.825628122643;

%ex40 incorrect boundss
% xl = -1.682342029834; 	 
% xu = -1.140051643180;
% yl = -3.268808442623; 	 
% yu = 1.285447188998;

%ex41 optimize more********************** (work with ex42)
% xl = -1.128342372595; 	 
% xu = 1.203547539665;
% yl = -1.124655917624;
% yu = 1.151394292581;

%ex42 incorrect bounds******************* (work with ex41)
% xl = -1.117264594814;	 
% xu = -0.663148165876;
% yl = -0.332357453895; 	 
% yu = 0.435926134061;

%ex43 similiar problem with ex41, ex42
% xl = 0.589643964511 	 
% xu = 1.118191097420
% yl = 0.289794390480 	 
% yu = 1.122096594862

fprintf('x-coordinate ranges: \t xl = %1.12f \t xu = %1.12f\n', xl, xu);
fprintf('y-coordinate ranges: \t yl = %1.12f \t yu = %1.12f\n', yl, yu);

% plot the surface of state bounds
x = linspace(xl, xu, 60);
y = linspace(yl, yu, 60);
z = tansig(y)' * logsig(x);
% z = logsig(x)' * tansig(y); 
h = surf(x,y,z);

% plot x, y, and z coordinates
% h = gca;
% plot3([min(0, h.XLim(1)), max(0, h.XLim(2))], [0 0], [0 0], 'Color', pink, 'LineWidth',2);
% plot3([0 0], [min(h.YLim(1), 0), max(0, h.YLim(2))], [0 0], 'Color', pink, 'LineWidth',2);
% plot3([0 0], [0 0], [min(h.ZLim(1), 0), max(0, h.ZLim(2))], 'Color', pink, 'LineWidth',2);

% plot four corner points
% pDerPoint = plot3(xl,yl, z1 ,'*y','LineWidth',4);
pDerPoint = plot3(xl, yl, logsigXtansig(xl, yl), '*y','LineWidth',4);
pDerPoint = plot3(xu, yu, logsigXtansig(xu, yu), '*m','LineWidth',4);
pDerPoint = plot3(xu, yl, logsigXtansig(xu, yl), '*c','LineWidth',4);
pDerPoint = plot3(xl, yu, logsigXtansig(xl, yu), '*g','LineWidth',4);



% Initialize index definition
global u_ l_ x_ y_ z_ si_ dzx_ dzy_ z_max z_min iux_ iuy_
u_ = 1;
l_ = 2;
x_ = 1;
y_ = 2;
z_ = 3;
si_ = 4; % sorted index
dzx_ = 4; % grandient in x-coordinate
dzy_ = 5; % grandient in y-coordinate
% dz_ = 6;
iux_ = 6; % intersection line in x-coordinate
iuy_ = 7; % intersection line in y-coordinate

z_max = 4;
z_min = 1;


% corner points of a function surface
%{
    pz = [xl, yl, z1, dz1/dx, dz1/dy;
          xu, yu, z2, dz2/dx, dz2/dy;
          xu, yl, z3, dz3/dx, dz3/dy;
          xl, yu, z4, dz4/dx, dz4/dy;
    dz represents the derivative of z (tangent line at point z(0, i))
%}

NUM_OF_POINTS = 4; % number of points
pz = zeros(NUM_OF_POINTS, NUM_OF_POINTS);
pz(1, x_:y_) = [xl, yl];
pz(2, x_:y_) = [xu, yu];
pz(3, x_:y_) = [xu, yl];
pz(4, x_:y_) = [xl, yu];

for i = 1:NUM_OF_POINTS
    pz(i, z_) = logsigXtansig(pz(i, x_), pz(i, y_));
    pz(i, dzx_:dzy_) = gradient_logsigXtansig(pz(i, x_), pz(i, y_));
end

fprintf('\n\n');
fprintf('pz(1): [xl, yl, z1, dz1/dx, dz1/dy], z1 = logsigXtansig(xl, yl), yellow color\n');
fprintf('pz(2): [xu, yu, z2, dz2/dx, dz1/dy], z2 = logsigXtansig(xu, yu), magenta color\n');
fprintf('pz(3): [xu, yl, z3, dz3/dx, dz1/dy], z3 = logsigXtansig(xu, yl), cyan color\n');
fprintf('pz(4): [xl, yu, z4, dz4/dx, dz1/dy], z4 = logsigXtansig(xl, yu), green color\n');
pz


% sort z-coordinate points in ascending order
[z_sorted, z_sorted_i] = sort(pz(:, z_));
pzs = pz(z_sorted_i, :);
% pzs = [pzs, nan(4, 6)];
pzs = [pzs, nan(4, 2)];

fmt = ['asceding order sort points in z-coordinate (z_sorted_i): \n [', ...
    repmat('%g, ', 1, numel(z_sorted_i)-1), '%g]\n'];
fprintf(fmt, z_sorted_i);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                       Upper bound case
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('\n\n');
fprintf('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n');
fprintf('\t\t\t\t Upper bound case\n');
fprintf('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n');

su_max2min_i = nan;
% upper bound case for finding slopes from global maximum to other points
for i = z_max-1:-1:z_min
    z_max_range = pzs(z_max, z_) - pzs(i, z_); % slope in z coordinate
    x_max_range = pzs(z_max, x_) - pzs(i, x_);
    y_max_range = pzs(z_max, y_) - pzs(i, y_);
    pzs(i, iux_) = z_max_range / x_max_range; % slope in x coordinate
    pzs(i, iuy_) = z_max_range / y_max_range; % slope in y coordinate
    if x_max_range ~= 0 && y_max_range ~= 0
        pzs(i, iux_) = pzs(i, iux_) / 2.0;
        pzs(i, iuy_) = pzs(i, iuy_) / 2.0;
        su_max2min_i = i;
    end
end

fprintf('pz(1): [x, \t y, \t z1_min, \t dz1/dx, \t dz1/dy \t intl_x, \t intl_y]\n');
fprintf('pz(2): [x, \t y, \t z2, \t\t dz2/dx, \t dz1/dy \t intl_x, \t intl_y]\n');
fprintf('pz(3): [x, \t y, \t z3, \t\t dz3/dx, \t dz1/dy \t intl_x, \t intl_y]\n');
fprintf('pz(4): [x, \t y, \t z4_max, \t dz4/dx, \t dz1/dy \t intl_x, \t intl_y]\n');
pzs

tan_i_ = 1;
int_max2min_i_ = su_max2min_i + tan_i_;
%% finding minimal slopes among tangent lines and intersection lines.
% min x-coordinate slopes; check (gradient) tangent line and intersection
% lines from global maximum
u_x_slope = [pzs(z_max,dzx_), pzs(z_min:z_max-1, iux_)'];
[u_x_min_slope, u_x_slope_i] = min(abs(u_x_slope));

% min y-coordinate slopes; check (gradient) tangent line and intersection 
% lines from global maximum
u_y_slope = [pzs(z_max,dzy_), pzs(z_min:z_max-1, iuy_)'];
[u_y_min_slope, u_y_slope_i] = min(abs(u_y_slope));

% special case: slope of one coordinate is greater than 1/2 of coordinate 
% of intersection line between  global maxima and global minima, while
% other slope is smaller
% Let c = slope of intersection line between 
%   global maximum and global minimum
px_py_points = 0; % for plotting purposes only
if (  abs(u_x_min_slope) == abs(pzs(su_max2min_i, iux_))  ) && (  abs(u_y_min_slope) < abs(pzs(su_max2min_i, iuy_))  )
    % for case in which
    %   x-axis slope == 0.5 * dc/dx and
    %   y-axis slope <  0.5 * dc/dy
    %   ==> there is a possibility that x-axis can be greater than 0.5 * dc/dx
    u_x_slope(u_x_slope_i) = u_x_slope(u_x_slope_i) * 2.0;
    u_x_min_slope = min(abs(u_x_slope))
    
    % In x-axis, if optimal slope is not tangent line
    if u_x_min_slope ~= u_x_slope(tan_i_)
        [px_i, py_i] = get_pointsXY(pzs);
        % get a point that has the same y-axis value as the global maximum but
        % different x-axis value to check tangent slope on y-axis => px
        px = pzs(px_i,:);
        px(z_) = pzs(z_max, z_) - u_x_min_slope*(xu-xl);
        
        % get a point that has the same x-axis value as the global maximum but
        % different y-axis value to check tangent slope on x-axis => py
        py = pzs(py_i,:);
        py(z_) = pzs(z_max, z_) - u_y_min_slope*(yu-yl);
    
        % a point opposite to global maximum on x-axis and y-axis
        po = pzs(su_max2min_i, :);
        
        % intersection line on y-axis (pxy) between two local points
        pxy_inter_y = ( px(z_) - po(z_) ) / (yu - yl);
       
        % tangent line on y-axis
        pxy_dxdy = gradient_logsigXtansig(px(x_), px(y_));
        
        % compare x-axis slopes with the x-axis slope in the opposite face
        u_y_min_slope = min(abs([u_y_min_slope, pxy_inter_y, pxy_dxdy(y_)]));
    
    else
        % a point opposite to global maximum on x-axis and y-axis
        po = pzs(su_max2min_i, x_:z_);
        new_z = pzs(z_max, z_) - u_x_min_slope*(xu-xl) - u_y_min_slope*(yu-yl);
        
        % po(z_) = pzs(z_max, z_) - u_x_min_slope*(xu-xl) - u_y_min_slope*(yu-yl)
        if new_z < po(z_)
            z_range = po(z_) - pzs(z_max, z_);
            x_slope = -(z_range + u_y_min_slope*(yu-yl))/(xu-xl);

            % get a point that has the same x-axis value as the global maximum but
            % different y-axis value to check tangent slope on x-axis => py
            py_i = get_pointY(pzs);
            py = pzs(py_i,x_:z_);
            pyx_dxdy = gradient_logsigXtansig(py(x_), py(y_));

            u_x_min_slope = min(abs([u_x_min_slope, x_slope,  pyx_dxdy(x_)]));

        end
    end


elseif (  abs(u_x_min_slope) < abs(pzs(su_max2min_i, iux_))  ) && (  abs(u_y_min_slope) == abs(pzs(su_max2min_i, iuy_))  )
    % for case in which
    %   x-axis slope <  0.5 * dc/dx and
    %   y-axis slope == 0.5 * dc/dy
    %   ==> there is a possibility that y-axis can be greater than 0.5 * dc/dy
    u_y_slope(u_y_slope_i) = u_y_slope(u_y_slope_i) * 2.0;
    u_y_min_slope = min(abs(u_y_slope));
    
    % In y-axis, if optimal slope is not tangent line
    if u_y_min_slope ~= u_y_slope(tan_i_)
        [px_i, py_i] = get_pointsXY(pzs);
        % get a point that has the same y-axis value as the global maximum but
        % different x-axis value to check tangent slope on y-axis => px
        px = pzs(px_i,x_:z_);
        px(z_) = pzs(z_max, z_) - u_x_min_slope*(xu-xl);

        % get a point that has the same x-axis value as the global maximum but
        % different y-axis value to check tangent slope on x-axis => py
        py = pzs(py_i,x_:z_);
        py(z_) = pzs(z_max, z_) - u_y_min_slope*(yu-yl);

        % a point opposite to global maximum on x-axis and y-axis
        po = pzs(su_max2min_i, x_:z_);
    
        % intersection line on x-axis (pyx) between two local points
        pyx_inter_x = ( py(z_) - po(z_) ) / (xu - xl);
        
        % tangent line on x-axis
        pyx_dxdy = gradient_logsigXtansig(py(x_), py(y_));   
    
        % compare x-axis slopes with the x-axis slope in the opposite face
        u_x_min_slope = min(abs([u_x_min_slope, pyx_inter_x, pyx_dxdy(x_)]));

    else
        % a point opposite to global maximum on x-axis and y-axis
        po = pzs(su_max2min_i, x_:z_);
        new_z = pzs(z_max, z_) - u_x_min_slope*(xu-xl) - u_y_min_slope*(yu-yl);
        
        % po(z_) = pzs(z_max, z_) - u_x_min_slope*(xu-xl) - u_y_min_slope*(yu-yl)
        if new_z < po(z_)
            z_range = po(z_) - pzs(z_max, z_) 
            y_slope = -(z_range + u_x_min_slope*(xu-xl))/(yu-yl)
            
            % get a point that has the same y-axis value as the global maximum but
            % different x-axis value to check tangent slope on y-axis => px
            px_i = get_pointX(pzs);
            px = pzs(px_i,x_:z_);
            pxy_dxdy = gradient_logsigXtansig(px(x_), px(y_));

            u_y_min_slope = min(abs([u_y_min_slope, y_slope, pxy_dxdy(y_)]));

        end
    end

% if optimal slope in x-axis is an interslection line between global
% maximum and a local point in x-axis
elseif (u_x_slope_i ~= tan_i_) && (u_x_slope_i ~= int_max2min_i_)
    px_i = get_pointX(pzs);
    px_dxdy = gradient_logsigXtansig(pzs(px_i, x_), pzs(px_i, y_));
    u_y_min_slope = min(abs([u_y_min_slope, px_dxdy(y_)]));


% if optimal slope in y-axis is an interslection line between global
% maximum and a local point in y-axis
elseif (u_y_slope_i ~= tan_i_) && (u_y_slope_i ~= int_max2min_i_)
    py_i = get_pointY(pzs);
    py_dxdy = gradient_logsigXtansig(pzs(py_i, x_), pzs(py_i, y_));
    u_x_min_slope = min(abs([u_x_min_slope, py_dxdy(x_)]));

end


%%
nexttile;
% plot x,y,z 3D surface of x * y activation function
view(3);
xlabel('x');
ylabel('y'); 
zlabel('z'); 
hold on;

% plot the surface of state bounds
x = linspace(xl, xu, 60);
y = linspace(yl, yu, 60);
z = tansig(y)' * logsig(x);
% z = logsig(x)' * tansig(y); 
h = surf(x,y,z);

% plot x, y, and z coordinates
h = gca;

% plot four corner points
% pDerPoint = plot3(xl,yl, z1 ,'*y','LineWidth',4);
pDerPoint = plot3(pz(1,x_), pz(1,y_), pz(1,z_) ,'*y','LineWidth',4);
pDerPoint = plot3(pz(2,x_), pz(2,y_), pz(2,z_) ,'*m','LineWidth',4);
pDerPoint = plot3(pz(3,x_), pz(3,y_), pz(3,z_) ,'*c','LineWidth',4);
pDerPoint = plot3(pz(4,x_), pz(4,y_), pz(4,z_) ,'*g','LineWidth',4);

if px_py_points
    plot3(py(x_), py(y_), py(z_), '+k','LineWidth',10);
    plot3(px(x_), px(y_), px(z_), 'xk','LineWidth',10);
end

% polytope slopes on x-coordinate and y-coordinate
if z_sorted_i(z_max) == 2
    U_x = -u_x_min_slope;
else
    U_x = u_x_min_slope;
end

if z_sorted_i(z_max) == 2
    U_y = -u_y_min_slope;
else
    U_y = -u_y_min_slope;
end
U_b = pzs(z_max,z_) + U_x*pzs(z_max,x_) + U_y*pzs(z_max,y_);

fprintf('\n\n');
fprintf('New upper polytope plane constraints:\n');
fprintf('\t\tU_x: %f \t\t U_y: %f \t U_b: %f\n\n', U_x, U_y, U_b);

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
     pzs(z_max, z_);
     -pzs(z_min, z_)];
T = Polyhedron(A,b);
% P_cyan = plot(T, 'color', 'cyan');
% alpha(P_cyan, 0.23);

% plot constrained are
A = [ 1         0       0;
     -1         0       0;
      0         1       0;
      0        -1       0;
    U_x       U_y       1;
      0         0      -1];
b = [xu;
     -xl;
     yu;
     -yl;
     U_b;
     -pzs(z_min, z_)];
P = Polyhedron(A,b);
P_pink = plot(P, 'color', 'pink');
alpha(P_pink, 0.65);


function [al, au] = reverse_bound(al, au)
    at = au;
    au = -al;
    al = -au;
end

function point_index = get_pointX(a)
    global z_min z_max iux_ iuy_
    for i = z_max-1:-1:z_min
        if a(i, iuy_) == inf
            point_index = i;
            break;
        end
    end
end

function point_index = get_pointY(a)
    global z_min z_max iux_ iuy_
    for i = z_max-1:-1:z_min
        if a(i, iux_) == inf
            point_index = i;
            break;
        end
    end
end

function [p_xi, p_yi] = get_pointsXY(a)
    global z_min z_max iux_ iuy_
    for i = z_max-1:-1:z_min
        if a(i, iuy_) == inf
            p_xi = i;
        elseif a(i, iux_) == inf
            p_yi = i;
        end
    end
end

function b = pNOR_x(a)
    global xl xu x_
    if a(x_) == xl
        b = xu;
    elseif a(x_) == xu
        b = xl;
    else
        error('Please enter xl or xu');
    end
end

function b = pNOR_y(a)
    global yl yu y_
    if a(y_) == yl
        b = yu;
    elseif a(y_) == yu
        b = yl;
    else
        error('Please enter yl or yu');
    end
end

function b = NOR_(a)
    global xl xu yl yu
    if a == xl
        b = xu;
    elseif a == xu
        b = xl;
    elseif a == yl
        b = yu;
    elseif a == yu
        b = yl;
    end
end

function b = NOR_x(a)
    global xl xu
    if a == xl
        b = xu;
    elseif a == xu
        b = xl;
    end
end
function b = NOR_y(a)
    global yl yu
    if a == yl
        b = yu;
    elseif a == yu
        b = yl;
    end
end

function z = logsigXtansig(x, y)
    z = logsig(x) * tansig(y);
end

function dz = derivative_logsigXtansig(x, y)
    dz = logsig('dn', x) * tanh(y) + logsig(x) * tansig('dn', y);
end

function dz = gradient_logsigXtansig(x, y)
    dz = [logsig('dn', x) * tanh(y); logsig(x) * tansig('dn', y)];
end