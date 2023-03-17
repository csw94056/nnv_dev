classdef LogsigXTansig
    % logsigXidentity class contains method for reachability analysis for
    % Layer with logsig(x) * tansig(y) function
    % author: Sung Woo Choi
    % date: 08/24/2022
    % modified: 02/23/2023

    properties (Constant)
        u_ = 1; % index for upper bound case
        l_ = 2; % index for lower bound case
    
        x_ = 1; % index for x-coordinate constraint
        y_ = 2; % index for y-coordinate constraint
        z_ = 3; % index for z-coordinate constraint

        dzx_ = 4; % grandient on x-coordinate
        dzy_ = 5; % grandient on y-coordinate

        iux_ = 6; % intersection line on x-coordinate for upper bound case
        iuy_ = 7; % intersection line on y-coordinate for upper bound case
        ilx_ = 8; % intersection line on x-coordinate for lower bound case
        ily_ = 9; % intersection line on y-coordinate for lower bound case
        
        z_max = 4;
        z_min = 1;
    
        num_of_points = 4;
    end
    
    methods(Static) % evaluate method and over-approximate reachability analysis with stars

        % evaluation
        function z = f(x, y)
            z = logsig(x) .* tansig(y);
        end

        function dz = gradient_f(x, y)
            dz = [logsig('dn', x) .* tanh(y), logsig(x) .* tansig('dn', y)];
        end


        function [U_x, U_y, U_b, L_x, L_y, L_b, zmax, zmin] = getMultiConstraints(xl, xu, hl, hu)
            u_ = LogsigXTansig.u_;
            l_ = LogsigXTansig.l_;
        
            x_ = LogsigXTansig.x_;
            y_ = LogsigXTansig.y_;
            z_ = LogsigXTansig.z_;
            dzx_ = LogsigXTansig.dzx_;
            dzy_ = LogsigXTansig.dzy_;
            iux_ = LogsigXTansig.iux_;
            iuy_ = LogsigXTansig.iuy_;
            ilx_ = LogsigXTansig.ilx_;
            ily_ = LogsigXTansig.ily_;
            
            z_max = LogsigXTansig.z_max;
            z_min = LogsigXTansig.z_min;
            
            num_of_points = LogsigXTansig.num_of_points;

            n = length(xl);

            pz = nan(num_of_points, ily_, n);

            pz(1, x_:y_, :) = [xl'; hl'];
            pz(2, x_:y_, :) = [xu'; hu'];
            pz(3, x_:y_, :) = [xu'; hl'];
            pz(4, x_:y_, :) = [xl'; hu'];

            pz(:, z_, :) = LogsigXTansig.f(pz(:, x_, :), pz(:, y_, :));
            pz(:, dzx_:dzy_, :) = LogsigXTansig.gradient_f(pz(:, x_, :), pz(:, y_, :));

            % sort z-coordinate points in ascending order
            [~, z_sorted_i] = sort(pz(:, z_, :, :));
            pzs = pz(z_sorted_i(:,1), : , 1:num_of_points);           

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %                       Upper bound case
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            su_max2min_i = nan(1,n);
            % upper bound case for finding slopes from global maximum to other points
%             for i = z_max-1:-1:z_min
%                 z_max_range = pzs(z_max, z_, :) - pzs(i, z_, :) % slope in z coordinate
%                 x_max_range = pzs(z_max, x_, :) - pzs(i, x_, :)
%                 y_max_range = pzs(z_max, y_, :) - pzs(i, y_, :)
%                 pzs(i, iux_, :) = z_max_range ./ x_max_range % slope in x coordinate
%                 pzs(i, iuy_, :) = z_max_range ./ y_max_range % slope in y coordinate
% 
%                 su_max2min_i = (x_max_range ~= 0 & y_max_range ~= 0);
%                 pzs(su_max2min_i, iux_, :) = pzs(su_max2min_i, iux_, :) ./ 2.0;
%                 pzs(su_max2min_i, iuy_, :) = pzs(su_max2min_i, iuy_, :) ./ 2.0;
%             end

            z_max_range = pzs(z_max, z_, :) - pzs(z_max-1:-1:z_min, z_, :); % slope in z coordinate
            x_max_range = pzs(z_max, x_, :) - pzs(z_max-1:-1:z_min, x_, :);
            y_max_range = pzs(z_max, y_, :) - pzs(z_max-1:-1:z_min, y_, :);
            pzs(z_max-1:-1:z_min, iux_, :) = z_max_range ./ x_max_range; % slope in x coordinate
            pzs(z_max-1:-1:z_min, iuy_, :) = z_max_range ./ y_max_range; % slope in y coordinate
% 
%             su_max2min_i = find(x_max_range ~= 0 & y_max_range ~= 0)
%             pzs(su_max2min_i, iux_, 1:num_of_points) = pzs(su_max2min_i, iux_, 1:num_of_points) ./ 2.0;
%             pzs(su_max2min_i, iuy_, 1:num_of_points) = pzs(su_max2min_i, iuy_, 1:num_of_points) ./ 2.0;

            tan_i_ = 1;
            int_max2min_i_ = su_max2min_i + tan_i_;
            %% finding minimal slopes among tangent lines and intersection lines.

            u_slope = cell(1,2);
            u_min_slope = nan(1, 2);
            u_min_slope_i = nan(1, 2);
            
            % min x-axis slopes
            % check (gradient_f) tangent line and intersection lines from global maximum
            u_slope{x_} = [pzs(z_max, dzx_), pzs(z_min : z_max-1, iux_)'];
            [u_min_slope(x_), u_min_slope_i(x_)] = min(abs(u_slope{x_}));

            % min y-axis slopes
            % check (gradient_f) tangent line and intersection lines from global maximum
            u_slope{y_} = [pzs(z_max, dzy_), pzs(z_min:z_max-1, iuy_)'];
            [u_min_slope(y_), u_min_slope_i(y_)] = min(abs(u_slope{y_}));
            
            % special case: slope of one coordinate is greater than 1/2 of coordinate 
            % of intersection line between  global maxima and global minima, while
            % other slope is smaller
            % Let c = slope of intersection line between 
            %   global maximum and global minimum
            if (abs(u_min_slope(x_)) == abs(pzs(su_max2min_i, iux_))) && (abs(u_min_slope(y_)) < abs(pzs(su_max2min_i, iuy_)))
                % for case in which
                %   x-axis slope == 0.5 * dc/dx and
                %   y-axis slope <  0.5 * dc/dy
                %   ==> there is a possibility that x-axis can be greater than 0.5 * dc/dx
                u_slope{x_}(u_min_slope_i(x_)) = u_slope{x_}(u_min_slope_i(x_)) * 2.0;
                u_min_slope(x_) = min(abs(u_slope{x_}));

                % In x-axis, if optimal slope is not tangent line
                if u_min_slope(x_) ~= u_slope{x_}(tan_i_)
                    px = LogsigXTansig.get_pointY(pzs, u_);
                    % get a point that has the same y-axis value as the global maximum but
                    % different x-axis value to check tangent slope on y-axis => px
                    px(z_) = pzs(z_max, z_) - u_min_slope(x_)*(xu-xl);
                    
                    % a point opposite to global maximum on x-axis and y-axis
                    po_z = pzs(su_max2min_i, z_);
                    
                    % intersection line on y-axis (pxy) between two local points
                    px_inter_y = ( px(z_) - po_z ) / (hu - hl);
                    
                    % compare x-axis slopes with the x-axis slope in the opposite face
                    u_min_slope(y_) = min(abs([u_min_slope(y_), px_inter_y, px(dzy_)]));
                   
                else
                    % get a point that has the same x-axis value as the global maximum but
                    % different y-axis value to check tangent slope on x-axis => py
                    py = LogsigXTansig.get_pointY(pzs, u_);
            
                    % a point opposite to global maximum on x-axis and y-axis
                    po_z = pzs(su_max2min_i, z_);
                    new_z = pzs(z_max, z_) - u_min_slope(x_)*(xu-xl) - u_min_slope(y_)*(hu-hl);
                    
                    % po(z_) = pzs(z_max, z_) - u_min_slope(x_)*(xu-xl) - u_min_slope(y_)*(hu-hl)
                    if new_z < po_z
                        z_range = po_z - pzs(z_max, z_);
                        x_slope = -(z_range + u_min_slope(y_)*(hu-hl))/(xu-xl);
                        u_min_slope(x_) = min(abs([u_min_slope(x_), x_slope,  py(dzx_)]));
                    else
                        u_min_slope(x_) = min(abs([u_min_slope(x_), py(dzx_)]));
                    end
                end

            elseif (abs(u_min_slope(x_)) < abs(pzs(su_max2min_i, iux_))) && (abs(u_min_slope(y_)) == abs(pzs(su_max2min_i, iuy_)))
                % for case in which
                %   x-axis slope <  0.5 * dc/dx and
                %   y-axis slope == 0.5 * dc/dy
                %   ==> there is a possibility that y-axis can be greater than 0.5 * dc/dy
                u_slope{y_}(u_min_slope_i(y_)) = u_slope{y_}(u_min_slope_i(y_)) * 2.0;
                u_min_slope(y_) = min(abs(u_slope{y_}));
                
                % In y-axis, if optimal slope is not tangent line
                if u_min_slope(y_) ~= u_slope{y_}(tan_i_)
                    py = LogsigXTansig.get_pointY(pzs, u_);
                    % get a point that has the same x-axis value as the global maximum but
                    % different y-axis value to check tangent slope on x-axis => py
                    py(z_) = pzs(z_max, z_) - u_min_slope(y_)*(hu-hl);
            
                    % a point opposite to global maximum on x-axis and y-axis
                    po_z = pzs(su_max2min_i, z_);
                
                    % intersection line on x-axis (pyx) between two local points
                    py_inter_x = ( py(z_) - po_z ) / (xu - xl);
                
                    % compare x-axis slopes with the x-axis slope in the opposite face
                    u_min_slope(x_) = min(abs([u_min_slope(x_), py_inter_x, py(dzx_)]));
                else
                    % get a point that has the same y-axis value as the global maximum but
                    % different x-axis value to check tangent slope on y-axis => px
                    px = LogsigXTansig.get_pointX(pzs, u_);
            
                    % a point opposite to global maximum on x-axis and y-axis
                    po_z = pzs(su_max2min_i, z_);
                    new_z = pzs(z_max, z_) - u_min_slope(x_)*(xu-xl) - u_min_slope(y_)*(hu-hl);
                    
                    % po(z_) = pzs(z_max, z_) - u_min_slope(x_)*(xu-xl) - u_min_slope(y_)*(hu-hl)
                    if new_z < po_z
                        z_range = po_z - pzs(z_max, z_);
                        y_slope = -(z_range + u_min_slope(x_)*(xu-xl))/(hu-hl);
                        u_min_slope(y_) = min(abs([u_min_slope(y_), y_slope, px(dzy_)]));
                    else
                        u_min_slope(y_) = min(abs([u_min_slope(y_), px(y_)]));
                    end
                end
            
            % if optimal slope in x-axis is an interslection line between global
            % maximum and a local point in x-axis
            elseif (u_min_slope_i(x_) ~= tan_i_) && (u_min_slope_i(x_) ~= int_max2min_i_)
                px = LogsigXTansig.get_pointX(pzs, u_);
                u_min_slope(y_) = min(abs([u_min_slope(y_), px(dzy_)]));

            % if optimal slope in y-axis is an interslection line between global
            % maximum and a local point in y-axis
            elseif (u_min_slope_i(y_) ~= tan_i_) && (u_min_slope_i(y_) ~= int_max2min_i_)
                py = LogsigXTansig.get_pointY(pzs, u_);
                u_min_slope(x_) = min(abs([u_min_slope(x_), py(dzx_)]));
            end

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %                       Lower bound case
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            sl_min2max_i = nan;
            % upper bound case for finding slopes from global maximum to other points
            for i = z_min+1:z_max
                z_min_range = pzs(i, z_) - pzs(z_min, z_); % slope in z coordinate
                x_min_range = pzs(i, x_) - pzs(z_min, x_);
                y_min_range = pzs(i, y_) - pzs(z_min, y_);
                pzs(i, ilx_) = z_min_range / x_min_range; % slope in x coordinate
                pzs(i, ily_) = z_min_range / y_min_range; % slope in y coordinate
                if x_min_range ~= 0 && y_min_range ~= 0
                    pzs(i, ilx_) = pzs(i, ilx_) / 2.0;
                    pzs(i, ily_) = pzs(i, ily_) / 2.0;
                    sl_min2max_i = i;
                end
            end
            
            tan_i_ = 1;
            int_min2max_i_ = sl_min2max_i;
            %% finding minimal slopes among tangent lines and intersection lines.
            l_slope = cell(1,2);
            l_min_slope = nan(1, 2);
            l_min_slope_i = nan(1, 2);
            % min x-coordinate slopes; check (gradient_f) tangent line and intersection
            % lines from global maximum
            l_slope{x_} = [pzs(z_min,dzx_), pzs(z_min+1:z_max, ilx_)'];
            [l_min_slope(x_), l_min_slope_i(x_)] = min(abs(l_slope{x_}));
            
            % min y-coordinate slopes; check (gradient_f) tangent line and intersection 
            % lines from global maximum
            l_slope{y_} = [pzs(z_min,dzy_), pzs(z_min+1:z_max, ily_)'];
            [l_min_slope(y_), l_min_slope_i(y_)] = min(abs(l_slope{y_}));
            
            % special case: slope of one coordinate is greater than 1/2 of coordinate 
            % of intersection line between  global maxima and global minima, while
            % other slope is smaller
            % Let c = slope of intersection line between 
            %   global maximum and global minimum
            if (  abs(l_min_slope(x_)) == abs(pzs(sl_min2max_i, ilx_))  ) && (  abs(l_min_slope(y_)) < abs(pzs(sl_min2max_i, ily_))  )
                % for case in which
                %   x-axis slope == 0.5 * dc/dx and
                %   y-axis slope <  0.5 * dc/dy
                %   ==> there is a possibility that x-axis can be greater than 0.5 * dc/dx
            %     l_slope{x_}(sl_min2max_i) = l_slope{x_}(sl_min2max_i) * 2.0;
                l_slope{x_}(l_min_slope_i(x_)) = l_slope{x_}(l_min_slope_i(x_)) * 2.0;
                [l_min_slope(x_), l_min_slope_i(x_)] = min(abs(l_slope{x_}));
                
                % In x-axis, if optimal slope is not tangent line
                if l_min_slope(x_) ~= l_slope{x_}(tan_i_)
                    px = LogsigXTansig.get_pointY(pzs, l_);
                    % get a point that has the same y-axis value as the global maximum but
                    % different x-axis value to check tangent slope on y-axis => px
                    px(z_) = pzs(z_min, z_) + l_min_slope(x_)*(xu-xl);
                
                    % a point opposite to global maximum on x-axis and y-axis
                    po = pzs(sl_min2max_i, x_:z_);
                    
                    % intersection line on y-axis (pxy) between two local points
                    px_inter_y = ( px(z_) - po(z_) ) / (hu - hl);
                    
                    % compare x-axis slopes with the x-axis slope in the opposite face
                    l_min_slope(y_) = min(abs([l_min_slope(y_), px_inter_y, px(dzy_)]));
                else
                    % get a point that has the same x-axis value as the global maximum but
                    % different y-axis value to check tangent slope on x-axis => py
                    py = LogsigXTansig.get_pointY(pzs, u_);
            
                    % a point opposite to global maximum on x-axis and y-axis
                    po = pzs(sl_min2max_i, x_:z_);
                    new_z = pzs(z_min, z_) + l_min_slope(x_)*(xu-xl) + l_min_slope(y_)*(hu-hl);
                    
                    % po(z_) = pzs(z_min, z_) + l_min_slope(x_)*(xu-xl) + l_min_slope(y_)*(hu-hl)
                    if new_z < po(z_)
                        z_range = po(z_) - pzs(z_min, z_);
                        x_slope = -(z_range + l_min_slope(y_)*(hu-hl))/(xu-xl);
                        l_min_slope(x_) = min(abs([l_min_slope(x_), x_slope, py(dzx_)]));
                    else
                        l_min_slope(x_) = min(abs([l_min_slope(x_), py(x_)]));
                    end
                end
            
            elseif (  abs(l_min_slope(x_)) < abs(pzs(sl_min2max_i, ilx_))  ) && (  abs(l_min_slope(y_)) == abs(pzs(sl_min2max_i, ily_))  )
                % for case in which
                %   x-axis slope <  0.5 * dc/dx and
                %   y-axis slope == 0.5 * dc/dy
                %   ==> there is a possibility that y-axis can be greater than 0.5 * dc/dy
                l_slope{y_}(l_min_slope_i(y_)) = l_slope{y_}(l_min_slope_i(y_)) * 2.0;
                [l_min_slope(y_), l_min_slope_i(y_)] = min(abs(l_slope{y_}));
                
                % In y-axis, if optimal slope is not tangent line
                if l_min_slope(y_) ~= l_slope{y_}(tan_i_)
                    py = LogsigXTansig.get_pointY(pzs, l_);

                    % get a point that has the same x-axis value as the global maximum but
                    % different y-axis value to check tangent slope on x-axis => py
                    py(z_) = pzs(z_min, z_) + l_min_slope(y_)*(hu-hl);
            
                    % a point opposite to global maximum on x-axis and y-axis
                    po = pzs(sl_min2max_i, x_:z_);
                
                    % intersection line on x-axis (pyx) between two local points
                    py_inter_x = ( py(z_) - po(z_) ) / (xu - xl);
                
                    % compare x-axis slopes with the x-axis slope in the opposite face 
                    l_min_slope(x_) = min(abs([l_min_slope(x_), py_inter_x, py(dzx_)])); 
                else
                    % get a point that has the same y-axis value as the global maximum but
                    % different x-axis value to check tangent slope on y-axis => px
                    px = LogsigXTansig.get_pointX(pzs, l_);
            
                    % a point opposite to global maximum on x-axis and y-axis
                    po = pzs(sl_min2max_i, x_:z_);
                    new_z = pzs(z_min, z_) + l_min_slope(x_)*(xu-xl) + l_min_slope(y_)*(hu-hl);
                    
                    % po(z_) = pzs(z_max, z_) - l_min_slope(x_)*(xu-xl) - l_min_slope(y_)*(hu-hl)
                    if new_z > po(z_)
                        z_range = po(z_) - pzs(z_min, z_) ;
                        y_slope = -(z_range + l_min_slope(x_)*(xu-xl))/(hu-hl);
                        l_min_slope(y_) = min(abs([l_min_slope(y_), y_slope, px(dzy_)]));
                    else
                        l_min_slope(y_) = min(abs([l_min_slope(y_), px(dzy_)]));
                    end
                end
            
            % if optimal slope in x-axis is an interslection line between global
            % maximum and a local point in x-axis
            elseif (l_min_slope_i(x_) ~= tan_i_) && (l_min_slope_i(x_) ~= int_min2max_i_)
                px = LogsigXTansig.get_pointX(pzs, l_);
                l_min_slope(y_) = min(abs([l_min_slope(y_), px(dzy_)]));
            
            % if optimal slope in y-axis is an interslection line between global
            % maximum and a local point in y-axis
            elseif (l_min_slope_i(y_) ~= tan_i_) && (l_min_slope_i(y_) ~= int_min2max_i_)
                py = LogsigXTansig.get_pointY(pzs, l_);
                l_min_slope(x_) = min(abs([l_min_slope(x_), py(dzx_)]));
            end

            % upper linear constraintss on x-coordinate and y-coordinate
            if z_sorted_i(z_max) == 1
                U_x = u_min_slope(LogsigXTansig.x_);
                U_y = u_min_slope(y_);
            elseif z_sorted_i(z_max) == 2
                U_x = -u_min_slope(x_);
                U_y = -u_min_slope(y_);
            elseif z_sorted_i(z_max) == 3
                U_x = -u_min_slope(x_);
                U_y = u_min_slope(y_);
            elseif z_sorted_i(z_max) == 4
                U_x = u_min_slope(x_);
                U_y = -u_min_slope(y_);
            else
                error('ERROR: unknown point');
            end
            U_b = pzs(z_max,z_) + U_x*pzs(z_max,x_) + U_y*pzs(z_max,y_);

            % lower linear constraints on x-coordinate and y-coordinate
            if z_sorted_i(z_min) == 1
                L_x = -l_min_slope(x_);
                L_y = -l_min_slope(y_);
            elseif z_sorted_i(z_min) == 2
                L_x = l_min_slope(x_);
                L_y = l_min_slope(y_);
            elseif z_sorted_i(z_min) == 3
                L_x = l_min_slope(x_);
                L_y = -l_min_slope(y_);
            elseif z_sorted_i(z_min) == 4
                L_x = -l_min_slope(x_);
                L_y = l_min_slope(y_);
            else
                error('ERROR: unknown point');
            end
            L_b = pzs(z_min, z_) + L_x*pzs(z_min,x_) + L_y*pzs(z_min,y_);
            
            zmax = pzs(z_max, z_);
            zmin = pzs(z_min, z_);
        end


        function [U_x, U_y, U_b, L_x, L_y, L_b, zmax, zmin] = getConstraints(xl, xu, hl, hu)
            u_ = LogsigXTansig.u_;
            l_ = LogsigXTansig.l_;
        
            x_ = LogsigXTansig.x_;
            y_ = LogsigXTansig.y_;
            z_ = LogsigXTansig.z_;
            dzx_ = LogsigXTansig.dzx_;
            dzy_ = LogsigXTansig.dzy_;
            iux_ = LogsigXTansig.iux_;
            iuy_ = LogsigXTansig.iuy_;
            ilx_ = LogsigXTansig.ilx_;
            ily_ = LogsigXTansig.ily_;
            
            z_max = LogsigXTansig.z_max;
            z_min = LogsigXTansig.z_min;
            
            num_of_points = LogsigXTansig.num_of_points;

            pz = zeros(num_of_points, num_of_points);
            pz(1, x_:y_) = [xl, hl];
            pz(2, x_:y_) = [xu, hu];
            pz(3, x_:y_) = [xu, hl];
            pz(4, x_:y_) = [xl, hu];

            pz(:, z_) = LogsigXTansig.f(pz(:, x_), pz(:, y_));
            pz(:, dzx_:dzy_) = LogsigXTansig.gradient_f(pz(:, x_), pz(:, y_));

            % sort z-coordinate points in ascending order
            [~, z_sorted_i] = sort(pz(:, z_));
            pzs = [pz(z_sorted_i, :), nan(4, 4)];

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %                       Upper bound case
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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

            tan_i_ = 1;
            int_max2min_i_ = su_max2min_i + tan_i_;
            %% finding minimal slopes among tangent lines and intersection lines.

            u_slope = cell(1,2);
            u_min_slope = nan(1, 2);
            u_min_slope_i = nan(1, 2);
            
            u_slope{x_} = [pzs(z_max, dzx_), pzs(z_min : z_max-1, iux_)'];
            [u_min_slope(x_), u_min_slope_i(x_)] = min(abs(u_slope{x_}));
            u_slope{y_} = [pzs(z_max, dzy_), pzs(z_min:z_max-1, iuy_)'];
            [u_min_slope(y_), u_min_slope_i(y_)] = min(abs(u_slope{y_}));
            
            if (abs(u_min_slope(x_)) == abs(pzs(su_max2min_i, iux_))) && (abs(u_min_slope(y_)) < abs(pzs(su_max2min_i, iuy_)))
                u_slope{x_}(u_min_slope_i(x_)) = u_slope{x_}(u_min_slope_i(x_)) * 2.0;
                u_min_slope(x_) = min(abs(u_slope{x_}));

                % In x-axis, if optimal slope is not tangent line
                if u_min_slope(x_) ~= u_slope{x_}(tan_i_)
                    px = LogsigXTansig.get_pointX(pzs, u_);
                    px(z_) = pzs(z_max, z_) - u_min_slope(x_)*(xu-xl);
                    po = pzs(su_max2min_i, x_:z_);
                    
                    px_inter_y = ( px(z_) - po(z_) ) / (hu - hl);
                    u_min_slope(y_) = min(abs([u_min_slope(y_), px_inter_y, px(dzy_)]));
                   
                else
                    py = LogsigXTansig.get_pointY(pzs, u_);
                    po = pzs(su_max2min_i, x_:z_);
                    new_z = pzs(z_max, z_) - u_min_slope(x_)*(xu-xl) - u_min_slope(y_)*(hu-hl);
                    
                    % po(z_) = pzs(z_max, z_) - u_min_slope(x_)*(xu-xl) - u_min_slope(y_)*(hu-hl)
                    if new_z < po(z_)
                        z_range = po(z_) - pzs(z_max, z_);
                        x_slope = -(z_range + u_min_slope(y_)*(hu-hl))/(xu-xl);
                        u_min_slope(x_) = min(abs([u_min_slope(x_), x_slope,  py(dzx_)]));
                    else
                        u_min_slope(x_) = min(abs([u_min_slope(x_), py(dzx_)]));
                    end
                end

            elseif (abs(u_min_slope(x_)) < abs(pzs(su_max2min_i, iux_))) && (abs(u_min_slope(y_)) == abs(pzs(su_max2min_i, iuy_)))
                u_slope{y_}(u_min_slope_i(y_)) = u_slope{y_}(u_min_slope_i(y_)) * 2.0;
                u_min_slope(y_) = min(abs(u_slope{y_}));
                
                % In y-axis, if optimal slope is not tangent line
                if u_min_slope(y_) ~= u_slope{y_}(tan_i_)
                    py = LogsigXTansig.get_pointY(pzs, u_);
                    py(z_) = pzs(z_max, z_) - u_min_slope(y_)*(hu-hl);
                    po = pzs(su_max2min_i, x_:z_);
              
                    py_inter_x = ( py(z_) - po(z_) ) / (xu - xl);
                    u_min_slope(x_) = min(abs([u_min_slope(x_), py_inter_x, py(dzx_)]));
                else
                    px = LogsigXTansig.get_pointX(pzs, u_);
                    po = pzs(su_max2min_i, x_:z_);
                    new_z = pzs(z_max, z_) - u_min_slope(x_)*(xu-xl) - u_min_slope(y_)*(hu-hl);
                    
                    % po(z_) = pzs(z_max, z_) - u_min_slope(x_)*(xu-xl) - u_min_slope(y_)*(hu-hl)
                    if new_z < po(z_)
                        z_range = po(z_) - pzs(z_max, z_);
                        y_slope = -(z_range + u_min_slope(x_)*(xu-xl))/(hu-hl);
                        u_min_slope(y_) = min(abs([u_min_slope(y_), y_slope, px(dzy_)]));
                    else
                        u_min_slope(y_) = min(abs([u_min_slope(y_), px(y_)]));
                    end
                end
            
            elseif (u_min_slope_i(x_) ~= tan_i_) && (u_min_slope_i(x_) ~= int_max2min_i_)
                px = LogsigXTansig.get_pointX(pzs, u_);
                u_min_slope(y_) = min(abs([u_min_slope(y_), px(dzy_)]));

            elseif (u_min_slope_i(y_) ~= tan_i_) && (u_min_slope_i(y_) ~= int_max2min_i_)
                py = LogsigXTansig.get_pointY(pzs, u_);
                u_min_slope(x_) = min(abs([u_min_slope(x_), py(dzx_)]));
            end

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %                       Lower bound case
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            sl_min2max_i = nan;
            % upper bound case for finding slopes from global maximum to other points
            for i = z_min+1:z_max
                z_min_range = pzs(i, z_) - pzs(z_min, z_); % slope in z coordinate
                x_min_range = pzs(i, x_) - pzs(z_min, x_);
                y_min_range = pzs(i, y_) - pzs(z_min, y_);
                pzs(i, ilx_) = z_min_range / x_min_range; % slope in x coordinate
                pzs(i, ily_) = z_min_range / y_min_range; % slope in y coordinate
                if x_min_range ~= 0 && y_min_range ~= 0
                    pzs(i, ilx_) = pzs(i, ilx_) / 2.0;
                    pzs(i, ily_) = pzs(i, ily_) / 2.0;
                    sl_min2max_i = i;
                end
            end
            
            tan_i_ = 1;
            int_min2max_i_ = sl_min2max_i;
            %% finding minimal slopes among tangent lines and intersection lines.
            l_slope = cell(1,2);
            l_min_slope = nan(1, 2);
            l_min_slope_i = nan(1, 2);

            l_slope{x_} = [pzs(z_min,dzx_), pzs(z_min+1:z_max, ilx_)'];
            [l_min_slope(x_), l_min_slope_i(x_)] = min(abs(l_slope{x_}));
            l_slope{y_} = [pzs(z_min,dzy_), pzs(z_min+1:z_max, ily_)'];
            [l_min_slope(y_), l_min_slope_i(y_)] = min(abs(l_slope{y_}));
            
            if (  abs(l_min_slope(x_)) == abs(pzs(sl_min2max_i, ilx_))  ) && (  abs(l_min_slope(y_)) < abs(pzs(sl_min2max_i, ily_))  )
                l_slope{x_}(l_min_slope_i(x_)) = l_slope{x_}(l_min_slope_i(x_)) * 2.0;
                l_min_slope(x_) = min(abs(l_slope{x_}));
                
                % In x-axis, if optimal slope is not tangent line
                if l_min_slope(x_) ~= l_slope{x_}(tan_i_)
                    px = LogsigXTansig.get_pointX(pzs, l_);
                    px(z_) = pzs(z_min, z_) + l_min_slope(x_)*(xu-xl);
                    po = pzs(sl_min2max_i, x_:z_);

                    px_inter_y = ( px(z_) - po(z_) ) / (hu - hl);
                    l_min_slope(y_) = min(abs([l_min_slope(y_), px_inter_y, px(dzy_)]));
                else
                    py = LogsigXTansig.get_pointY(pzs, u_);
                    po = pzs(sl_min2max_i, x_:z_);
                    new_z = pzs(z_min, z_) + l_min_slope(x_)*(xu-xl) + l_min_slope(y_)*(hu-hl);
                    
                    % po(z_) = pzs(z_min, z_) + l_min_slope(x_)*(xu-xl) + l_min_slope(y_)*(hu-hl)
                    if new_z < po(z_)
                        z_range = po(z_) - pzs(z_min, z_);
                        x_slope = -(z_range + l_min_slope(y_)*(hu-hl))/(xu-xl);
                        l_min_slope(x_) = min(abs([l_min_slope(x_), x_slope, py(dzx_)]));
                    else
                        l_min_slope(x_) = min(abs([l_min_slope(x_), py(x_)]));
                    end
                end
            
            elseif (  abs(l_min_slope(x_)) < abs(pzs(sl_min2max_i, ilx_))  ) && (  abs(l_min_slope(y_)) == abs(pzs(sl_min2max_i, ily_))  )
                l_slope{y_}(l_min_slope_i(y_)) = l_slope{y_}(l_min_slope_i(y_)) * 2.0;
                l_min_slope(y_) = min(abs(l_slope{y_}));
                
                % In y-axis, if optimal slope is not tangent line
                if l_min_slope(y_) ~= l_slope{y_}(tan_i_)
                    py = LogsigXTansig.get_pointY(pzs, l_);
                    py(z_) = pzs(z_min, z_) + l_min_slope(y_)*(hu-hl);
                    po = pzs(sl_min2max_i, x_:z_);

                    py_inter_x = ( py(z_) - po(z_) ) / (xu - xl);
                    l_min_slope(x_) = min(abs([l_min_slope(x_), py_inter_x, py(dzx_)])); 
                else
                    px = LogsigXTansig.get_pointX(pzs, l_);
                    po = pzs(sl_min2max_i, x_:z_);
                    new_z = pzs(z_min, z_) + l_min_slope(x_)*(xu-xl) + l_min_slope(y_)*(hu-hl);
                    
                    % po(z_) = pzs(z_max, z_) - l_min_slope(x_)*(xu-xl) - l_min_slope(y_)*(hu-hl)
                    if new_z > po(z_)
                        z_range = po(z_) - pzs(z_min, z_) ;
                        y_slope = -(z_range + l_min_slope(x_)*(xu-xl))/(hu-hl);
                        l_min_slope(y_) = min(abs([l_min_slope(y_), y_slope, px(dzy_)]));
                    else
                        l_min_slope(y_) = min(abs([l_min_slope(y_), px(dzy_)]));
                    end
                end
            
            elseif (l_min_slope_i(x_) ~= tan_i_) && (l_min_slope_i(x_) ~= int_min2max_i_)
                px = LogsigXTansig.get_pointX(pzs, l_);
                l_min_slope(y_) = min(abs([l_min_slope(y_), px(dzy_)]));
            
            elseif (l_min_slope_i(y_) ~= tan_i_) && (l_min_slope_i(y_) ~= int_min2max_i_)
                py = LogsigXTansig.get_pointY(pzs, l_);
                l_min_slope(x_) = min(abs([l_min_slope(x_), py(dzx_)]));
            end

            % upper linear constraintss on x-coordinate and y-coordinate
            if z_sorted_i(z_max) == 1
                U_x = u_min_slope(x_);
                U_y = u_min_slope(y_);
            elseif z_sorted_i(z_max) == 2
                U_x = -u_min_slope(x_);
                U_y = -u_min_slope(y_);
            elseif z_sorted_i(z_max) == 3
                U_x = -u_min_slope(x_);
                U_y = u_min_slope(y_);
            elseif z_sorted_i(z_max) == 4
                U_x = u_min_slope(x_);
                U_y = -u_min_slope(y_);
            else
                error('ERROR: unknown point');
            end
            U_b = pzs(z_max,z_) + U_x*pzs(z_max,x_) + U_y*pzs(z_max,y_);

            % lower linear constraints on x-coordinate and y-coordinate
            if z_sorted_i(z_min) == 1
                L_x = -l_min_slope(x_);
                L_y = -l_min_slope(y_);
            elseif z_sorted_i(z_min) == 2
                L_x = l_min_slope(x_);
                L_y = l_min_slope(y_);
            elseif z_sorted_i(z_min) == 3
                L_x = l_min_slope(x_);
                L_y = -l_min_slope(y_);
            elseif z_sorted_i(z_min) == 4
                L_x = -l_min_slope(x_);
                L_y = l_min_slope(y_);
            else
                error('ERROR: unknown point');
            end
            L_b = pzs(z_min, z_) + L_x*pzs(z_min,x_) + L_y*pzs(z_min,y_);
            
            zmax = pzs(z_max, z_);
            zmin = pzs(z_min, z_);
        end


        function p = get_pointX(a, type)
            if type == LogsigXTansig.u_
                for i = LogsigXTansig.z_max-1:-1:LogsigXTansig.z_min
                    if a(i, LogsigXTansig.iuy_) == inf
                        p = a(i, LogsigXTansig.x_:LogsigXTansig.dzy_);
                        break;
                    end
                end
            elseif type == LogsigXTansig.l_
                for i = LogsigXTansig.z_min+1:LogsigXTansig.z_max
                    if a(i, LogsigXTansig.ily_) == inf
                        p = a(i, LogsigXTansig.x_:LogsigXTansig.dzy_);
                        break;
                    end
                end
            end
        end
        
        function p = get_pointY(a, type)
            if type == LogsigXTansig.u_
                for i = LogsigXTansig.z_max-1:-1:LogsigXTansig.z_min
                    if abs(a(i, LogsigXTansig.iux_)) == inf
                        p = a(i, LogsigXTansig.x_:LogsigXTansig.dzy_);
                        break;
                    end
                end
            elseif type == LogsigXTansig.l_
                for i = LogsigXTansig.z_min+1:LogsigXTansig.z_max
                    if a(i, LogsigXTansig.ilx_) == inf
                        p = a(i, LogsigXTansig.x_:LogsigXTansig.dzy_);
                        break;
                    end
                end
            end
        end

        % main method
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%% Star %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function S = reach_star_approx(varargin)
            % author: Sung Woo Choi
            % date: 08/24/2022

            switch nargin
                case 6
                    X = varargin{1};
                    H = varargin{2};
                    method = varargin{3};
                    relaxFactor = varargin{4};
                    dis_opt = varargin{5}; % display option
                    lp_solver = varargin{6};
                case 5
                    X = varargin{1};
                    H = varargin{2};
                    method = varargin{3};
                    relaxFactor = varargin{4};
                    dis_opt = varargin{5}; % display option
                    lp_solver = 'glpk';
                case 4
                    X = varargin{1};
                    H = varargin{2};
                    method = varargin{3};
                    relaxFactor = varargin{4};
                    dis_opt = [];
                    lp_solver = 'glpk';
                case 3
                    X = varargin{1};
                    H = varargin{2};
                    method = varargin{3};
                    relaxFactor = 0; % for relaxed approx-star method
                    dis_opt = [];
                    lp_solver = 'glpk';
                case 2
                    X = varargin{1};
                    H = varargin{2};
                    method = 'approx-star';
                    relaxFactor = 0; % for relaxed approx-star method
                    dis_opt = [];
                    lp_solver = 'glpk';
                otherwise
                    error('Invalid number of input arguments, should be 2, 3, 4, 5, or 6');
            end

            if ~(isa(X, 'Star') && isa(H, 'Star'))
                error('Input sets are not star set');
            end

            if X.dim ~= H.dim
                error('Dimension of input sets does not match');
            end

            if ~strcmp(method, 'approx-star')
                error('Unknown reachability method');
            end

            if relaxFactor == 0
                n = X.dim;
                S = X.Sum(H);
                for i = 1:n
                    S = LogsigXTansig.stepLogsigXTansig(S, X, H, i, dis_opt, lp_solver);
                end
%                 S = LogsigXTansig.multiStepLogsigXTansig(X, H, dis_opt, lp_solver);
            else
                n = X.dim;
                S = X.Sum(H);
                for i = 1:n
                    S = LogsigXTansig.relaxedStepLogsigXTansig(S, X, H, i, relaxFactor, dis_opt, lp_solver);
                end
%                 S = LogsigXTansig.relaxedMultiStepLogsigXTansig(X, H, relaxFactor, dis_opt, lp_solver);
            end
        end

        

        function S = stepLogsigXTansig(I, X, H, index, dis_opt, lp_solver)
            % @X: input star set (input state)
            % @H: input star set (hidden state)
            % @i: index of the neuron

            if strcmp(lp_solver, 'glpk') || strcmp(lp_solver, 'linprog')

                if strcmp(dis_opt, 'display')
                    fprintf('\nComputing lower-bounds: ');
                end
                xl = X.getMins(index, [], dis_opt, lp_solver);
                hl = H.getMins(index, [], dis_opt, lp_solver);
                if strcmp(dis_opt, 'display')
                    fprintf('\nComputing upper-bounds: ');  
                end
                xu = X.getMaxs(index, [], dis_opt, lp_solver);
                hu = H.getMaxs(index, [], dis_opt, lp_solver);


            elseif strcmp(lp_solver, 'estimate')

                if strcmp(dis_opt, 'display')
                    fprintf('\nComputing estimate lower- and upper-bounds: ');
                end
                [xl, xu] = X.estimateRange(index);
                [hl, hu] = H.estimateRange(index); 

            end
            
            if xl == xu && hl == hu
                zl = LogsigXTansig.f(xl, hl);
                
                new_V = I.V;
                new_V(index, 1:I.nVar+1) = 0;
                new_V(index, 1) = zl;
                S = Star(new_V, I.C, I.d, I.predicate_lb, I.predicate_ub);

            elseif xl == xu && hl ~= hu
                % logisg(xl) * tansig(H)

                zl = tansig(hl);
                zu = tansig(hu);
                dzl = tansig('dn', hl);
                dzu = tansig('dn', hu);
                
                nI = I.nVar + 1;
                if hl >= 0
                    % constraint 1: z <= tansig'(hl) * (h - hl) + tansig(hl)
                    C1 = [-dzl*I.V(index, 2:nI), 1];
                    d1 = dzl*(H.V(index, 1) - hl) + zl;  

                    % constraint 2: z <= tansig'(hu) * (h - hu) + tansig(hu)
                    C2 = [-dzu*I.V(index, 2:nI), 1];
                    d2 = dzu*(H.V(index, 1) - hu) + zu;

                    % constraint 3: z >= (tansig(hu) - tansig(hl)) * (h - hl) / (hu - hl) + tansig(hl);
                    a = (zu - zl)/(hu - hl);
                    C3 = [a*I.V(index, 2:nI), -1];
                    d3 = -a*(H.V(index, 1) - hl) - zl;

                elseif hu <= 0
                    % wrong constraints
                    % constraint 1: z >= tansig'(hl) * (h - hl) + tansig(hl)
                    C1 = [dzl*I.V(index, 2:nI), -1];
                    d1 = -dzl*(H.V(index, 1) - hl) - zl;

                    % constraint 2: z >= tansig'(hu) * (h - hu) + tansig(hu)
                    C2 = [dzu*I.V(index, 2:nI), -1];
                    d2 = -dzu*(H.V(index, 1) - hu) - zu;

                    % constraint 3: z <= (tansig(hu) - tansig(hl)) * (h - hl) / (hu - hl) + tansig(hl);
                    a = (zu - zl)/(hu - hl);
                    C3 = [-a*I.V(index, 2:nI), 1];
                    d3 = a*(H.V(index, 1) - hl) + zl;

                elseif hl < 0 && hu > 0
                    dmin = min(dzl, dzu);
        
                    % constraint 1: z >= dmin * (h - hl) + tansig(hl)
                    C1 = [dmin*I.V(index, 2:nI), -1];
                    d1 = -dmin*(H.V(index, 1) - hl) - zl; 

                    % constraint 2: z <= dmin * (h - hu) + tansig(hu)
                    C2 = [-dmin*I.V(index, 2:nI), 1];
                    d2 = dmin*(H.V(index, 1) - hu) + zu;
                    
                    gu_x = (zu - dmin * hu) / (1 - dmin); % upper intersect x
                    gu_y = gu_x;% upper intersect y
                    gl_x = (zl - dmin * hl) / (1 - dmin); % lower intersect x
                    gl_y = gl_x; % lower intersect y

                    mu = (zl - gu_y) / (hl - gu_x);
                    ml = (zu - gl_y) / (hu - gl_x);

                    % constraint 3: z[index] >= ml * h[index] - ml*hu + zu
                    C31 = [ml*I.V(index, 2:nI), -1];
                    d31 = -ml*(H.V(index, 1) - hu) - zu;

                    % constraint 4: z[index] <= mu * h[index] - mu*hl + zl
                    C32 = [-mu*I.V(index, 2:nI), 1];
                    d32 = mu*(H.V(index, 1) - hl) + zl;

                    C3 = [C31; C32];
                    d3 = [d31; d32];

                end

                m = size(I.C, 1);
                C0 = [I.C zeros(m, 1)];
                d0 = I.d;
                new_C = [C0; C1; C2; C3];
                new_d = [d0; d1; d2; d3];
                new_V = [I.V zeros(I.dim, 1)];
                new_V(index, :) = zeros(1, nI+1);
                new_V(index, nI+1) = logsig(xl);

                new_predicate_lb = [I.predicate_lb; zl];
                new_predicate_ub = [I.predicate_ub; zu];

                S = Star(new_V, new_C, new_d, new_predicate_lb, new_predicate_ub);

            elseif xl ~= xu &&  hl == hu
                % logsig(X) * tansig(hl)

                zl = logsig(xl);
                zu = logsig(xu);
                dzl = logsig('dn', xl);
                dzu = logsig('dn', xu);

                nI = I.nVar + 1;
                if xl >= 0
                    % z is convex when xl >= 0

                    % constraint 1: z <= hl * ( logsig'(xl) * (x - xl) + logsig(xl) )
                    C1 = [-dzl*I.V(index, 2:nI), 1];
                    d1 = dzl*(X.V(index, 1) - xl) + zl;

                    % constarint 2: z <= hl * ( logsig'(xu) * (x - xu) + logsig(xu) )
                    C2 = [-dzu*I.V(index, 2:nI), 1];
                    d2 = dzu*(X.V(index, 1) - xu) + zu;

                    % constraint 3: z >= hl * (logsig(xu) - logsig(xl)) * (x - xl) / (xu - lx) + logsig(xl);
                    a = (zu - zl)/(xu - xl);
                    C3 = [a*I.V(index, 2:nI), -1];
                    d3 = -a*(X.V(index, 1) - xl) - zl;

                elseif xu <= 0
                    % z is concave when xu <= 0
                    
                    % constraint 1: z >= logsig'(xl) * (x - xl) + logsig(xl)
                    C1 = [dzl*I.V(index, 2:nI), -1];
                    d1 = -dzl*(X.V(index, 1) - xl) - zl;

                    % constraint 2: z >= logsig'(xu) * (x - xu) + logsig(xu)
                    C2 = [dzu*I.V(index, 2:nI), -1];
                    d2 = -dzu*(X.V(index, 1) - xu) - zu;

                    % constraint 3: z <= (logsig(xu) - logsig(xl)) * (x - xl) / (xu - xl) + logsig(xl);
                    a = (zu - zl)/(xu - xl);
                    C3 = [-a*I.V(index, 2:nI), 1];
                    d3 = a*(X.V(index, 1) - xl) + zl;
                
                elseif xl < 0 && xu > 0
                    dmin = min(dzl, dzu);

                    % constraint 1: z >= hl * ( dmin * (x - xl) + tansig(xl) )
                    C1 = [dmin*I.V(index, 2:nI), -1];
                    d1 = -dmin*(X.V(index, 1) - xl) - zl; 

                    % constraint 2: z <= hl * ( dmin * (x - xu) + tansig(xu) )
                    C2 = [-dmin*I.V(index, 2:nI), 1];
                    d2 = dmin*(X.V(index, 1) - xu) + zu;

                    gu_x = (zu - dmin * xu - 0.5) / (0.25 - dmin); % upper intersect x
                    gu_y = 0.25 * gu_x + 0.5;% upper intersect y
                    gl_x = (zl - dmin * xl - 0.5) / (0.25 - dmin); % lower intersect x
                    gl_y = 0.25 * gl_x + 0.5; % lower intersect y
                    
                    mu = (zl - gu_y) /(xl - gu_x);
                    ml = (zu - gl_y) /(xu - gl_x);

                    % constraint 3: z[index] >= ml * (x[index] - xu) + zu
                    C31 = [ml*I.V(index, 2:nI), -1];
                    d31 = -ml*(X.V(index, 1) - xu) - zu;

                    % constraint 4: z[index] <= mu * (x[index] - xl) + zl
                    C32 = [-mu*I.V(index, 2:nI), 1];
                    d32 = mu*(X.V(index, 1) - xl) + zl;

                    C3 = [C31; C32];
                    d3 = [d31; d32];
                end

                m = size(I.C, 1);
                C0 = [I.C zeros(m, 1)];
                d0 = I.d;
                new_C = [C0; C1; C2; C3]; %empty set is caused by C3, d3 (may be it is because of wrong constraints)
                new_d = [d0; d1; d2; d3];
                new_V = [I.V zeros(I.dim, 1)];
                new_V(index, :) = zeros(1, nI+1);
                new_V(index, nI+1) = tansig(hl);

                new_predicate_lb = [I.predicate_lb; zl];
                new_predicate_ub = [I.predicate_ub; zu];

                S = Star(new_V, new_C, new_d, new_predicate_lb, new_predicate_ub);

            else
                %logsig(x)*tansig(y)
                % X star set
                % H star set

                [Ux, Uy, Ub, Lx, Ly, Lb, zu, zl] =  LogsigXTansig.getConstraints(xl, xu, hl, hu);

                nX = X.nVar + 1;
                nH = H.nVar + 1;
                o = I.nVar - X.nVar - H.nVar;

                C1 = [-Lx*X.V(index, 2:nX), -Ly*H.V(index, 2:nH), zeros(1, o), -1];
                d1 = Lx*X.V(index, 1) + Ly*H.V(index, 1) - Lb;

                C2 = [Ux*X.V(index, 2:nX), Uy*H.V(index, 2:nH), zeros(1, o), 1];
                d2 = -(Ux*X.V(index, 1) + Uy*H.V(index, 1)) + Ub;

                nI = I.nVar + 1;
                m = size(I.C, 1);
                C0 = [I.C zeros(m, 1)];
                d0 = I.d;
                new_C = [C0; C1; C2];
                new_d = [d0; d1; d2];

                new_V = [I.V zeros(I.dim, 1)];
                new_V(index, :) = zeros(1, nI+1);
                new_V(index, nI+1) = 1;

                new_predicate_lb = [I.predicate_lb; zl];
                new_predicate_ub = [I.predicate_ub; zu];

                S = Star(new_V, new_C, new_d, new_predicate_lb, new_predicate_ub);
            end
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%% Star END %%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%% SparseStar %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
       

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%% SparseStar END %%%%%%%%%%%%%%%%%%%%%%%%%%%
    end

    methods(Static) % main reach method

        % main function for reachability analysis
        function R = reach(varargin)
            % author: Sung Woo Choi
            % date: 10/11/2022

            switch nargin
                case 6
                    X = varargin{1};
                    H = varargin{2};
                    method = varargin{3};
                    relaxFactor = varargin{4};
                    dis_opt = varargin{5}; % display option
                    lp_solver = varargin{6};
                case 5
                    X = varargin{1};
                    H = varargin{2};
                    method = varargin{3};
                    relaxFactor = varargin{4};
                    dis_opt = varargin{5}; % display option
                    lp_solver = 'glpk';
                case 4
                    X = varargin{1};
                    H = varargin{2};
                    method = varargin{3};
                    relaxFactor = varargin{4};
                    dis_opt = [];
                    lp_solver = 'glpk';
                case 3
                    X = varargin{1};
                    H = varargin{2};
                    method = varargin{3};
                    relaxFactor = 0; % for relaxed approx-star method
                    dis_opt = [];
                    lp_solver = 'glpk';
                case 2
                    X = varargin{1};
                    H = varargin{2};
                    method = 'approx-star';
                    relaxFactor = 0; % for relaxed approx-star method
                    dis_opt = [];
                    lp_solver = 'glpk';
                otherwise
                    error('Invalid number of input arguments, should be 2, 3, 4, 5, or 6');
            end

            if strcmp(method, 'approx-star')

                R = LogsigXTansig.reach_star_approx(X, H, method, relaxFactor, dis_opt, lp_solver);

%             elseif strcmp(method, 'approx-sparse-star')
%                 
%                 R = LogsigXTansig.reach_sparse_star_approx(X, H, method, relaxFactor, dis_opt, lp_solver);

            else
                error('Unknown or unsupported reachability method for layer with LogSig activation function');
            end
        end
    end
end

