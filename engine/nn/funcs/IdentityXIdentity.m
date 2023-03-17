classdef IdentityXIdentity
    % IdentityXIdentity class contains method for rechability analysis for
    % Layer with x * y function
    % author: Sung Woo Choi
    % date: 10/10/2022
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
            z = x .* y;
        end

        function dz = gradient_f(x, y)
            dz = [x, y];
        end

        function [U_x, U_y, U_b, L_x, L_y, L_b, zmax, zmin] = getMultiConstraints(xl, xu, hl, hu)
            u_ = IdentityXIdentity.u_;
            l_ = IdentityXIdentity.l_;
        
            x_ = IdentityXIdentity.x_;
            y_ = IdentityXIdentity.y_;
            z_ = IdentityXIdentity.z_;
            dzx_ = IdentityXIdentity.dzx_;
            dzy_ = IdentityXIdentity.dzy_;
            iux_ = IdentityXIdentity.iux_;
            iuy_ = IdentityXIdentity.iuy_;
            ilx_ = IdentityXIdentity.ilx_;
            ily_ = IdentityXIdentity.ily_;
            
            z_max = IdentityXIdentity.z_max;
            z_min = IdentityXIdentity.z_min;
            
            num_of_points = IdentityXIdentity.num_of_points;

            pz = zeros(num_of_points, num_of_points);
            pz(1, x_:y_) = [xl, hl];
            pz(2, x_:y_) = [xu, hu];
            pz(3, x_:y_) = [xu, hl];
            pz(4, x_:y_) = [xl, hu];

            pz(:, z_) = IdentityXIdentity.f(pz(:, x_), pz(:, y_));
            pz(:, dzx_:dzy_) = IdentityXIdentity.gradient_f(pz(:, x_), pz(:, y_));

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
                    px = IdentityXIdentity.get_pointX(pzs, u_);
                    px(z_) = pzs(z_max, z_) - u_min_slope(x_)*(xu-xl);
                    po = pzs(su_max2min_i, x_:z_);
                    
                    px_inter_y = ( px(z_) - po(z_) ) / (hu - hl);
                    u_min_slope(y_) = min(abs([u_min_slope(y_), px_inter_y, px(dzy_)]));
                   
                else
                    py = IdentityXIdentity.get_pointY(pzs, u_);
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
                    py = IdentityXIdentity.get_pointY(pzs, u_);
                    py(z_) = pzs(z_max, z_) - u_min_slope(y_)*(hu-hl);
                    po = pzs(su_max2min_i, x_:z_);
              
                    py_inter_x = ( py(z_) - po(z_) ) / (xu - xl);
                    u_min_slope(x_) = min(abs([u_min_slope(x_), py_inter_x, py(dzx_)]));
                else
                    px = IdentityXIdentity.get_pointX(pzs, u_);
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
                px = IdentityXIdentity.get_pointX(pzs, u_);
                u_min_slope(y_) = min(abs([u_min_slope(y_), px(dzy_)]));

            elseif (u_min_slope_i(y_) ~= tan_i_) && (u_min_slope_i(y_) ~= int_max2min_i_)
                py = IdentityXIdentity.get_pointY(pzs, u_);
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
                    px = IdentityXIdentity.get_pointX(pzs, l_);
                    px(z_) = pzs(z_min, z_) + l_min_slope(x_)*(xu-xl);
                    po = pzs(sl_min2max_i, x_:z_);

                    px_inter_y = ( px(z_) - po(z_) ) / (hu - hl);
                    l_min_slope(y_) = min(abs([l_min_slope(y_), px_inter_y, px(dzy_)]));
                else
                    py = IdentityXIdentity.get_pointY(pzs, u_);
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
                    py = IdentityXIdentity.get_pointY(pzs, l_);
                    py(z_) = pzs(z_min, z_) + l_min_slope(y_)*(hu-hl);
                    po = pzs(sl_min2max_i, x_:z_);

                    py_inter_x = ( py(z_) - po(z_) ) / (xu - xl);
                    l_min_slope(x_) = min(abs([l_min_slope(x_), py_inter_x, py(dzx_)])); 
                else
                    px = IdentityXIdentity.get_pointX(pzs, l_);
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
                px = IdentityXIdentity.get_pointX(pzs, l_);
                l_min_slope(y_) = min(abs([l_min_slope(y_), px(dzy_)]));
            
            elseif (l_min_slope_i(y_) ~= tan_i_) && (l_min_slope_i(y_) ~= int_min2max_i_)
                py = IdentityXIdentity.get_pointY(pzs, l_);
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

        function [U_x, U_y, U_b, L_x, L_y, L_b, zmax, zmin] = getConstraints(xl, xu, hl, hu)
            u_ = IdentityXIdentity.u_;
            l_ = IdentityXIdentity.l_;
        
            x_ = IdentityXIdentity.x_;
            y_ = IdentityXIdentity.y_;
            z_ = IdentityXIdentity.z_;
            dzx_ = IdentityXIdentity.dzx_;
            dzy_ = IdentityXIdentity.dzy_;
            iux_ = IdentityXIdentity.iux_;
            iuy_ = IdentityXIdentity.iuy_;
            ilx_ = IdentityXIdentity.ilx_;
            ily_ = IdentityXIdentity.ily_;
            
            z_max = IdentityXIdentity.z_max;
            z_min = IdentityXIdentity.z_min;
            
            num_of_points = IdentityXIdentity.num_of_points;

            pz = zeros(num_of_points, num_of_points);
            pz(1, x_:y_) = [xl, hl];
            pz(2, x_:y_) = [xu, hu];
            pz(3, x_:y_) = [xu, hl];
            pz(4, x_:y_) = [xl, hu];

            pz(:, z_) = IdentityXIdentity.f(pz(:, x_), pz(:, y_));
            pz(:, dzx_:dzy_) = IdentityXIdentity.gradient_f(pz(:, x_), pz(:, y_));

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

            z_max_range = pzs(z_max, z_, :) - pzs(z_max-1:-1:z_min, z_, :); % slope in z coordinate
            x_max_range = pzs(z_max, x_, :) - pzs(z_max-1:-1:z_min, x_, :);
            y_max_range = pzs(z_max, y_, :) - pzs(z_max-1:-1:z_min, y_, :);
            pzs(z_max-1:-1:z_min, iux_, :) = z_max_range ./ x_max_range; % slope in x coordinate
            pzs(z_max-1:-1:z_min, iuy_, :) = z_max_range ./ y_max_range; % slope in y coordinate

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
                    px = IdentityXIdentity.get_pointX(pzs, u_);
                    px(z_) = pzs(z_max, z_) - u_min_slope(x_)*(xu-xl);
                    po = pzs(su_max2min_i, x_:z_);
                    
                    px_inter_y = ( px(z_) - po(z_) ) / (hu - hl);
                    u_min_slope(y_) = min(abs([u_min_slope(y_), px_inter_y, px(dzy_)]));
                   
                else
                    py = IdentityXIdentity.get_pointY(pzs, u_);
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
                    py = IdentityXIdentity.get_pointY(pzs, u_);
                    py(z_) = pzs(z_max, z_) - u_min_slope(y_)*(hu-hl);
                    po = pzs(su_max2min_i, x_:z_);
              
                    py_inter_x = ( py(z_) - po(z_) ) / (xu - xl);
                    u_min_slope(x_) = min(abs([u_min_slope(x_), py_inter_x, py(dzx_)]));
                else
                    px = IdentityXIdentity.get_pointX(pzs, u_);
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
                px = IdentityXIdentity.get_pointX(pzs, u_);
                u_min_slope(y_) = min(abs([u_min_slope(y_), px(dzy_)]));

            elseif (u_min_slope_i(y_) ~= tan_i_) && (u_min_slope_i(y_) ~= int_max2min_i_)
                py = IdentityXIdentity.get_pointY(pzs, u_);
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
                    px = IdentityXIdentity.get_pointX(pzs, l_);
                    px(z_) = pzs(z_min, z_) + l_min_slope(x_)*(xu-xl);
                    po = pzs(sl_min2max_i, x_:z_);

                    px_inter_y = ( px(z_) - po(z_) ) / (hu - hl);
                    l_min_slope(y_) = min(abs([l_min_slope(y_), px_inter_y, px(dzy_)]));
                else
                    py = IdentityXIdentity.get_pointY(pzs, u_);
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
                    py = IdentityXIdentity.get_pointY(pzs, l_);
                    py(z_) = pzs(z_min, z_) + l_min_slope(y_)*(hu-hl);
                    po = pzs(sl_min2max_i, x_:z_);

                    py_inter_x = ( py(z_) - po(z_) ) / (xu - xl);
                    l_min_slope(x_) = min(abs([l_min_slope(x_), py_inter_x, py(dzx_)])); 
                else
                    px = IdentityXIdentity.get_pointX(pzs, l_);
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
                px = IdentityXIdentity.get_pointX(pzs, l_);
                l_min_slope(y_) = min(abs([l_min_slope(y_), px(dzy_)]));
            
            elseif (l_min_slope_i(y_) ~= tan_i_) && (l_min_slope_i(y_) ~= int_min2max_i_)
                py = IdentityXIdentity.get_pointY(pzs, l_);
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
            if type == IdentityXIdentity.u_
                for i = IdentityXIdentity.z_max-1:-1:IdentityXIdentity.z_min
                    if a(i, IdentityXIdentity.iuy_) == inf
                        p = a(i, IdentityXIdentity.x_:IdentityXIdentity.dzy_);
                        break;
                    end
                end
            elseif type == IdentityXIdentity.l_
                for i = IdentityXIdentity.z_min+1:IdentityXIdentity.z_max
                    if a(i, IdentityXIdentity.ily_) == inf
                        p = a(i, IdentityXIdentity.x_:IdentityXIdentity.dzy_);
                        break;
                    end
                end
            end
        end
        
        function p = get_pointY(a, type)
            if type == IdentityXIdentity.u_
                for i = IdentityXIdentity.z_max-1:-1:IdentityXIdentity.z_min
                    if abs(a(i, IdentityXIdentity.iux_)) == inf
                        p = a(i, IdentityXIdentity.x_:IdentityXIdentity.dzy_);
                        break;
                    end
                end
            elseif type == IdentityXIdentity.l_
                for i = IdentityXIdentity.z_min+1:IdentityXIdentity.z_max
                    if a(i, IdentityXIdentity.ilx_) == inf
                        p = a(i, IdentityXIdentity.x_:IdentityXIdentity.dzy_);
                        break;
                    end
                end
            end
            disp('');
        end

        % main method
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%% Star %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function S = reach_star_approx(varargin)
            % author: Sung Woo Choi
            % date: 10/10/2022

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
%                 for i = 1:n
%                     S = IdentityXIdentity.stepIdentityXIdentity(S, X, H, i, dis_opt, lp_solver);
%                 end
                S = IdentityXIdentity.multiStepIdentityXIdentity(S, X, H, dis_opt, lp_solver);
            else
                n = X.dim;
                S = X.Sum(H);
                for i = 1:n
                    S = IdentityXIdentity.relaxedStepIdentityXIdentity(S, X, H, i, relaxFactor, dis_opt, lp_solver);
                end
%                 S = IdentityXIdentity.relaxedMultiStepIdentityXIdentity(X, H, relaxFactor, dis_opt, lp_solver);
            end
        end

        function S = multiStepIdentityXIdentity(I, X, H, dis_opt, lp_solver)
            % @X: input star set (input state)
            % @H: input star set (hidden state)
            % @i: index of the neuron
            
            N = I.dim;
            inds = 1:N;
            if strcmp(lp_solver, 'estimate')
                if strcmp(dis_opt, 'display')
                    fprintf('\nComputing estimate lower- and upper-bounds: ');
                end
                [xl, xu] = X.estimateRanges;
                [hl, hu] = H.estimateRanges;
            elseif strcmp(lp_solver, 'glpk') || strcmp(lp_solver, 'linprog')
                if strcmp(dis_opt, 'display')
                    fprintf('\nComputing lower-bounds: ');
                end
                xl = X.getMins(inds, [], dis_opt, lp_solver);
                hl = H.getMins(inds, [], dis_opt, lp_solver);
                if strcmp(dis_opt, 'display')
                    fprintf('\nComputing upper-bounds: ');  
                end
                xu = X.getMaxs(inds, [], dis_opt, lp_solver);
                hu = H.getMaxs(inds, [], dis_opt, lp_solver);
            end

            nv = I.nVar + 1;
            
            map2 = find(xl ~= xu & hl ~= hu);
            m = length(map2);
            V2 = zeros(N, m);
            for i=1:m
                V2(map2(i), i) = 1;
            end

            % new basis matrix
            new_V = [zeros(N, nv) V2];

            map1 = find(xl == xu & hl == hu);
            zl = IdentityXIdentity.f(xl(map1), hl(map1));
            new_V(map1, 1) = zl;
            new_V(map1, 2:nv+m) = 0;

            map1 = find(xl == xu & hl ~= hu);
            new_V(map1, 1) = xl(map1) .* H.V(map1, 1);
            new_V(map1, 2:nv+m) = xl(map1) .* I.V(map1, 2:nv);

            map1 = find(xl ~= xu &  hl == hu);
            new_V(map1, 1) = hl(map1) .* X.V(map1, 1);
            new_V(map1, 2:nv+m) = hl(map1) .* I.V(map1, 2:nv);
        
            if ~isempty(map2)
                [Ux, Uy, Ub, Lx, Ly, Lb, Zl, Zu] = deal(zeros(m, 1));
                for i = 1:m
                    [Ux(i), Uy(i), Ub(i), Lx(i), Ly(i), Lb(i), Zu(i), Zl(i)] =  IdentityXIdentity.getConstraints(xl(map2(i)), xu(map2(i)), hl(map2(i)), hu(map2(i)));
                end
                nX = X.nVar + 1;
                nH = H.nVar + 1;

                C11 = [-Lx.*X.V(map2, 2:nX), -Ly.*H.V(map2, 2:nH), -V2(map2, :)];
                d11 = Lx.*X.V(map2, 1) + Ly.*H.V(map2, 1) - Lb;
                
                C12 = [Ux.*X.V(map2, 2:nX), Uy.*H.V(map2, 2:nH), V2(map2, :)];
                d12 = -(Ux.*X.V(map2, 1) + Uy.*H.V(map2, 1)) + Ub;

                C1 = [C11; C12];
                d1 = [d11; d12];
            else
                C1 = [];
                d1 = [];
            end

            n = size(I.C, 1);
            C0 = [I.C zeros(n, m)];
            d0 = I.d;
            new_C = [C0; C1];
            new_d = [d0; d1];

            new_predicate_lb = [I.predicate_lb; Zl];
            new_predicate_ub = [I.predicate_ub; Zu];

            S = Star(new_V, new_C, new_d, new_predicate_lb, new_predicate_ub);
        end

        function S = stepIdentityXIdentity(I, X, H, index, dis_opt, lp_solver)
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
                zl = IdentityXIdentity.f(xl, hl);
                
                new_V = I.V;
                new_V(index, 1:I.nVar+1) = 0;
                new_V(index, 1) = zl;
                S = Star(new_V, I.C, I.d, I.predicate_lb, I.predicate_ub);

            elseif xl == xu && hl ~= hu
                % xl * H
                n = I.nVar + 1;
                
                new_V = I.V;
                new_V(index, 1) = xl * H.V(index, 1);
                new_V(index, 2:n) = xl * I.V(index, 2:n);
                S = Star(new_V, I.C, I.d, I.predicate_lb, I.predicate_ub);

            elseif xl ~= xu &&  hl == hu
                % X * hl
                n = I.nVar + 1;
                
                new_V = I.V;
                new_V(index, 1) = hl * X.V(index, 1);
                new_V(index, 2:n) = hl * I.V(index, 2:n);
                S = Star(new_V, I.C, I.d, I.predicate_lb, I.predicate_ub);

            else
                % x * y
                % X star set
                % H star set

                [Ux, Uy, Ub, Lx, Ly, Lb, zu, zl] =  IdentityXIdentity.getConstraints(xl, xu, hl, hu);

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
        
        %%%%%%%%%%%%%%%%%%%%%%%%%% SparseStar %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function S = reach_sparse_star_approx(varargin)
            % author: Sung Woo Choi
            % date: 03/08/2023

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

            if ~(isa(X, 'SparseStar') && isa(H, 'SparseStar'))
                error('Input sets are not star set');
            end

            if X.dim ~= H.dim
                error('Dimension of input sets does not match');
            end

            if ~strcmp(method, 'approx-sparse-star')
                error('Unknown reachability method');
            end

            if relaxFactor == 0
                n = X.dim;
                S = X.Sum(H);
%                 for i = 1:n
%                     S = IdentityXIdentity.sparse_stepIdentityXIdentity(S, X, H, i, dis_opt, lp_solver);
%                 end
                S = IdentityXIdentity.sparse_multiStepIdentityXIdentity(S, X, H, dis_opt, lp_solver);
            else
                n = X.dim;
                S = X.Sum(H);
                for i = 1:n
                    S = IdentityXIdentity.sparse_relaxedStepIdentityXIdentity(S, X, H, i, relaxFactor, dis_opt, lp_solver);
                end
            end
        end

        

        function S = sparse_multiStepIdentityXIdentity(I, X, H, dis_opt, lp_solver)
            % @X: input star set (input state)
            % @H: input star set (hidden state)
            % @i: index of the neuron
            
            N = I.dim;
            inds = 1:N;
            if strcmp(lp_solver, 'estimate')
                if strcmp(dis_opt, 'display')
                    fprintf('\nComputing estimate lower- and upper-bounds: ');
                end
                [xl, xu] = X.estimateRanges;
                [hl, hu] = H.estimateRanges;
            elseif strcmp(lp_solver, 'glpk') || strcmp(lp_solver, 'linprog')
                if strcmp(dis_opt, 'display')
                    fprintf('\nComputing lower-bounds: ');
                end
                xl = X.getMins(inds, [], dis_opt, lp_solver);
                hl = H.getMins(inds, [], dis_opt, lp_solver);
                if strcmp(dis_opt, 'display')
                    fprintf('\nComputing upper-bounds: ');  
                end
                xu = X.getMaxs(inds, [], dis_opt, lp_solver);
                hu = H.getMaxs(inds, [], dis_opt, lp_solver);
            
            end
            
            map2 = find(xl ~= xu & hl ~= hu);
            m = length(map2);
            isAsparse = m > 3;
            if isAsparse
                A2 = sparse(N, m);
            else
                A2 = zeros(N, m);
            end
            for i=1:m
                A2(map2(i), i) = 1;
            end

            % new basis matrix
            if isAsparse
                new_A = [sparse(N, 1), A2];
            else
                new_A = [zeros(N, 1), A2];
            end

            map1 = find(xl == xu & hl == hu);
            zl = IdentityXIdentity.f(xl(map1), hl(map1));
            new_A(map1, 1) = zl;
            new_A(map1, 2:m+1) = 0;

            [nIA, mIA] = size(I.A);
            mXA = size(X.A, 2);
            mHA = size(H.A, 2);

            map1 = find(xl == xu & hl ~= hu);
            new_A(map1, 1) = xl(map1) .* H.A(map1, 1);
            new_A(map1, 2:m+1) = xl(map1) .* I.A(map1, 2:mIA);

            map1 = find(xl ~= xu &  hl == hu);
            new_A(map1, 1) = hl(map1) .* X.A(map1, 1);
            new_A(map1, 2:m+1) = hl(map1) .* I.A(map1, 2:mIA);

            if ~isempty(map2)
                mZ = I.nVar + 2 -mXA - mHA;
                isSparse = nnz(I.A(:, 2:mA) + N)  < 0.5 * (mZ*I.dim + nIA*(mIA-1) + N*m);
                if isSparse
                    Z = sparse(length(map2), mZ);
                else
                    Z = zeros(length(map2), mZ);
                end

                [Ux, Uy, Ub, Lx, Ly, Lb, Zl, Zu] = deal(zeros(m, 1));
                for i = 1:m
                    [Ux(i), Uy(i), Ub(i), Lx(i), Ly(i), Lb(i), Zu(i), Zl(i)] =  IdentityXIdentity.getConstraints(xl(map2(i)), xu(map2(i)), hl(map2(i)), hu(map2(i)));
                end
                
                C11 = [Z, -Lx.*X.A(map2, 2:mXA), -Ly.*H.A(map2, 2:mHA), -A2(map2, :)];
                d11 = Lx.*X.A(map2, 1) + Ly.*H.A(map2, 1) - Lb;
                
                C12 = [Z, Ux.*X.A(map2, 2:mXA), Uy.*H.A(map2, 2:mHA), A2(map2, :)];
                d12 = -(Ux.*X.A(map2, 1) + Uy.*H.A(map2, 1)) + Ub;
        
                C1 = [C11; C12];
                d1 = [d11; d12];
            else
                C1 = [];
                d1 = [];
            end

            n = size(I.C, 1);
            if n == 1 && nnz(I.C) == 0
                C0 = [];
                d0 = [];
            else
                C0 = [I.C zeros(n, m)];
                d0 = I.d;
            end
            new_C = [C0; C1];
            new_d = [d0; d1];

            new_pred_lb = [I.pred_lb; Zl];
            new_pred_ub = [I.pred_ub; Zu];

            S = SparseStar(new_A, new_C, new_d, new_pred_lb, new_pred_ub);

        end
        %%%%%%%%%%%%%%%%%%%%%%%%%% SparseStar END %%%%%%%%%%%%%%%%%%%%%%%%%

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

                R = IdentityXIdentity.reach_star_approx(X, H, method, relaxFactor, dis_opt, lp_solver);
            
            elseif strcmp(method, 'approx-sparse-star')

                R = IdentityXIdentity.reach_sparse_star_approx(X, H, method, relaxFactor, dis_opt, lp_solver);

            else
                error('Unknown or unsupported reachability method for layer with LogSig activation function');
            end
        end
    end
end

