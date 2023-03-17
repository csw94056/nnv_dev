function [R] = deleteRedundantSets(R,Rold,options)
% deleteRedundantSets - delete reachable sets that are already covered by
% other sets
%
% Syntax:  
%    [R] = deleteRedundantSets(R,Rold,options)
%
% Inputs:
%    R - reachable sets
%    Rold - reachable sets of previous time steps
%    options - options for the computation of the reachable set
%
% Outputs:
%    R - reachable sets
%
% Example: 
%
% Other m-files required: none
% Subfunctions: none
% MAT-files required: none
%
% See also: none

% Author:       Matthias Althoff
% Written:      29-June-2009
% Last update:  03-February-2011
%               29-June-2018
% Last revision:---

%------------- BEGIN CODE --------------


%set reduction method
redMethod='pca';

%increase internal count
try
    R.internalCount=Rold.internalCount+1;
catch
    %for first run
    R.internalCount=3;
end

%if internal count='some constant'
if R.internalCount==options.reductionInterval
    %reset internal count
    R.internalCount=1;
    %overapproximate reachable set of time point(!) by parallelpipeds and
    %save them as polytopes
    R.P=[];
    for i=1:length(R.tp)
        R.tp{i}.set=reduce(R.tp{i}.set,redMethod,1);
        %generate polytope
        R.P{i}=polytope(R.tp{i}.set,options);
    end    
elseif R.internalCount==2
    %intersect each reachable set with each previous reachable set
    for iNewSet=1:length(R.tp)
        %approximate new set of time points by parallelpiped 
        R.tp{iNewSet}.set=reduce(R.tp{iNewSet}.set,redMethod,1);
        %generate mpt polytope
        Pnew{iNewSet}=polytope(R.tp{iNewSet}.set,options);
    end
    %initialize Pcut
    Pcut=Pnew;
    %intersection with previous time step
    for iNewSet=1:length(Pcut)
        %intersect with previous sets
        for iOldSet=1:length(Rold.P)           
            Pcut{iNewSet}=Pcut{iNewSet}\Rold.P{iOldSet};
        end
    end
    
    %reset iChecked counter
    iChecked=1;
    
    %figure;
    %hold on
    
    %intersection with actual time step
    for iNewSet=1:length(Pcut)
        %intersect with actual sets
        for iOtherSet=1:length(Pnew)  
            if iOtherSet~=iNewSet
                Pcut{iNewSet}=Pcut{iNewSet}\Pnew{iOtherSet};
            end
        end
        %is polytope empty?
        if ~isempty(Pcut{iNewSet})
            Rnew{iChecked}=R.tp{iNewSet}.set;
            iChecked=iChecked+1;
            %plot(Rnew{iChecked-1});
        else
            disp('canceled!!');
        end    
    end
    
    %copy only ckecked reachable sets
    R.tp=[];
    for i=1:length(Rnew)
        R.tp{i}.set=Rnew{i};
        R.tp{i}.error=0*options.maxError;
    end
end


%------------- END OF CODE --------------