function [loss] = getAggregateLoss(Y,timeMat,varargin)
    timeInt = [];
    w = 0.5;
    for i = 1:length(varargin)/2
        switch lower(varargin{i*2-1})
            case 'w'
            w2 = varargin{i*2};
            case 'timeint'
            timeInt = varargin{i*2};
        end
    end
    % Determining the number of regions
    intNum = max(poissrnd(12),5);
    timeTransMat_l = getTimeTransMat(timeMat,intNum,[]);
    % loss calculation
    l = getAggregateTrendLoss(timeTransMat_l,Y,'w',w);  
    loss = l;

end