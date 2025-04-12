function [loss] = getAggregateLoss(Y,timeMat,varargin)
    timeInt = [];
    w = 0.5;
    for i = 1:length(varargin)/2
        switch lower(varargin{i*2-1})
            case 'w'
            w = varargin{i*2};
            case 'timeint'
            timeInt = varargin{i*2};
        end
    end
    % Determining the number of regions
    timeTransMat_l = @(x)getTimeTransMat(timeMat,x,[],timeInt);
    % loss calculation
    l = getAggregateTrendLoss(timeTransMat_l,Y,'w',w);  
    loss = l;

end