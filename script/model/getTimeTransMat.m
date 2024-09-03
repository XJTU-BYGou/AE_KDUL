function [timeTransMat] = getTimeTransMat(timeMat,intNum,seedNum,timeInt)
% Define the timeTransMat
    if ~isempty(seedNum)
        rng(seedNum);
    end
    if nargin <4
        timeInt = [];
    end
    if isempty(timeInt)
        maxTime = ceil(max(timeMat)) + 10;
        minTime = max(floor(min(timeMat)) - 10,0);
    else
        minTime = timeInt(1);
        maxTime = timeInt(2);
    end
    
   
    timePart = [0;sort(rand(intNum,1));1] * (maxTime - minTime) + minTime;
    ttM = timeMat >= timePart(1:end-1) & timeMat < timePart(2:end);
    timelen = diff(timePart);
    remFlag = sum(ttM,2) > 0;
    timeTransMat.Mat = ttM(remFlag,:);
    timeTransMat.Timelen = timelen(remFlag,:);

    timeTransMat.N = size(timeTransMat.Mat,1);
    timeTransMat.Num = timeTransMat.Mat*ones(size(timeTransMat.Mat,2),1);
        
end
