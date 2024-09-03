function res = avgPercentageCalculation(res_pri,label,tInt,varargin)
label = reshape(double(label),[],1);
overlapInt = tInt*1/2;

for i = 1:length(varargin)/2
    switch lower(varargin{i*2-1})
        case 'overlap'
            overlapInt = varargin{i*2};
    end
end

if isstruct(res_pri) || istable(res_pri)
    T = reshape([res_pri.Time],1,[]);
else
    T = reshape(res_pri,1,[]);
end

stTime = -tInt/2:tInt - overlapInt:T(end);
edTime = stTime + tInt;
cenTime = (stTime + edTime) ./2;
ratio = sum(label.*(T' >= stTime & T' <edTime))./sum(T' >= stTime & T' <edTime) - 1;


reptNum = 200;
ratioRept = nan(reptNum,numel(stTime));
for i = 1:numel(stTime)
    tmpLabel = label(T' >= stTime(i) & T' <edTime(i));
for j = 1:reptNum
    ratioRept(j,i) = mean(randsample(tmpLabel,numel(tmpLabel),true)) - 1;
end
end



res.StTime = stTime;
res.EdTime = edTime;
res.CenTime = cenTime;
res.Percentage = ratio;
res.OverlapInt = overlapInt;
res.RepeatNum = reptNum;
res.PercentageRepeat = ratioRept;
res.Errorbar = std(ratioRept,1);
res.ErrorUpp = prctile(ratioRept,95,1);
res.ErrorLow = prctile(ratioRept,5,1);
res.PercentageMean = mean(ratioRept,1);
end