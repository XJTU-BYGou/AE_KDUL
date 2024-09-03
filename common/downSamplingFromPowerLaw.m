function [otptx,selIndex] = downSamplingFromPowerLaw(inptx,xInt)
% [otptx,selIndex] = downSamplingFromPowerLaw(inptx,xInt)
% Gby, Powered by MATLAB R2021a
%
%
%
rng(0);

selIndex = [];
otptx = [];
for i = 1:numel(xInt)-1
    tmpIndex = find(inptx>=xInt(i)&inptx<xInt(i+1));    
    if length(tmpIndex) > 0
        selIndex = [selIndex,tmpIndex(randperm(numel(tmpIndex),1))];
    end
end
otptx = inptx(selIndex);

end