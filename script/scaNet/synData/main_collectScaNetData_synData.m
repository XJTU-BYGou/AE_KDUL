%%
clear
warning('off');
%% set env
addpath '..\..\..\common';
addpath '..\..\';
addpath '..\..\synData';

%%
PSDstrategy = 2;
batchShape = 51;
fs = 2e7;
for trendType = [1,2,3]
load(['..\..\..\data\synData\synDataset_PSDstrategy',num2str(PSDstrategy),'_Trend',num2str(trendType),'.mat']);

%% Padding & Cut
wfTime = arrayfun(@(x)single(x+ [1:batchShape]./fs),sampleTime,'UniformOutput',false);

wfData = 10.^synPSD_Dataset*1e6;
wfTime = cell2mat(wfTime);

wfData = reshape(wfData,[],1,batchShape);
wfTime = reshape(wfTime,[],1,batchShape);

save(['synData_PSD-Trend',num2str(trendType),'.mat'],'wfData','wfTime','-v7.3');


end
