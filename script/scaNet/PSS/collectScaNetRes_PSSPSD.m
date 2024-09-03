%%
clear
warning('off');
%% set env
addpath '..\..\..\common';
addpath '..\..\..\data';

%% Import Data
% pss_dataloader;

load('training_data.mat','databasename','fs','res_pri','res_tra');
[vecT,vecTR] = dataoperator(res_tra,0);

timePoints = [2000,11060];
regimeLabel = getLabelWith3Regim([res_pri.Time],timePoints);
%% 
freq = [0:2e4:1e6];
PSD_Dataset = wave2psd(vecTR,fs,freq);
PSD_Dataset = log10(PSD_Dataset);
batchShape = 51;
wfTime = arrayfun(@(x)single(x.Time+ [1:batchShape]./fs),res_pri,'UniformOutput',false);

wfData = PSD_Dataset;
wfTime = cell2mat(wfTime);

wfData = reshape(wfData,[],1,batchShape);
wfTime = reshape(wfTime,[],1,batchShape);

% wfData = wfData + randn(size(wfData)).*1e-2;
save('.\trainingData_scaNet.mat','wfData','wfTime','-v7.3');


%% 
h5File = '..\..\trainedModel\summaries\PSSres_clus2\clusters.h5';

idx = h5read(h5File,'/epoch_00199/hot');
idx = reshape(idx,[],numel(res_pri));
clusIdx = mean(idx,1);

%%
predLabel = idx' + 1;
predLabel = 2 - idx';

%%
exportPath = '..\..\trainedModel';
mkdir(exportPath)
save(fullfile(exportPath,'res_scaNet_PSS.mat'),'predLabel','h5File');

% %% plot
% figure;
% aeMLPlotor([res_pri.Eny],double(predLabel));
% figure;
% semilogy([res_pri(double(predLabel)==1).Time],[res_pri(double(predLabel)==1).Eny],'b.');
% hold on;
% semilogy([res_pri(double(predLabel)==2).Time],[res_pri(double(predLabel)==2).Eny],'r.');
% 
% 
% tInt = 300;
% overlapInt = tInt/2;
% stTime = 0:tInt - overlapInt:[res_pri(end).Time];
% edTime = tInt - overlapInt:tInt - overlapInt:[res_pri(end).Time] + tInt - overlapInt;
% ratio21 = sum(double(predLabel).*([res_pri.Time]' >= stTime & [res_pri.Time]' <edTime))./sum([res_pri.Time]' >= stTime & [res_pri.Time]' <edTime) - 1;
% figure;
% plot(stTime,ratio21,'o-');
% 
% 
% figure;
% aeMLPlotor([res_pri(regimeLabel==0).Eny],double(predLabel(regimeLabel==0)));
