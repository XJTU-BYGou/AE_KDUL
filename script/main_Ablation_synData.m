%% Plot Fig
clear
addpath '.\synData';
addpath '.\model';
addpath '.\trainedModel';
addpath '..\common';
%%
varVol = 3;
rng(0);
for trendType = [1,2,3]
load(['..\data\synData\synDataset_PSDstrategy2_Trend',num2str(trendType),'.mat']);
%%
switch varVol
    case 0
        modelFolder = ['res_trainMdlSyn'];
    case 1
        modelFolder = ['res_trainMdlSyn_variant1'];
    case 2
        modelFolder = ['res_trainMdlSyn_variant2'];
    case 3
        modelFolder = ['res_trainMdlSyn_nongrad'];
end
%%
reptNum = 100;
modelIndex = 1;
% 1 The proposed NN
cName{modelIndex} = 'The proposed method';
load(fullfile('.\trainedModel\synData',modelFolder,['res_synDataset_Trend',num2str(trendType)],'res_NN.mat'),...
    'net','predLabel','predScore','Info');
predLabelCell{modelIndex} = double(predLabel);
predScoreCell{modelIndex} = double(predScore);
if varVol == 1
predLoss{modelIndex} = arrayfun(@(x)getAggregateLoss_variant1(predScoreCell{modelIndex},sampleTime'),1:reptNum);
elseif varVol == 2
predLoss{modelIndex} = arrayfun(@(x)getAggregateLoss_variant2(predScoreCell{modelIndex},sampleTime'),1:reptNum);
else
predLoss{modelIndex} = arrayfun(@(x)getAggregateLoss(predScoreCell{modelIndex},sampleTime'),1:reptNum);    
end
predLoss_avg(modelIndex) = mean(predLoss{modelIndex});
predLoss_std(modelIndex) = std(predLoss{modelIndex});
paramNum(modelIndex) = numel(net.Layers(2).Weights) + numel(net.Layers(2).Bias) + ...
    numel(net.Layers(5).Weights) + numel(net.Layers(5).Bias) + ...
    numel(net.Layers(8).Weights) + numel(net.Layers(8).Bias); % 237 

modelIndex = 2;
% 2 Logistic Regression
cName{modelIndex} = 'LR';
load(fullfile('.\trainedModel\synData',modelFolder,['res_synDataset_Trend',num2str(trendType)],'res_LR.mat'),...
    'model','trainHistory');
predScore = appLogisticRegression(model.LR.Model,XTrain);
if isdlarray(predScore)
    predScore = extractdata(predScore);
end
[~,predLabelCell{modelIndex}] = max(predScore,[],1);
predScoreCell{modelIndex} = double(predScore);
if varVol == 1
predLoss{modelIndex} = arrayfun(@(x)getAggregateLoss_variant1(predScoreCell{modelIndex},sampleTime'),1:reptNum);
elseif varVol == 2
predLoss{modelIndex} = arrayfun(@(x)getAggregateLoss_variant2(predScoreCell{modelIndex},sampleTime'),1:reptNum);
else
predLoss{modelIndex} = arrayfun(@(x)getAggregateLoss(predScoreCell{modelIndex},sampleTime'),1:reptNum);    
end
predLoss_avg(modelIndex) = mean(predLoss{modelIndex});
predLoss_std(modelIndex) = std(predLoss{modelIndex});
paramNum(modelIndex) = numel(model.LR.Model.W)+numel(model.LR.Model.B); % 17

modelIndex = 3;
cName{modelIndex} = 'GDA';
load(fullfile('.\trainedModel\synData',modelFolder,['res_synDataset_Trend',num2str(trendType)],'res_GDA.mat'),...
    'model','trainHistory');
predScore = appGaussianDiscriminative(model.GDA.BestModel.Model,XTrain);
[~,predLabelCell{modelIndex}] = max(predScore,[],1);
predScoreCell{modelIndex} = double(predScore);
if varVol == 1
predLoss{modelIndex} = arrayfun(@(x)getAggregateLoss_variant1(predScoreCell{modelIndex},sampleTime'),1:reptNum);
elseif varVol == 2
predLoss{modelIndex} = arrayfun(@(x)getAggregateLoss_variant2(predScoreCell{modelIndex},sampleTime'),1:reptNum);
else
predLoss{modelIndex} = arrayfun(@(x)getAggregateLoss(predScoreCell{modelIndex},sampleTime'),1:reptNum);    
end
predLoss_avg(modelIndex) = mean(predLoss{modelIndex});
predLoss_std(modelIndex) = std(predLoss{modelIndex});
paramNum(modelIndex) = numel(model.GDA.Model.Alpha) + numel(model.GDA.Model.Mu) +...
    (size(model.GDA.Model.Sigma,1)+1)*size(model.GDA.Model.Sigma,1)*size(model.GDA.Model.Sigma,3)/2; % 306

modelIndex = 4;
% 4 Kernel SVM -rbf
cName{modelIndex} = 'SVM';
load(fullfile('.\trainedModel\synData',modelFolder,['res_synDataset_Trend',num2str(trendType)],'res_KSVM.mat'),...
    'model','trainHistory');
predScore = appKernelSupportVectorMachine(model.KSVM.BestModel.Model,XTrain,XTrain);
if isdlarray(predScore)
    predScore = extractdata(predScore);
end
[~,predLabelCell{modelIndex}] = max(predScore,[],1);
predScoreCell{modelIndex} = double(predScore);
if varVol == 1
predLoss{modelIndex} = arrayfun(@(x)getAggregateLoss_variant1(predScoreCell{modelIndex},sampleTime'),1:reptNum);
elseif varVol == 2
predLoss{modelIndex} = arrayfun(@(x)getAggregateLoss_variant2(predScoreCell{modelIndex},sampleTime'),1:reptNum);
else
predLoss{modelIndex} = arrayfun(@(x)getAggregateLoss(predScoreCell{modelIndex},sampleTime'),1:reptNum);    
end
predLoss_avg(modelIndex) = mean(predLoss{modelIndex});
predLoss_std(modelIndex) = std(predLoss{modelIndex});
paramNum(modelIndex) = size(XTrain,1)+1;

if varVol<3
modelIndex = 5;
% 5 GBDT
cName{modelIndex} = 'GBDT';
% load(fullfile('.\trainedModel\synData',modelFolder,['res_synDataset_Trend',num2str(trendType)],'res_GBDT.mat'),...
%     'model','trainHistory');
% predScore = appGBDT(model.GBDT.Model,XTrain);
% paramNum(modelIndex) = sum(cellfun(@(x)sum(x.IsBranchNode),model.GBDT.Model),'all');
load(fullfile('.\trainedModel\synData',modelFolder,['res_synDataset_Trend',num2str(trendType)],'res_GBDT.mat'),...
    'predScore','trainHistory');
paramNum(modelIndex) = 3e5;
[~,predLabelCell{modelIndex}] = max(predScore,[],1);
predScoreCell{modelIndex} = double(predScore);
if varVol == 1
predLoss{modelIndex} = arrayfun(@(x)getAggregateLoss_variant1(predScoreCell{modelIndex},sampleTime'),1:reptNum);
elseif varVol == 2
predLoss{modelIndex} = arrayfun(@(x)getAggregateLoss_variant2(predScoreCell{modelIndex},sampleTime'),1:reptNum);
else
predLoss{modelIndex} = arrayfun(@(x)getAggregateLoss(predScoreCell{modelIndex},sampleTime'),1:reptNum);    
end
predLoss_avg(modelIndex) = mean(predLoss{modelIndex});
predLoss_std(modelIndex) = std(predLoss{modelIndex});
end
%%
tInt = 500;
overlapInt = 400;
ratioCell = cellfun(@(x)avgPercentageCalculation(sampleTime,double(x),tInt,'overlap',overlapInt),...
    predLabelCell,'UniformOutput',false);
ratioReal = avgPercentageCalculation(sampleTime,synLabel,tInt,'overlap',overlapInt);
accMat = cellfun(@(x)mean(synLabel==reshape(double(x),[],1)),predLabelCell);

switch trendType
    case 1
Precision = cellfun(@(x)sum(reshape(double(x),[],1)==2 & synLabel==2) ./ sum(double(x)==2),predLabelCell);
Recall = cellfun(@(x)sum(reshape(double(x),[],1)==2 & synLabel==2) ./ sum(synLabel==2),predLabelCell);
F1Score = 2.*Precision.*Recall./(Precision+Recall);
    case 3
Precision = cellfun(@(x)sum(reshape(double(x),[],1)==2 & synLabel==2) ./ sum(double(x)==2),predLabelCell);
Recall = cellfun(@(x)sum(reshape(double(x),[],1)==2 & synLabel==2) ./ sum(synLabel==2),predLabelCell);
F1Score = 2.*Precision.*Recall./(Precision+Recall);
    otherwise
Precision = cellfun(@(x)sum(reshape(double(x),[],1)==2 & synLabel==2) ./ sum(double(x)==2),predLabelCell);
Recall = cellfun(@(x)sum(reshape(double(x),[],1)==2 & synLabel==2) ./ sum(synLabel==2),predLabelCell);
F1Score = 2.*Precision.*Recall./(Precision+Recall);
end

RMSE_Ratio = cellfun(@(x)sqrt(mean((ratioReal.Percentage - x.Percentage).^2.*10000)),ratioCell);
NRMSE_Ratio = cellfun(@(x)sqrt(mean((ratioReal.Percentage - x.Percentage).^2.*10000)),ratioCell);

SSE_Ratio = cellfun(@(x)sum((ratioReal.Percentage - x.Percentage).^2),ratioCell);
SST_Ratio = sum((ratioReal.Percentage - mean(ratioReal.Percentage)).^2);
R2_Ratio = 1-SSE_Ratio./SST_Ratio;

AB2C = cellfun(@(x)trapz(x.CenTime,abs(ratioReal.Percentage - x.Percentage)),ratioCell);

tmp = struct('cName',cName,'RMSE',num2cell(RMSE_Ratio),'R2',num2cell(R2_Ratio),'AB2C',num2cell(AB2C),'ratioCell',ratioCell,'ratioReal',ratioReal',...
    'Accuracy',num2cell(accMat),'Precision',num2cell(Precision),'Recall',num2cell(Recall),'F1Score',num2cell(F1Score));

resData{trendType} = tmp;
%%
end
%%
save(fullfile('.\export',[modelFolder,'_dataCompRes.mat']),...
    'resData')
%%
modelFolder1 = 'res_trainMdlSyn_variant1';
modelFolder2 = 'res_trainMdlSyn_variant2';
modelFolder3 = 'res_trainMdlSyn_nongrad';
load(fullfile('.\export',[modelFolder3,'_dataCompRes.mat']));
resData_nongrad = resData;
load(fullfile('.\export',[modelFolder2,'_dataCompEes.mat']));
resData_var2 = resData;
load(fullfile('.\export',[modelFolder1,'_dataCompRes.mat']));
resData_var1 = resData;
load(fullfile('.\export',['res_trainMdlSyn','_dataCompRes.mat']));
% load(fullfile('.\export','data_FigS4.mat'));

%%
% alignInd = [1,2,3,4,5];
% selInd = 5;
% cName = arrayfun(@(x)x.cName,resData{1}(alignInd),'UniformOutput',false);
% clear F1Score accMat RMSE_Ratio Precision Recall R2_Ratio AB2C
% for trendType = 1:3
% % ratioCell = arrayfun(@(x)x.ratioCell,resData{trendType});
% % ratioReal = arrayfun(@(x)x.ratioReal,resData{trendType});
% % cName = arrayfun(@(x)x.cName,resData{trendType},'UniformOutput',false);
% F1Score_tmp = [arrayfun(@(x)x.F1Score,resData{trendType}(alignInd));...
%     arrayfun(@(x)x.F1Score,resData_var1{trendType}(alignInd));...
%     arrayfun(@(x)x.F1Score,resData_var2{trendType}(alignInd))];
% accMat_tmp = [arrayfun(@(x)x.Accuracy,resData{trendType}(alignInd));...
%     arrayfun(@(x)x.Accuracy,resData_var1{trendType}(alignInd));...
%     arrayfun(@(x)x.Accuracy,resData_var2{trendType}(alignInd))];
% RMSE_Ratio_tmp = [arrayfun(@(x)x.RMSE,resData{trendType}(alignInd));...
%     arrayfun(@(x)x.RMSE,resData_var1{trendType}(alignInd));...
%     arrayfun(@(x)x.RMSE,resData_var2{trendType}(alignInd))];
% Precision_tmp = [arrayfun(@(x)x.Precision,resData{trendType}(alignInd));...
%     arrayfun(@(x)x.Precision,resData_var1{trendType}(alignInd));...
%     arrayfun(@(x)x.Precision,resData_var2{trendType}(alignInd))];
% Recall_tmp = [arrayfun(@(x)x.Recall,resData{trendType}(alignInd));...
%     arrayfun(@(x)x.Recall,resData_var1{trendType}(alignInd));...
%     arrayfun(@(x)x.Recall,resData_var2{trendType}(alignInd))];
% R2_Ratio_tmp = max([arrayfun(@(x)x.R2,resData{trendType}(alignInd));...
%     arrayfun(@(x)x.R2,resData_var1{trendType}(alignInd));...
%     arrayfun(@(x)x.R2,resData_var2{trendType}(alignInd))],0);
% AB2C_tmp = max([arrayfun(@(x)x.AB2C,resData{trendType}(alignInd));...
%     arrayfun(@(x)x.AB2C,resData_var1{trendType}(alignInd));...
%     arrayfun(@(x)x.AB2C,resData_var2{trendType}(alignInd))],0);
% 
% F1Score(:,trendType) = F1Score_tmp(:,selInd);
% accMat(:,trendType) = accMat_tmp(:,selInd);
% RMSE_Ratio(:,trendType) = RMSE_Ratio_tmp(:,selInd);
% Precision(:,trendType) = Precision_tmp(:,selInd);
% Recall(:,trendType) = Recall_tmp(:,selInd);
% R2_Ratio(:,trendType) = R2_Ratio_tmp(:,selInd);
% AB2C(:,trendType) = AB2C_tmp(:,selInd);
% end
% 
% tbl = [reshape(accMat',[],1),reshape(Precision',[],1),reshape(Recall',[],1),...
%     reshape(F1Score',[],1),reshape(RMSE_Ratio',[],1),reshape(R2_Ratio',[],1),reshape(AB2C',[],1)];
% tbl = [reshape(accMat',[],1),reshape(F1Score',[],1),...
%     reshape(RMSE_Ratio',[],1),reshape(AB2C',[],1)];
