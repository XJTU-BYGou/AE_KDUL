%%
clear
addpath '.\synData';
addpath '.\model';
addpath '.\trainedModel';
addpath '..\common';
synDataGenerator;
%%
clear
%%
dateStr = '';
modelFolder = ['res_trainMdlSyn'];
for trendType = [1,2,3]
load(['..\data\synData\synDataset_PSDstrategy2_Trend',num2str(trendType),'.mat']);
%%
reptNum = 100;

modelIndex = 1;
% 1 The proposed NN
cName{modelIndex} = 'The proposed method';
load(fullfile('.\trainedModel\synData',modelFolder,['res_synDataset_Trend',num2str(trendType)],'res_NN.mat'),...
    'net','predLabel','predScore','Info');
predLabelCell{modelIndex} = double(predLabel);
predScoreCell{modelIndex} = double(predScore);
predLoss{modelIndex} = arrayfun(@(x)getAggregateLoss(predScoreCell{modelIndex},sampleTime'),1:reptNum);
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
predLoss{modelIndex} = arrayfun(@(x)getAggregateLoss(predScoreCell{modelIndex},sampleTime'),1:reptNum);
predLoss_avg(modelIndex) = mean(predLoss{modelIndex});
predLoss_std(modelIndex) = std(predLoss{modelIndex});
paramNum(modelIndex) = 2*(numel(model.LR.Model.W)+numel(model.LR.Model.B)); % 17
% paramNum(modelIndex) = 17;

modelIndex = 3;
cName{modelIndex} = 'GDA';
load(fullfile('.\trainedModel\synData',modelFolder,['res_synDataset_Trend',num2str(trendType)],'res_GDA.mat'),...
    'model','trainHistory');
predScore = appGaussianDiscriminative(model.GDA.BestModel.Model,XTrain);
[~,predLabelCell{modelIndex}] = max(predScore,[],1);
predScoreCell{modelIndex} = double(predScore);
predLoss{modelIndex} = arrayfun(@(x)getAggregateLoss(predScoreCell{modelIndex},sampleTime'),1:reptNum);% 1/3.5*
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
predLoss{modelIndex} = arrayfun(@(x)getAggregateLoss(predScoreCell{modelIndex},sampleTime'),1:reptNum);
predLoss_avg(modelIndex) = mean(predLoss{modelIndex});
predLoss_std(modelIndex) = std(predLoss{modelIndex});
paramNum(modelIndex) = size(XTrain,1)+1;

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
predLoss{modelIndex} = arrayfun(@(x)getAggregateLoss(predScoreCell{modelIndex},sampleTime'),1:reptNum);
predLoss_avg(modelIndex) = mean(predLoss{modelIndex});
predLoss_std(modelIndex) = std(predLoss{modelIndex});


% Peer machine learning method
modelIndex = 6;
% 6 unsupervised Cluster
cName{modelIndex} = 'Kmeans';
load(['.\trainedModel\synData\res_synDataset_unspervised_Trend',num2str(trendType),'.mat']);
predLabelCell{modelIndex} = predLabelKmeans;
predScoreCell{modelIndex} = nan(numel(sampleTime),2);
predLoss{modelIndex} = nan;
predLoss_avg(modelIndex) = mean(predLoss{modelIndex});
predLoss_std(modelIndex) = std(predLoss{modelIndex});
paramNum(modelIndex) = nan; % 237 


modelIndex = 7;
% 7 unsupervised ScaNet
cName{modelIndex} = 'ScaNet';
load(['.\trainedModel\synData\res_synDataset_ScaNet_Trend',num2str(trendType),'.mat'], 'predLabel');
predLabelCell{modelIndex} = double(predLabel);
predScoreCell{modelIndex} = nan(numel(sampleTime),2);
predLoss{modelIndex} = nan;
predLoss_avg(modelIndex) = mean(predLoss{modelIndex});
predLoss_std(modelIndex) = std(predLoss{modelIndex});
paramNum(modelIndex) = 0;


modelIndex = 8;
% 9 supervised NN with focal loss
cName{modelIndex} = 'IDNN';
load(['.\trainedModel\synData\res_synDataset_supervised_Trend',num2str(trendType),'.mat'], 'predLabel','predScore');
predLabelCell{modelIndex} = double(predLabel);
predScoreCell{modelIndex} = double(predScore');
predLoss{modelIndex} = nan;
predLoss_avg(modelIndex) = mean(predLoss{modelIndex});
predLoss_std(modelIndex) = std(predLoss{modelIndex});
paramNum(modelIndex) = numel(net.Layers(2).Weights) + numel(net.Layers(2).Bias) + ...
    numel(net.Layers(5).Weights) + numel(net.Layers(5).Bias) + ...
    numel(net.Layers(8).Weights) + numel(net.Layers(8).Bias); % 237 

modelIndex = 9;
rng(trendType);
randPred = randsample(2,numel(synLabel),true);
cName{modelIndex} = 'RandomGauss';
predLabelCell{modelIndex} = randPred;
predScoreCell{modelIndex} = nan(numel(sampleTime),2);
predLoss{modelIndex} = nan;
predLoss_avg(modelIndex) = mean(predLoss{modelIndex});
predLoss_std(modelIndex) = std(predLoss{modelIndex});
paramNum(modelIndex) = 0;

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
% [res_train,record_train] = cellfun(@(x)AEMixExponent(Eny,x),predLabelCell,'UniformOutput',false);

%%

predLoss = cellfun(@single,predLoss,'UniformOutput',false);

outlierCutOff = [0.635.*ones(size(predLoss))];
predLoss_eff = cellfun(@(x,y)x(x>y),predLoss,num2cell(outlierCutOff),'UniformOutput',false);
predLoss_avg = cellfun(@mean,predLoss_eff);
predLoss_std = cellfun(@std,predLoss);

razorK = 0.1;


[predOverallLoss,perfL,perfC] = cellfun(@(x,y) getOverallPerformance(x,y,razorK),...
    predLoss,mat2cell(paramNum,size(paramNum,1),ones(size(paramNum))),'UniformOutput',false);
predOverallLoss_avg = cellfun(@(x)mean(single(x)),predOverallLoss);
predOverallLoss_std = cellfun(@(x)std(single(x)),predOverallLoss);
perfL_std = cellfun(@(x)std(single(x)),perfL);
perfC_std = cellfun(@(x)std(single(x)),perfC);
perfL_avg = cellfun(@(x)mean(single(x)),perfL);
perfC_avg = cellfun(@(x)mean(single(x)),perfC);

[sortedNum] = sort(unique(perfL_avg(1:5)),'descend');
Hcrite1_L = arrayfun(@(x)find(ismember(sortedNum,x)),perfL_avg(1:5));
Hcrite2_L = perfL_avg(1:5)./max(perfL_avg(1:5));

[sortedNum] = sort(unique(perfC_avg(1:5)),'descend');
Hcrite1_C = arrayfun(@(x)find(ismember(sortedNum,x)),perfC_avg(1:5));
Hcrite2_C = perfC_avg(1:5)./max(perfC_avg(1:5));

%%
tmp = struct('cName',cName,'RMSE',num2cell(RMSE_Ratio),'R2',num2cell(R2_Ratio),'AB2C',num2cell(AB2C),'ratioCell',ratioCell,'ratioReal',ratioReal',...
    'Accuracy',num2cell(accMat),'Precision',num2cell(Precision),'Recall',num2cell(Recall),'F1Score',num2cell(F1Score));
resData{trendType} = tmp;

%%
tmp = struct('cName',cName,'paramNum',num2cell(paramNum),...
    'predScore',predScoreCell,'predLabel',predLabelCell,...
    'predLoss',predLoss,'predOverallLoss',predOverallLoss,...
    'predOverallLoss_avg',num2cell(predOverallLoss_avg),'predOverallLoss_std',num2cell(predOverallLoss_std),...
    'predLoss_avg',num2cell(predLoss_avg),'predLoss_std',num2cell(predLoss_std),...
    'perfL',perfL,'perfL_avg',num2cell(perfL_avg),'perfL_std',num2cell(perfL_std),...
    'perfC',perfC,'perfC_avg',num2cell(perfC_avg),'perfC_std',num2cell(perfC_std));
resDataPlus{trendType} = tmp;
%%
rng(trendType);
reptNum = 1000;
crtBoundary = sort(0.1+0.8.*rand(reptNum*2,2),2);
crtBoundary = crtBoundary(diff(crtBoundary,1,2)>0.1,:);
crtBoundary = crtBoundary(1:reptNum,:);
timeSegement = [zeros(reptNum,1),crtBoundary,ones(reptNum,1)]*endTime;

[segeRes] = cellfun(@(x)segementEvalution(synLabel,x,sampleTime,timeSegement),predLabelCell,'UniformOutput',false);
segeResData{trendType} = segeRes;
end

%%
exportPath = '.\export';
mkdir(exportPath);
save(fullfile(exportPath,'data_FigS4.mat'),...
    'resData','resDataPlus');
save(fullfile(exportPath,'data_TableS2_S5.mat'),...
    'resData','segeResData');

%%
function [segeRes] = segementEvalution(synLabel,predLabel,time,timeSegement)

tInt = 500;
overlapInt = 400;

predLabel = reshape(double(predLabel),[],1);
accMat = zeros(size(timeSegement,1),size(timeSegement,2)-1);
Precision = zeros(size(timeSegement,1),size(timeSegement,2)-1);
Recall = zeros(size(timeSegement,1),size(timeSegement,2)-1);
F1Score = zeros(size(timeSegement,1),size(timeSegement,2)-1);
for j = 1:size(timeSegement,1)
    time_r1 = time(time<=timeSegement(j,2));
    time_r2 = time(time>timeSegement(j,2) & time<=timeSegement(j,3)) - timeSegement(j,2);
    time_r3 = time(time>timeSegement(j,3))- timeSegement(j,3);
    synLabel_r1 = synLabel(time<=timeSegement(j,2));
    synLabel_r2 = synLabel(time>timeSegement(j,2) & time<=timeSegement(j,3));
    synLabel_r3 = synLabel(time>timeSegement(j,3));
    predLabel_r1 = predLabel(time<=timeSegement(j,2));
    predLabel_r2 = predLabel(time>timeSegement(j,2) & time<=timeSegement(j,3));
    predLabel_r3 = predLabel(time>timeSegement(j,3));
accMat(j,:) = [mean(synLabel_r1==predLabel_r1),...
    mean(synLabel_r2==predLabel_r2),...
    mean(synLabel_r3==predLabel_r3)];
Precision(j,:) = [sum(predLabel_r1==2 & synLabel_r1==2) ./ sum(predLabel_r1==2),...
    sum(predLabel_r2==2 & synLabel_r2==2) ./ sum(predLabel_r2==2),...
    sum(predLabel_r3==2 & synLabel_r3==2) ./ sum(predLabel_r3==2)];
Recall(j,:) = [sum(predLabel_r1==2 & synLabel_r1==2) ./ sum(synLabel_r1==2),...
    sum(predLabel_r2==2 & synLabel_r2==2) ./ sum(synLabel_r2==2),...
    sum(predLabel_r3==2 & synLabel_r3==2) ./ sum(synLabel_r3==2)];

ratioPred_r1 = avgPercentageCalculation(time_r1,predLabel_r1,tInt,'overlap',overlapInt,'reptnum',1);
ratioReal_r1 = avgPercentageCalculation(time_r1,synLabel_r1,tInt,'overlap',overlapInt,'reptnum',1);
ratioPred_r2 = avgPercentageCalculation(time_r2,predLabel_r2,tInt,'overlap',overlapInt,'reptnum',1);
ratioReal_r2 = avgPercentageCalculation(time_r2,synLabel_r2,tInt,'overlap',overlapInt,'reptnum',1);
ratioPred_r3 = avgPercentageCalculation(time_r3,predLabel_r3,tInt,'overlap',overlapInt,'reptnum',1);
ratioReal_r3 = avgPercentageCalculation(time_r3,synLabel_r3,tInt,'overlap',overlapInt,'reptnum',1);

RMSE(j,:) = [sqrt(mean((ratioReal_r1.Percentage - ratioPred_r1.Percentage).^2.*10000)),...
	sqrt(mean((ratioReal_r2.Percentage - ratioPred_r2.Percentage).^2.*10000)),...
	sqrt(mean((ratioReal_r3.Percentage - ratioPred_r3.Percentage).^2.*10000))];

SSE_Ratio = sum((ratioReal_r1.Percentage - ratioPred_r1.Percentage).^2);
SST_Ratio = sum((ratioReal_r1.Percentage - mean(ratioReal_r1.Percentage)).^2);
R2(j,1) = 1-SSE_Ratio./SST_Ratio;
SSE_Ratio = sum((ratioReal_r2.Percentage - ratioPred_r2.Percentage).^2);
SST_Ratio = sum((ratioReal_r2.Percentage - mean(ratioReal_r2.Percentage)).^2);
R2(j,2) = 1-SSE_Ratio./SST_Ratio;
SSE_Ratio = sum((ratioReal_r3.Percentage - ratioPred_r3.Percentage).^2);
SST_Ratio = sum((ratioReal_r3.Percentage - mean(ratioReal_r3.Percentage)).^2);
R2(j,3) = 1-SSE_Ratio./SST_Ratio;

AB2C(j,:) = [trapz(ratioPred_r1.CenTime,abs(ratioReal_r1.Percentage - ratioPred_r1.Percentage)),...
	trapz(ratioPred_r2.CenTime,abs(ratioReal_r2.Percentage - ratioPred_r2.Percentage)),...
	trapz(ratioPred_r3.CenTime,abs(ratioReal_r3.Percentage - ratioPred_r3.Percentage))];
end
F1Score = 2.*Precision.*Recall./(Precision+Recall);

segeRes = struct('Accuracy',mat2cell(accMat,size(accMat,1),ones(1,size(accMat,2))),...
    'Precision',mat2cell(Precision,size(accMat,1),ones(1,size(accMat,2))),'Recall',mat2cell(Recall,size(accMat,1),ones(1,size(accMat,2))),...
    'F1Score',mat2cell(F1Score,size(accMat,1),ones(1,size(accMat,2))),...
    'RMSE',mat2cell(RMSE,size(accMat,1),ones(1,size(accMat,2))),'R2',mat2cell(R2,size(accMat,1),ones(1,size(accMat,2))),'AB2C',mat2cell(AB2C,size(accMat,1),ones(1,size(accMat,2))),...
    'Accuracy_mean',mat2cell(mean(accMat,'omitnan'),1,ones(1,size(accMat,2))),'Accuracy_std',mat2cell(std(accMat,'omitnan'),1,ones(1,size(accMat,2))),...
    'Accuracy_Lo',mat2cell(prctile(accMat,10),1,ones(1,size(accMat,2))),'Accuracy_Up',mat2cell(prctile(accMat,90),1,ones(1,size(accMat,2))),...
    'Precision_mean',mat2cell(mean(Precision,'omitnan'),1,ones(1,size(accMat,2))),'Precision_std',mat2cell(std(Precision,'omitnan'),1,ones(1,size(accMat,2))),...
    'Precision_Lo',mat2cell(prctile(Precision,10),1,ones(1,size(accMat,2))),'Precision_Up',mat2cell(prctile(Precision,90),1,ones(1,size(accMat,2))),...
    'Recall_mean',mat2cell(mean(Recall,'omitnan'),1,ones(1,size(accMat,2))),'Recall_std',mat2cell(std(Recall,'omitnan'),1,ones(1,size(accMat,2))),...
    'Recall_Lo',mat2cell(prctile(Recall,10),1,ones(1,size(accMat,2))),'Recall_Up',mat2cell(prctile(Recall,90),1,ones(1,size(accMat,2))),...
    'F1Score_mean',mat2cell(mean(F1Score,'omitnan'),1,ones(1,size(accMat,2))),'F1Score_std',mat2cell(std(F1Score,'omitnan'),1,ones(1,size(accMat,2))),...
    'F1Score_Lo',mat2cell(prctile(F1Score,10),1,ones(1,size(accMat,2))),'F1Score_Up',mat2cell(prctile(F1Score,90),1,ones(1,size(accMat,2))),...
    'RMSE_mean',mat2cell(mean(RMSE,'omitnan'),1,ones(1,size(accMat,2))),'RMSE_std',mat2cell(std(RMSE,'omitnan'),1,ones(1,size(accMat,2))),...
    'RMSE_Lo',mat2cell(prctile(RMSE,10),1,ones(1,size(accMat,2))),'RMSE_Up',mat2cell(prctile(RMSE,90),1,ones(1,size(accMat,2))),...
    'R2_mean',mat2cell(mean(R2,'omitnan'),1,ones(1,size(accMat,2))),'R2_std',mat2cell(std(R2,'omitnan'),1,ones(1,size(accMat,2))),...
    'R2_Lo',mat2cell(prctile(R2,10),1,ones(1,size(accMat,2))),'R2_Up',mat2cell(prctile(R2,90),1,ones(1,size(accMat,2))),...
    'AB2C_mean',mat2cell(mean(AB2C,'omitnan'),1,ones(1,size(accMat,2))),'AB2C_std',mat2cell(std(AB2C,'omitnan'),1,ones(1,size(accMat,2))),...
    'AB2C_Lo',mat2cell(prctile(AB2C,10),1,ones(1,size(accMat,2))),'AB2C_Up',mat2cell(prctile(AB2C,90),1,ones(1,size(accMat,2))));
end