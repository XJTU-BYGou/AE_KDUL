clear
warning('off');
%% set env
addpath '..\common';
addpath '..\data';
addpath '.\model';
%% Import Data
% % define data from source .pridb , .tradb and .csv files 
% databasename = '';
% filepath = '..\data';
% cond = 'and counts>10 and Status = 0 ';
% [res_pri,res_tra,vecT,vecTR,fs,ssc_Time, ssc_Displacement, ssc_Force, ssc_Strain, ssc_Stress] ...
%     = dataloader(filepath,databasename,cond);

% % Import combined data - demo
load('training_data.mat','fs','res_pri','res_tra');
[vecT,vecTR] = dataoperator(res_tra,0);

%%
data_preprocess;
%%
reptNum = 100;
dateStr = datestr(datetime('now'),'yy-mm-dd_hh-MM-ss');
exportPath = fullfile('.\export',['res_trainMdl_',dateStr]);
mkdir(exportPath);
ClassifierNameCell = {'NN','LR','KLR','SVM','KSVM','GBDT','GDA'};
%%
predLoss = cell(1,numel(ClassifierNameCell));
%%
for n = 1:numel(ClassifierNameCell)
ClassifierName = ClassifierNameCell{n};
rng(0);
%% Training
switch ClassifierName
    case 'NN'
        layers = [
            featureInputLayer(numFeature,'Name','input')
            fullyConnectedLayer(10,'Name','fc11')
            leakyReluLayer('Name','relu1')
            dropoutLayer('Name','drop1')
            fullyConnectedLayer(5,'Name','fc12')
            leakyReluLayer('Name','relu2')
            dropoutLayer('Name','drop2')
            fullyConnectedLayer(2,'Name','fc13')
            softmaxLayer('Name','softmax')
            aggregatelosslayer('gradLoss',[res_pri.Time])];
        options = trainingOptions('adam','MiniBatchSize',numSample,'Shuffle','never',...
            'Plots','none','VerboseFrequency',10,'ExecutionEnvironment','gpu',...
            'InitialLearnRate',0.5e-3,'LearnRateSchedule','piecewise','LearnRateDropFactor',0.9,'LearnRateDropPeriod',2000,...
            'GradientThresholdMethod','l2norm','GradientThreshold',Inf,...
            'L2Regularization',0.0020,'ResetInputNormalization',false,...
            'MaxEpochs',2000,...
            'Verbose',1);
        [net,Info] = trainNetwork(XTrain,YTrain,layers,options);
        predScore = predict(net,XTrain);
    case 'LR'
        trainOptions = struct('MaxEpoch',30e2,'InitialLearnRate',10e-4,'GradientDecayFactor',0.9,...
            'LearnRateDropPeriod',5e2,'learnRateDropFactor',0.8,...
            'Verbose',true,'VerboseFrequency',50);
        [model,~,trainHistory,trainOptions] = trainLogisticRegression_rand([],XTrain,YTrain,...
            @(x)getAggregateLoss(x,[res_pri.Time]),...
            'options',trainOptions);
        [output,score] = appLogisticRegression(model.LR.Model,XTrain);
    case 'KLR'
        trainOptions = struct('MaxEpoch',5e2,'InitialLearnRate',5e-4,'GradientDecayFactor',0.9,...
            'LearnRateDropPeriod',5e2,'learnRateDropFactor',0.8,...
            'Verbose',true,'VerboseFrequency',50,'Optimizer','Adam');
        [model,~,trainHistory,trainOptions] = trainKernelLogisticRegression_rand([],XTrain,YTrain,...
            @(x)getAggregateLoss(x,[res_pri.Time]),...
            'options',trainOptions);
        [output,score] = appKernelLogisticRegression(model.KLR.Model,XTrain,XTrain);
    case 'SVM'
        trainOptions = struct('MaxEpoch',5e2,'InitialLearnRate',5e-4,'GradientDecayFactor',0.9,...
            'LearnRateDropPeriod',5e2,'learnRateDropFactor',0.8,...
            'Verbose',true,'VerboseFrequency',50);
        [model,~,trainHistory,trainOptions] = trainLinearSupportVectorMachine_rand_v2([],XTrain,YTrain,...
            @(x)getAggregateLoss(x,[res_pri.Time]),...
            'options',trainOptions);
        [output,score] = appPSVM(model.SVM.Model,XTrain);
    case 'KSVM'
        trainOptions = struct('MaxEpoch',5e2,'InitialLearnRate',5e-4,'GradientDecayFactor',0.9,...
            'LearnRateDropPeriod',5e2,'learnRateDropFactor',0.8,...
            'Verbose',true,'VerboseFrequency',50);
        [model,~,trainHistory,trainOptions] = trainKernelSupportVectorMachine_rand_v2([],XTrain,YTrain,...
            @(x)getAggregateLoss(x,[res_pri.Time]),...
            'options',trainOptions);
        [output,score] = appKernelSupportVectorMachine(model.KSVM.Model,XTrain,XTrain);
    case 'GBDT'
        trainOptions = struct('MaxEpoch',0.5e2,'InitialLearnRate',1e0,'GradientDecayFactor',0.9,...
            'LearnRateDropPeriod',5e1,'learnRateDropFactor',0.8,...
            'Verbose',true,'VerboseFrequency',10);
        [model,~,trainHistory,trainOptions] = trainGBDT_rand([],XTrain,YTrain,...
            @(x)getAggregateLoss(x,[res_pri.Time]),...
            'options',trainOptions);
        [output,score] = appGBDT(model.GBDT.Model,XTrain);
    case 'GDA'
        trainOptions = struct('MaxEpoch',5e2,'InitialLearnRate',1e0,'GradientDecayFactor',0.9,...
            'LearnRateDropPeriod',5e2,'learnRateDropFactor',0.8,...
            'Verbose',true,'VerboseFrequency',50);
        [model,~,trainHistory,trainOptions] = trainGaussianDiscriminative_rand([],XTrain,YTrain,...
            @(x)getAggregateLoss(x,[res_pri.Time]),...
            'options',trainOptions);
        [output,score] = appGaussianDiscriminative(model.GDA.Model,XTrain);
end
switch ClassifierName
    case 'NN'
        predScore = predScore';
        predLabel = double(classify(net,XTrain));
    otherwise
        if isdlarray(output)
            predScore = extractdata(output);
        else
            predScore = output;
        end
        [~,predLabel] = max(predScore,[],1);
        predLabel = reshape(predLabel,[],1);
end
%% Export Model
switch ClassifierName
    case 'NN'
        save(fullfile(exportPath,['res_NN.mat']),...
            'PCAProcessor','norm_mu','norm_scale',...
            'net','Info','layers','options','predLabel','predScore');
    case 'GBDT'
        save(fullfile(exportPath,['res_',ClassifierName,'.mat']),...
            'ClassifierName',...
            'model','trainHistory','trainOptions','output','predLabel','predScore',...
            '-v7.3');
    otherwise
        save(fullfile(exportPath,['res_',ClassifierName,'.mat']),...
            'ClassifierName',...
            'model','trainHistory','trainOptions','output','predLabel','predScore');
end
%% collect loss
predLoss{n} = arrayfun(@(x)getAggregateLoss(predScore,[res_pri.Time]),1:reptNum);
%% collect complexity of mdl
switch ClassifierName
    case 'NN'
        paramNum(n) = collectNumberLearnableParam(net.Layers);
    case 'LR'
        paramNum(n) = numel(model.LR.Model.W)+numel(model.LR.Model.B);
    case 'KLR'
        paramNum(n) = size(XTrain,1)+1;
    case 'SVM'
        paramNum(n) = numel(model.SVM.Model.weights.W) + numel(model.SVM.Model.weights.B);
    case 'KSVM'
        paramNum(n) = size(XTrain,1)+1;
    case 'GBDT'
        paramNum(n) = sum(cellfun(@(x)sum(x.IsBranchNode),model.GBDT.Model),'all');
    case  'GDA'
        paramNum(n) = numel(model.GDA.Model.Alpha) + numel(model.GDA.Model.Mu) +...
            (size(model.GDA.Model.Sigma,1)+1)*size(model.GDA.Model.Sigma,1)*size(model.GDA.Model.Sigma,3)/2;
end

end
%%
razorK = 0.1;
predOverallLoss = cellfun(@(x,y)getOverallPerformance(x,y,razorK),...
    predLoss,mat2cell(paramNum,size(paramNum,1),ones(size(paramNum))),'UniformOutput',false);
predOverallLoss_avg = cellfun(@(x)mean(single(x)),predOverallLoss);
predOverallLoss_std = cellfun(@(x)std(single(x)),predOverallLoss);

fig = figure;
ax1 = axes(fig,...
    'Color', 'none','Box','on',...
    'TickDir','in','Layer','top',...
    'LineWidth',2,...
    'FontName','Arial','FontSize',12,'FontWeight','bold');
hold on;
errorbar(1:numel(predOverallLoss_avg),predOverallLoss_avg,predOverallLoss_std,...
    'LineStyle','none','marker','o','markerFaceColor',[1 1 1],'LineWidth',2);
ax1.XLim = [0,numel(predOverallLoss_avg)]+0.5;
ax1.XTick = 1:numel(predOverallLoss_avg);
ax1.XTickLabel = ClassifierNameCell;
ylabel({'Overall Performance'},'FontName','Arial','FontSize',14,'FontWeight','bold');

save(fullfile(exportPath,['res_MdlEval.mat']),...
    'razorK','predOverallLoss','paramNum','predLoss',...
    'predOverallLoss_avg','predOverallLoss_std','ClassifierNameCell');

%% function set
function paramNum = collectNumberLearnableParam(layers)
paramNum = 0;
for ii = 1:size(layers,1)
    props = properties(layers(ii));
    for p = 1:numel(props)
        propName = props{p};
        if strcmp(propName,'Weights') 
            paramNum = paramNum + numel(layers(ii).Weights);
        elseif strcmp(propName,'Bias')
            paramNum = paramNum + numel(layers(ii).Bias);
        end
    end
end
end