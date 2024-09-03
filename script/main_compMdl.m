%%
clear
rng(0);
warning('off');
%%
addpath '..\common'
addpath '..\data'
addpath '.\model';
addpath '.\trainedModel';

%% Import Data
load('training_data.mat','fs','res_pri','res_tra','ssc_Time','ssc_Stress');
[vecT,vecTR] = dataoperator(res_tra,0);
%%
data_preprocess;
timePoints = [2000,11060];
regimeLabel = getLabelWith3Regim([res_pri.Time],timePoints);
%%
reptNum = 100;
%%
modelIndex = 1;
% 1 The proposed NN
cName{modelIndex} = 'BPNN';
load('.\trainedModel\res_NN_2.mat',...
    'net','predLabel','predScore','Info');
predLabelCell{modelIndex} = predLabel;
predScoreCell{modelIndex} = predScore';
predLoss{modelIndex} = arrayfun(@(x)getAggregateLoss(predScoreCell{modelIndex},[res_pri.Time]),1:reptNum);
predLoss_avg(modelIndex) = mean(predLoss{modelIndex});
predLoss_std(modelIndex) = std(predLoss{modelIndex});
paramNum(modelIndex) = numel(net.Layers(2).Weights) + numel(net.Layers(2).Bias) + ...
    numel(net.Layers(5).Weights) + numel(net.Layers(5).Bias) + ...
    numel(net.Layers(8).Weights) + numel(net.Layers(8).Bias); % 237 

modelIndex = 2;
% 2 Logistic Regression
cName{modelIndex} = 'Logistic Regression';
load('.\trainedModel\res_LR.mat',...
    'model','trainHistory');
predScore = extractdata(appLogisticRegression(model.LR.Model,XTrain));
[~,predLabelCell{modelIndex}] = max(predScore,[],1);
predScoreCell{modelIndex} = predScore;
predLoss{modelIndex} = arrayfun(@(x)getAggregateLoss(predScoreCell{modelIndex},[res_pri.Time]),1:reptNum);
predLoss_avg(modelIndex) = mean(predLoss{modelIndex});
predLoss_std(modelIndex) = std(predLoss{modelIndex});
paramNum(modelIndex) = numel(model.LR.Model.W)+numel(model.LR.Model.B); % 17

modelIndex = 3;
cName{modelIndex} = 'Gaussian Discriminant Analysis';
load('.\trainedModel\res_GDA.mat',...
    'model','trainHistory');
predScore = appGaussianDiscriminative(model.GDA.Model,XTrain);
predScore = 1 - predScore;
[~,predLabelCell{modelIndex}] = max(predScore,[],1);
predScoreCell{modelIndex} = predScore;
predLoss{modelIndex} = arrayfun(@(x)getAggregateLoss(predScoreCell{modelIndex},[res_pri.Time]),1:reptNum);
predLoss_avg(modelIndex) = mean(predLoss{modelIndex});
predLoss_std(modelIndex) = std(predLoss{modelIndex});
paramNum(modelIndex) = numel(model.GDA.Model.Alpha) + numel(model.GDA.Model.Mu) +...
    (size(model.GDA.Model.Sigma,1)+1)*size(model.GDA.Model.Sigma,1)*size(model.GDA.Model.Sigma,3)/2; % 306

modelIndex = 4;
% 4 Kernel SVM -rbf
cName{modelIndex} = 'Support Vector Machine';
load('.\trainedModel\res_KSVM.mat',...
    'model','trainHistory');
predScore = extractdata(appKernelSupportVectorMachine(model.KSVM.Model,XTrain,XTrain));
[~,predLabelCell{modelIndex}] = max(predScore,[],1);
predScoreCell{modelIndex} = predScore;
predLoss{modelIndex} = arrayfun(@(x)getAggregateLoss(predScoreCell{modelIndex},[res_pri.Time]),1:reptNum);
predLoss_avg(modelIndex) = mean(predLoss{modelIndex});
predLoss_std(modelIndex) = std(predLoss{modelIndex});
paramNum(modelIndex) = size(XTrain,1)+1;

modelIndex = 5;
% 5 GBDT
cName{modelIndex} = 'GBDT';
load('.\trainedModel\res_GBDT.mat',...
    'model','trainHistory');
predScore = appGBDT(model.GBDT.Model,XTrain);
[~,predLabelCell{modelIndex}] = max(predScore,[],1);
predScoreCell{modelIndex} = predScore;
predLoss{modelIndex} = arrayfun(@(x)getAggregateLoss(predScoreCell{modelIndex},[res_pri.Time]),1:reptNum);
predLoss_avg(modelIndex) = mean(predLoss{modelIndex});
predLoss_std(modelIndex) = std(predLoss{modelIndex});
paramNum(modelIndex) = sum(cellfun(@(x)sum(x.IsBranchNode),model.GBDT.Model),'all');

% Peer machine learning method
modelIndex = 6;
% 6 unsupervised Cluster
cName{modelIndex} = 'Clustering';
load('.\trainedModel\res_Kmeans.mat',...
    'idx')
predLabel = idx{2};
predLabelCell{modelIndex} = 3 - predLabel;
predScoreCell{modelIndex} = predScore';
predLoss{modelIndex} = nan;
predLoss_avg(modelIndex) = mean(predLoss{modelIndex});
predLoss_std(modelIndex) = std(predLoss{modelIndex});
paramNum(modelIndex) = nan; % 237 

modelIndex = 7;
% 7 unsupervised ScaNet
cName{modelIndex} = 'ScaNet';
load('.\trainedModel\res_scaNet_PSS.mat',...
    'predLabel')
predScore = predLabel'==[1;2];
predLabelCell{modelIndex} = double(predLabel)';
predScoreCell{modelIndex} = predScore;
predLoss{modelIndex} = nan;
predLoss_avg(modelIndex) = mean(predLoss{modelIndex});
predLoss_std(modelIndex) = std(predLoss{modelIndex});
paramNum(modelIndex) = numel(net.Layers(2).Weights) + numel(net.Layers(2).Bias) + ...
    numel(net.Layers(5).Weights) + numel(net.Layers(5).Bias) + ...
    numel(net.Layers(8).Weights) + numel(net.Layers(8).Bias); 


modelIndex = 8;
% 8 supervised NN 
cName{modelIndex} = 'Pseudo-label NN';
load('.\trainedModel\res_surpervised_PLNN.mat',...
    'net','predLabel','predScore','Info');
predLabelCell{modelIndex} = predLabel;
predScoreCell{modelIndex} = predScore';
predLoss{modelIndex} = nan;
predLoss_avg(modelIndex) = mean(predLoss{modelIndex});
predLoss_std(modelIndex) = std(predLoss{modelIndex});
paramNum(modelIndex) = numel(net.Layers(2).Weights) + numel(net.Layers(2).Bias) + ...
    numel(net.Layers(5).Weights) + numel(net.Layers(5).Bias) + ...
    numel(net.Layers(8).Weights) + numel(net.Layers(8).Bias); % 237 


modelIndex = 9;
% 9 supervised NN with focal loss
cName{modelIndex} = 'Pseudo-label Focal Loss NN';
load('.\trainedModel\res_surpervised_IDNN.mat',...
    'net','predLabel','predScore','Info');
predLabelCell{modelIndex} = predLabel;
predScoreCell{modelIndex} = predScore';
predLoss{modelIndex} = nan;
predLoss_avg(modelIndex) = mean(predLoss{modelIndex});
predLoss_std(modelIndex) = std(predLoss{modelIndex});
paramNum(modelIndex) = numel(net.Layers(2).Weights) + numel(net.Layers(2).Bias) + ...
    numel(net.Layers(5).Weights) + numel(net.Layers(5).Bias) + ...
    numel(net.Layers(8).Weights) + numel(net.Layers(8).Bias); % 237 


%%
res_pri_dislocation = res_pri(double(predLabelCell{1})==1);
res_pri_crack = res_pri(double(predLabelCell{1})==2);
[superJerkIDvec_crack,expMLEstimator_crack,enyExpFlag_crack,err_crack] = autoSuperjerkEstimtor(res_pri_crack,false,true);
[superJerkIDvec_dislocation,expMLEstimator_dislocation,enyExpFlag_dislocation,err_dislocation] = autoSuperjerkEstimtor(res_pri_dislocation,false,true);

enyExpFlag_crack(end-1) = false;
enyExpFlag_dislocation([4:6,9,12,16,18,19]) = false;
enyExpFlag_crack(5) = false;

%%
exportPath = '.\export';
mkdir(exportPath);
save(fullfile(exportPath,'data_Fig2.mat'),...
    'res_pri','ssc_Time','ssc_Stress','timePoints',...
    'res_pri_dislocation','superJerkIDvec_dislocation','expMLEstimator_dislocation','enyExpFlag_dislocation','err_dislocation',...
    'res_pri_crack','superJerkIDvec_crack','expMLEstimator_crack','enyExpFlag_crack','err_crack');

%%
[res_train,record_train] = cellfun(@(x)AEMixExponent([res_pri.Eny],x),predLabelCell,'UniformOutput',false);
[res_r2_train,record_r2_train] = cellfun(@(x)AEMixExponent([res_pri(regimeLabel==0).Eny],double(x(regimeLabel==0))),predLabelCell,'UniformOutput',false);
[res_all,record_all] = powerlawExponentMLEstimator([res_pri.Eny],[]);
[res_r2_all,record_r2_all] = powerlawExponentMLEstimator([res_pri(regimeLabel==0).Eny],[]);

rng(0);
predLabel_compRand = randsample([1;2],numel(res_pri),true);
[res_compRand,record_compRand] = AEMixExponent([res_pri.Eny],predLabel_compRand);

%%
cName{1} = 'The proposed method';
cName{2} = 'LR';
cName{3} = 'GDA';
cName{4} = 'SVM';
cName{5} = 'GBDT';
cName{6} = 'Kmeans';
cName{7} = 'ScaNet';
cName{8} = 'PLNN';
cName{9} = 'IDNN';

%% Expert results

resLabel{1}(1).StIndex = 12532;  resLabel{1}(1).EdIndex = 12728; resLabel{1}(1).PlantFlag = true;
resLabel{1}(2).StIndex = 24;  resLabel{1}(2).EdIndex = 308; resLabel{1}(2).PlantFlag = true;
resLabel{2}(1).StIndex = 11944;  resLabel{2}(1).EdIndex = 12026; resLabel{2}(1).PlantFlag = true;
resLabel{2}(2).StIndex = 223;  resLabel{2}(2).EdIndex = 698; resLabel{2}(2).PlantFlag = true;
resLabel{3}(1).StIndex = 0;  resLabel{3}(1).EdIndex = 0; resLabel{3}(1).PlantFlag = false;
resLabel{3}(2).StIndex = 0;  resLabel{3}(2).EdIndex = 0; resLabel{3}(2).PlantFlag = false;
resLabel{4}(1).StIndex = 12656;  resLabel{4}(1).EdIndex = 12719; resLabel{4}(1).PlantFlag = true;
resLabel{4}(2).StIndex = 105;  resLabel{4}(2).EdIndex = 190; resLabel{4}(2).PlantFlag = true;
resLabel{5}(1).StIndex = 12056;  resLabel{5}(1).EdIndex = 12161; resLabel{5}(1).PlantFlag = true;
resLabel{5}(2).StIndex = 62;  resLabel{5}(2).EdIndex = 612; resLabel{5}(2).PlantFlag = true;
resLabel{6}(1).StIndex = 0;  resLabel{6}(1).EdIndex = 0; resLabel{6}(1).PlantFlag = false;
resLabel{6}(2).StIndex = 0;  resLabel{6}(2).EdIndex = 0; resLabel{6}(2).PlantFlag = false;
resLabel{7}(1).StIndex = 0;  resLabel{7}(1).EdIndex = 0; resLabel{7}(1).PlantFlag = false;
resLabel{7}(2).StIndex = 0;  resLabel{7}(2).EdIndex = 0; resLabel{7}(2).PlantFlag = false;
resLabel{8}(1).StIndex = 0;  resLabel{8}(1).EdIndex = 0; resLabel{8}(1).PlantFlag = false;
resLabel{8}(2).StIndex = 0;  resLabel{8}(2).EdIndex = 0; resLabel{8}(2).PlantFlag = false;
resLabel{9}(1).StIndex = 12824;  resLabel{9}(1).EdIndex = 12885; resLabel{9}(1).PlantFlag = true;
resLabel{9}(2).StIndex = 8;  resLabel{9}(2).EdIndex = 79; resLabel{9}(2).PlantFlag = true;

resLabel_train = resLabel;
%%
platLen_train = cellfun(@(x,y)getPlatformLen(x,y),record_train,resLabel_train,'UniformOutput',false);
perf = platLen_train;
perfValue = cellfun(@(x)2*prod(x)/(sum(x)+1e-8),perf,'UniformOutput',true);

predLoss = cellfun(@single,predLoss,'UniformOutput',false);

outlierCutOff = [0.5,0.6,0.678,0,0.655,0,7.6,0,0.55];
predLoss_eff = cellfun(@(x,y)x(x>y),predLoss,num2cell(outlierCutOff),'UniformOutput',false);
predLoss_avg = cellfun(@mean,predLoss_eff);
predLoss_std = cellfun(@std,predLoss);

razorK = 0.1;

predOverallLoss_avg = getOverallPerformance(predLoss_avg,paramNum,razorK);
predOverallLoss = cellfun(@(x,y) getOverallPerformance(x,y,razorK),...
    predLoss,mat2cell(paramNum,size(paramNum,1),ones(size(paramNum))),'UniformOutput',false);
predOverallLoss_std = cellfun(@(x)std(single(x)),predOverallLoss);

%%
exportPath = '.\export';
mkdir(exportPath);
save(fullfile(exportPath,'data_Fig3a.mat'),...
    'perfValue','paramNum','cName',...
    'predOverallLoss_avg','predOverallLoss_std','predLoss_avg','predLoss_std');

%%
tInt = 1200;
ratioCell = cellfun(@(x)avgPercentageCalculation(res_pri,double(x),tInt,'overlap',1000),...
    predLabelCell,'UniformOutput',false);

exportPath = '.\export';
mkdir(exportPath);
save(fullfile(exportPath,'data_Fig3b.mat'),...
    'ratioCell');

%% VS. Statistic & clustering
predLabel_NN = predLabelCell{1};

[res_stat(1,1),record_stat(1,1)] = powerlawExponentMLEstimator([res_pri(double(regimeLabel)==1).Eny]);
[res_stat(2,1),record_stat(2,1)] = powerlawExponentMLEstimator([res_pri(double(regimeLabel)==2).Eny]);
[res_NN(1,1),record_NN(1,1)] = powerlawExponentMLEstimator([res_pri(double(predLabel_NN)==1).Eny]);
[res_NN(2,1),record_NN(2,1)] = powerlawExponentMLEstimator([res_pri(double(predLabel_NN)==2).Eny]);

predLabel_Kmeans = predLabelCell{6};

[res_kmeans(1,1),record_kmeans(1,1)] = powerlawExponentMLEstimator([res_pri(double(predLabel_Kmeans)==1).Eny]);
[res_kmeans(2,1),record_kmeans(2,1)] = powerlawExponentMLEstimator([res_pri(double(predLabel_Kmeans)==2).Eny]);

[res_r1,record_r1] = powerlawExponentMLEstimator([res_pri(regimeLabel==1).Eny]);
[res_r3,record_r3] = powerlawExponentMLEstimator([res_pri(regimeLabel==2).Eny]);
[res_r13NN(1,1),record_r13NN(1,1)] = powerlawExponentMLEstimator([res_pri(regimeLabel~=0 & double(predLabel_NN)==1).Eny]);
[res_r13NN(2,1),record_r13NN(2,1)] = powerlawExponentMLEstimator([res_pri(regimeLabel~=0 & double(predLabel_NN)==2).Eny]);

[res_r1NN(1,1),record_r1NN(1,1)] = powerlawExponentMLEstimator([res_pri(regimeLabel==1 & double(predLabel_NN)==1).Eny]);
[res_r3NN(1,1),record_r3NN(1,1)] = powerlawExponentMLEstimator([res_pri(regimeLabel==2 & double(predLabel_NN)==2).Eny]);

[res_r2,record_r2] = powerlawExponentMLEstimator([res_pri(~regimeLabel).Eny]);
[res_r2NN(1,1),record_r2NN(1,1)] = powerlawExponentMLEstimator([res_pri(regimeLabel==0 & double(predLabel_NN)==1).Eny]);
[res_r2NN(2,1),record_r2NN(2,1)] = powerlawExponentMLEstimator([res_pri(regimeLabel==0 & double(predLabel_NN)==2).Eny]);

exportPath = '.\export';
mkdir(exportPath);
save(fullfile(exportPath,'data_Fig3c.mat'),...
    'record_r1','record_r3','record_r1NN','record_r3NN');
save(fullfile(exportPath,'data_Fig3d.mat'),...
    'record_r2','record_r2NN');

%% Fracture theory
clear TimeCell AmpCell TimeFittingCell AmpFittingCell fityCell gofMat fittingY
for i = 1:numel(predLabelCell)
TimeCell{i} = [res_pri(double(predLabelCell{i})==2).Time];
AmpCell{i} = [res_pri(double(predLabelCell{i})==2).Amp];

selFlag = TimeCell{i}<14600;
TimeFittingCell{i} = TimeCell{i}(selFlag);
AmpFittingCell{i} = AmpCell{i}(selFlag);

rng(0);
if ~isempty(TimeCell{i})
% [fityCell{i},gofMat(i,:)] = fitTBCModel(TimeFittingCell{i},cumsum(AmpFittingCell{i}));
[fityCell{i},gofMat(i,:)] = fitNormTBCModel(TimeFittingCell{i},cumsum(AmpFittingCell{i}));
fittingY{i} = appNormTBCModel(TimeFittingCell{i},fityCell{i})./sum(AmpCell{i});
else
    tmpfityStruct.a = nan;  tmpfityStruct.b = nan;  tmpfityStruct.c = nan;
    tmpfityStruct.d = nan;  tmpfityStruct.e = nan;  tmpfityStruct.f = nan;
    tmpfityStruct.g = nan;
    fityCell{i} = tmpfityStruct;
    fittingY{i} = nan;
    tmpGofStruct.sse = nan;  tmpGofStruct.rsquare = nan;  tmpGofStruct.dfe = nan;
    tmpGofStruct.adjrsquare = nan;  tmpGofStruct.rmse = nan;
    gofMat(i,:) = tmpGofStruct;
end
clear tmpfityStruct tmpGofStruct
end

exportPath = '.\export';
mkdir(exportPath);
save(fullfile(exportPath,'data_Fig3e.mat'),...
    'cName','AmpCell','TimeCell','TimeFittingCell','fityCell','fittingY');

%% VS ScaNet
predLabel_ScaNet = reshape(predLabelCell{7},[],1);

rng(1);
[res_NN(1,1),record_NN(1,1)] = powerlawExponentMLEstimator([res_pri(double(predLabel_NN)==1).Eny]);
[res_NN(2,1),record_NN(2,1)] = powerlawExponentMLEstimator([res_pri(double(predLabel_NN)==2).Eny]);

[res_scanet(1,1),record_scanet(1,1)] = powerlawExponentMLEstimator([res_pri(double(predLabel_ScaNet)==1).Eny]);
[res_scanet(2,1),record_scanet(2,1)] = powerlawExponentMLEstimator([res_pri(double(predLabel_ScaNet)==2).Eny]);

exportPath = '.\export';
mkdir(exportPath);
save(fullfile(exportPath,'data_FigS5.mat'),...
    'record_scanet','record_NN');

%% VS. Memory
load('.\trainedModel\dataMigration_res.mat', 'approximateLabelCell', 'predTestLabel','res_pri_test');
predLabel_Memory = approximateLabelCell{3};

[res_memory(1,1),record_memory(1,1)] = powerlawExponentMLEstimator([res_pri_test(double(predLabel_Memory)==1).Eny]);
[res_memory(2,1),record_memory(2,1)] = powerlawExponentMLEstimator([res_pri_test(double(predLabel_Memory)==2).Eny]);
[res_NN_test(1,1),record_NN_test(1,1)] = powerlawExponentMLEstimator([res_pri_test(double(predTestLabel)==1).Eny]);
[res_NN_test(2,1),record_NN_test(2,1)] = powerlawExponentMLEstimator([res_pri_test(double(predTestLabel)==2).Eny]);
res_pri_dislocation_memory = res_pri_test(double(predLabel_Memory)==1);
res_pri_crack_memory = res_pri_test(double(predLabel_Memory)==2);


load('..\data\testing_data.mat','vecTR');
load('.\trainedModel\res_NN_2.mat','net','fs','PCAProcessor','norm_mu','norm_scale');
data_preprocess_testing;
predTestLabel = classify(net,XTest);
res_pri_dislocation = res_pri_test(double(predTestLabel)==1);
res_pri_crack = res_pri_test(double(predTestLabel)==2);

exportPath = '.\export';
mkdir(exportPath);
save(fullfile(exportPath,'data_FigS2.mat'),...
    'res_pri_dislocation','res_pri_crack','res_pri_dislocation_memory','res_pri_crack_memory',...
    'record_memory','record_NN_test','predLabel_Memory','predTestLabel');

%%
%% Function Set -------------------------------------
function len = getPlatformLen(record,reslabel)
% get H_MLE
len = zeros(numel(record),1);
for i = 1:numel(record)
    if reslabel(i).PlantFlag
        len(i) = diff(log10(record(i).Xmin([reslabel(i).StIndex,reslabel(i).EdIndex])));
    else
        len(i) = 0;
    end
end
end

function overallPerf = getOverallPerformance(loss,Nc,razorK)
overallPerf = (1./(loss)) .* (1 - razorK) + razorK.* (1./(Nc));
end


function [ratio,stTime] = fastAvgRatioCalculation(res_pri,label,tInt)

overlapInt = tInt*1/2;
stTime = 0:tInt - overlapInt:[res_pri(end).Time];
edTime = tInt - overlapInt:tInt - overlapInt:[res_pri(end).Time] + tInt - overlapInt;
ratio = sum(double(label).*([res_pri.Time]' >= stTime & [res_pri.Time]' <edTime))./sum([res_pri.Time]' >= stTime & [res_pri.Time]' <edTime) - 1;

end

function [fity,gof,output] = fitTBCModel(x,y,varargin)

if mod(length(varargin),2) ~= 0 
    error('Input Error.')
end

% default Parameter
aL = 1e0;
aU = 1e10;
a0 = max(y);
bL = -1e6;
bU = -1e-6;
b0 = -1;
cL = 0;
cU = 10;
c0 = 1;
dL = max(x) + 1;
dU = max(x)*5;
d0 = max(x) + 1;
eL = 1e-1;
eU = 1e5;
e0 = 5;
fL = 0;
fU = 2*pi;
f0 = 0;
gL = 0.1;
gU = 5;
g0 = 1;


for i = 1:length(varargin)/2
    switch varargin{i*2-1} 
        case 'a'
        aL = varargin{i*2};
        aU = varargin{i*2};
        a0 = varargin{i*2};
        case 'b'
        bL = varargin{i*2};
        bU = varargin{i*2};
        b0 = varargin{i*2};
        case 'c'
        cL = varargin{i*2};
        cU = varargin{i*2};
        c0 = varargin{i*2};
        case 'd'
        dL = varargin{i*2};
        dU = varargin{i*2};
        d0 = varargin{i*2};
        case 'e'
        eL = varargin{i*2};
        eU = varargin{i*2};
        e0 = varargin{i*2};
        case 'f'
        fL = varargin{i*2};
        fU = varargin{i*2};
        f0 = varargin{i*2};
        case 'g'
        gL = varargin{i*2};
        gU = varargin{i*2};
        g0 = varargin{i*2};
    end
end

x = reshape(x,[],1);
y = reshape(y,[],1);




%     logX = log(x);
%     logY = log(y);
    fo = fitoptions('Method','NonlinearLeastSquares',...
               'Lower',[aL,bL,cL,dL,eL,fL,gL],...
               'Upper',[aU,bU,cU,dU,eU,fU,gU],...
               'StartPoint',[a0,b0,c0,d0,e0,f0,g0]);

    f = fittype(['a + b.*(1 + c.*cos(2.*pi.*log(d-x)./e + f)).*(d-x).^g'],'options',fo);
    [fity,gof,output] = fit(x,y,f);
    

end

function [fity,gof,output] = fitNormTBCModel(x,y,varargin)

if mod(length(varargin),2) ~= 0 
    error('Input Error.')
end
safetyTime = 200;

% default Parameter
aL = 1e0;
aU = 1e10;
a0 = max(y);
cL = 0;
cU = 10;
c0 = 1;
dL = 14600;
dU = 1.52e4;
d0 = 14600;
eL = 1e-1;
eU = 1e5;
e0 = 5;
fL = -2*pi;
fU = 2*pi;
f0 = 0;
gL = 0.01;
gU = 0.8;
g0 = 0.3;
% g0 = 1;

for i = 1:length(varargin)/2
    switch varargin{i*2-1} 
        case 'a'
            % base line
        aL = varargin{i*2};
        aU = varargin{i*2};
        a0 = varargin{i*2};
        case 'b'
            % scale
        bL = varargin{i*2};
        bU = varargin{i*2};
        b0 = varargin{i*2};
        case 'c'
        cL = varargin{i*2};
        cU = varargin{i*2};
        c0 = varargin{i*2};
        case 'd'
            % critical time
        dL = varargin{i*2};
        dU = varargin{i*2};
        d0 = varargin{i*2};
        case 'e'
            % log(\lambda)
        eL = varargin{i*2};
        eU = varargin{i*2};
        e0 = varargin{i*2};
        case 'f'
        fL = varargin{i*2};
        fU = varargin{i*2};
        f0 = varargin{i*2};
        case 'g'
            % exponent
        gL = varargin{i*2};
        gU = varargin{i*2};
        g0 = varargin{i*2};
    end
end

x = reshape(x,[],1);
y = reshape(y,[],1);



%     logX = log(x);
%     logY = log(y);
    fo = fitoptions('Method','NonlinearLeastSquares',...
               'Lower',[aL,cL,dL,eL,fL,gL],...
               'Upper',[aU,cU,dU,eU,fU,gU],...
               'StartPoint',[a0,c0,d0,e0,f0,g0],...
               'MaxIter',5e3);

    f = fittype(['a.*(1 - (1 + c.*cos(2.*pi.*log(d-x)./e + f))./(1 + c.*cos(2.*pi.*log(d)./e + f)).*((d-x)./d).^g)'],'options',fo);
    [fity,gof,output] = fit(x,y,f);
    
end
function y = appNormTBCModel(x,fity)
    y = fity.a.*(1 - (1 + fity.c.*cos(2.*pi.*log(fity.d-x)./fity.e + fity.f))...
        ./(1 + fity.c.*cos(2.*pi.*log(fity.d)./fity.e + fity.f)).*((fity.d-x)./fity.d).^fity.g);
end

function res = avgPercentageCalculation(res_pri,label,tInt,varargin)
label = reshape(double(label),[],1);
overlapInt = tInt*1/2;

for i = 1:length(varargin)/2
    switch lower(varargin{i*2-1})
        case 'overlap'
            overlapInt = varargin{i*2};
    end
end

stTime = 0:tInt - overlapInt:[res_pri(end).Time];
edTime = stTime + tInt;
cenTime = (stTime + edTime) ./2;
ratio = sum(label.*([res_pri.Time]' >= stTime & [res_pri.Time]' <edTime))./sum([res_pri.Time]' >= stTime & [res_pri.Time]' <edTime) - 1;

reptNum = 200;
ratioRept = nan(reptNum,numel(stTime));
for i = 1:numel(stTime)
    tmpLabel = label([res_pri.Time]' >= stTime(i) & [res_pri.Time]' <edTime(i));
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
