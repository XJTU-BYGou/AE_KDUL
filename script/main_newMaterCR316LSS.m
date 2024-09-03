clear
warning('off');
%% set env
addpath '..\common';
addpath '..\data';
addpath '.\model';
addpath '.\trainedModel';

%% Import Data
% % define data from source .pridb , .tradb and .csv files 
% databasename = '316L-1.5-z1-AE-20180921';
% filepath = '..\..\data';
% cond = 'and counts>=6 and Status = 0 ';
% [res_pri,res_tra,vecT,vecTR,fs,ssc_Time, ssc_Displacement, ssc_Force, ssc_Strain, ssc_Stress] ...
%     = dataloader(filepath,databasename,cond);

% % Import combined data
load('CR316LSS_data.mat','fs','res_pri','res_tra','ssc_Time','ssc_Stress');
load('.\trainedModel\res_NN_316LSS.mat','net','PCAProcessor','norm_mu','norm_scale','label');
[vecT,vecTR] = dataoperator(res_tra,0);

%% 
ssc_Time_corr = ssc_Time([1:80825,81276:end]);
ssc_Stress_corr = ssc_Stress([1:80825,81276:end]);
data_preprocess_testing;
%%
predLabel = classify(net,XTest);
predScore = predict(net,XTest);

%%
[MLEres_ori1,MLErecord_ori1] = powerlawExponentMLEstimator([res_pri(label==1).Eny],downSamplingFromPowerLaw([res_pri(label==1).Eny],10.^[-1:0.08:6]));
[MLEres_ori2,MLErecord_ori2] = powerlawExponentMLEstimator([res_pri(label==2).Eny],downSamplingFromPowerLaw([res_pri(label==1).Eny],10.^[-1:0.08:6]));
[MLEres_our1,MLErecord_our1] = powerlawExponentMLEstimator([res_pri(double(predLabel)==2).Eny],downSamplingFromPowerLaw([res_pri(label==1).Eny],10.^[-1:0.08:6]));
[MLEres_our2,MLErecord_our2] = powerlawExponentMLEstimator([res_pri(double(predLabel)==1).Eny],downSamplingFromPowerLaw([res_pri(label==1).Eny],10.^[-1:0.08:6]));
[MLEres_all,MLErecord_all] = powerlawExponentMLEstimator([res_pri.Eny],downSamplingFromPowerLaw([res_pri(label==1).Eny],10.^[-1:0.08:6]));

label = reshape(label,[],1);
predLabel = 3 - double(predLabel);

%% Save result
exportPath = '.\export';
mkdir(exportPath)
save(fullfile(exportPath,'data_FigS7.mat'),...
    'res_pri','ssc_Time_corr','ssc_Stress_corr','label','predLabel',...
    'MLErecord_ori1','MLErecord_ori2','MLErecord_our1','MLErecord_our2');
