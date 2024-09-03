%% import PSS Dataset from source database
clear
warning('off');
%% set env
addpath '..\common';
addpath '..\data';
addpath '.\model';
%%
databasename = 'Porous 316L-tension test-0.01-AE-DIC-20210621';
filepath = '..\data';
cond = 'and counts>10 and Status = 0 and Chan=2 and Time<6200 ';

%%
[res_pri,res_tra,vecT,vecTR,fs,ssc_Time, ssc_Displacement, ssc_Force, ssc_Strain, ssc_Stress] ...
    = dataloader(filepath,databasename,cond);
%% Export
save('..\data\testing_data.mat');

