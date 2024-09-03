%% import PSS Dataset from source database
clear
warning('off');
%% set env
addpath '..\common';
addpath '..\data';
addpath '.\model';
%%
databasename = 'PSS-1-tension test-3-z1-AE-20190217';
filepath = '..\data';
cond = 'and counts>10 and Status = 0 and Time<15400 ';

%%
[res_pri,res_tra,vecT,vecTR,fs,ssc_Time, ssc_Displacement, ssc_Force, ssc_Strain, ssc_Stress] ...
    = dataloader(filepath,databasename,cond);
%% Export
save('..\data\training_data.mat');

