function [res_pri,res_tra,vecT,vecTR,fs,ssc_Time, ssc_Displacement, ssc_Force, ssc_Strain, ssc_Stress] ...
    = dataloader(filepath,databasename,cond)
%% set env
addpath '..\common';
addpath '..\data';
%%
res_pri = importaedata(fullfile(filepath,databasename),1,['WHERE SetType = 2 and TRAI>0 ',cond]);
res_tra = importSpecialWaveformData(res_pri,filepath,databasename);

dB0 = 20*log10(res_tra(1).Thr);
fs = res_tra(1).SampleRate;
[vecT,vecTR] = dataoperator(res_tra,0);

[ssc_Time, ssc_Displacement, ssc_Force, ssc_Strain, ssc_Stress] = importSSCData(...
fullfile(filepath,[databasename,'.is_tens_RawData'],'Specimen_RawData_2.csv'),[3, Inf]);

end