% Data preprocessing for new Dataset
% Get PSD
freq = [0:2e4:1e6];
PSD_Dataset = wave2psd(vecTR,fs,freq);
PSD_Dataset = log10(PSD_Dataset);

[XTest] = appPCAProcess(PSD_Dataset,PCAProcessor);
[XTest] = (XTest-norm_mu)./norm_scale;
[numTestSimple] = size(XTest,1);