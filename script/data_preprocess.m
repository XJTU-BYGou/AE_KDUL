%% Data preprocessing
% Fake labels for placeholder
placeholder_label = randsample(2,numel(res_pri),true);

% Get PSD
freq = [0:2e4:1e6];
PSD_Dataset = wave2psd(vecTR,fs,freq);
PSD_Dataset = log10(PSD_Dataset);

[XTrain,PCAProcessor] = getPCADataProcess(PSD_Dataset,0.95);
[XTrain,norm_mu,norm_scale] = normalize(XTrain,1);
[numSample,numFeature] = size(XTrain);
YTrain = categorical(placeholder_label);


