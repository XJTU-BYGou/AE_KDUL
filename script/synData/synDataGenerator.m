%%
warning('off');
% generate synthetic dataset
%
%%
clear
addpath '..\';
addpath '..\..\common';

%%
dateStr = datestr(datetime('now'),'yy-mm-dd_hh-MM-ss');
exportPath = fullfile('..\data\synData');
mkdir(exportPath);

%%
freq = [0:2e4:1e6];
%%
rng(0);
endTime = 13500;
sample1 = 1e4;
sample2 = 1e4;

% % uniform distribution
stTime = 200;
T = (endTime).*rand(sample1+sample2,1);
[sampleTime,rebID] = sort(T);
timePoints = [0,endTime];

%%
PSDstrategy = 2;
numClasses = 2;
for trendType = [1:3]
%%
domainPercen1 = 0.025;%0.05;
domainPercen2 = 0.025;%0.05;
synTime = [timePoints(1):timePoints(2)];

switch trendType
    case 1
%         timeScale = 0.5e3; p = 1;
        timeScale = 5e3; p = 1;
        k = (domainPercen1 - 1 + domainPercen2)./((timeScale - timePoints(1) + timePoints(2))^-p - (timeScale - timePoints(2) + timePoints(2))^-p);
        c = domainPercen1 - k.*(timeScale - timePoints(1) + timePoints(2))^-p;
        exCrackPercen = (timeScale - synTime + timePoints(2)).^-p.*k + c;
    case 2
        % linear
        exCrackPercen = (1 - domainPercen1 - domainPercen2)/(timePoints(2) - timePoints(1)) * synTime + domainPercen1;
    case 3
%         timeScale = 0.5e3; p = 1;
        timeScale = 5e3; p = 1;
        k = (domainPercen1 - 1 + domainPercen2)./((timeScale + timePoints(1) + timePoints(1))^-p - (timeScale + timePoints(2) + timePoints(1))^-p);
        c = domainPercen1 - k.*(timeScale + timePoints(1) + timePoints(1))^-p;
        exCrackPercen = (timeScale + synTime + timePoints(1)).^-p.*k + c;
    case 4
        exCrackPercen = 1.*(synTime>0.9.*sum(timePoints));
    case 5
        % Number of signals
        exCrackPercen =  (0.2)/(timePoints(2) - timePoints(1)) * synTime;
    case 6
        timeScale = 0.5e3; p = 1;
        exCrackPercen = (1 - ((1 - domainPercen1 - domainPercen2)/((timeScale + timePoints(1))^-p - (timeScale + timePoints(2))^-p) * (timePoints(1)+synTime+timeScale).^-p + ...
            - (timeScale + timePoints(2))^-p/((timeScale + timePoints(1))^-p - (timeScale + timePoints(2))^-p))).*0.1;
    case 7
        exCrackPercen = 0.4.*(synTime>0.75.*sum(timePoints));
end
        
%%
synLabel = ones(size(sampleTime));
rng(0);

switch trendType
    case 1
        crackPossi = (timeScale - sampleTime + timePoints(2)).^-p.*k + c;
    case 2
        % linear
        crackPossi = (1 - domainPercen1 - domainPercen2)/(timePoints(2) - timePoints(1)) * sampleTime + domainPercen1;
    case 3
        crackPossi = (timeScale + sampleTime + timePoints(1)).^-p.*k + c;
    case 4
        % linear
        crackPossi = 1.*(sampleTime>0.9.*sum(timePoints));
    case 5
        % Number of signals 9:1
        crackPossi =  (0.2)/(timePoints(2) - timePoints(1)) * sampleTime;
    case 6
        crackPossi = (1 - ((1 - domainPercen1 - domainPercen2)/((timeScale + timePoints(1))^-p - (timeScale + timePoints(2))^-p) * (timePoints(1)+sampleTime+timeScale).^-p + ...
            - (timeScale + timePoints(2))^-p/((timeScale + timePoints(1))^-p - (timeScale + timePoints(2))^-p))).*0.1;
    case 7
        crackPossi = 0.4.*(sampleTime>0.75.*sum(timePoints));
end
rng(0);
synLabel(rand(size(sampleTime))<crackPossi & sampleTime>timePoints(1) & sampleTime<=timePoints(2)) = 2;


%%

synAvgPSD_dislo = [1.0049823e-09,2.1998685e-09,8.3715657e-09,2.9287822e-08,...
    8.3073573e-08,1.7558074e-07,2.6631233e-07,3.0430891e-07,2.9097680e-07,...
    2.8840236e-07,3.1367398e-07,3.7453873e-07,4.3595841e-07,4.8203771e-07,...
    5.5437033e-07,7.1048447e-07,1.0296591e-06,1.4558334e-06,1.7448209e-06,...
    1.8448264e-06,1.6567944e-06,1.3271479e-06,1.1541354e-06,1.1636847e-06,...
    1.3567461e-06,1.8634503e-06,2.6389953e-06,3.2058690e-06,3.1870054e-06,...
    2.9342827e-06,2.9257230e-06,3.0169704e-06,2.7565343e-06,2.4536844e-06,...
    2.1787500e-06,1.8182284e-06,1.4120983e-06,9.8060093e-07,6.1299448e-07,...
    3.7699351e-07,2.3824992e-07,1.4480091e-07,7.9064556e-08,4.0498609e-08,...
    2.2196813e-08,1.4977914e-08,1.2154426e-08,1.0732733e-08,9.7545918e-09,...
    8.4061504e-09,6.5041541e-09];

synAvgPSD_crack = [6.9496298e-10,1.5876607e-09,6.4326886e-09,2.4481986e-08,...
    9.6267662e-08,8.2508876e-07,1.3508621e-06,1.3490703e-06,6.2608751e-07,...
    4.3797289e-07,2.9667126e-07,5.3024917e-07,1.0978329e-06,8.5315315e-07,...
    4.5528412e-07,1.0297781e-06,1.2911795e-06,1.3836448e-06,3.2669752e-06,...
    2.8152463e-06,2.2552938e-06,1.0832483e-06,7.6162678e-07,7.6282419e-07,...
    1.0110357e-06,1.6773423e-06,1.8428427e-06,2.2256643e-06,2.1758465e-06,...
    1.5715242e-06,1.8782538e-06,2.5249371e-06,1.7139001e-06,1.7821408e-06,...
    1.9661202e-06,2.1469234e-06,1.4174176e-06,1.0467937e-06,6.8398634e-07,...
    4.8211376e-07,4.4737862e-07,2.8573757e-07,1.7758595e-07,9.5153162e-08,...
    4.3339647e-08,2.9102063e-08,3.4097003e-08,5.5495430e-08,5.2838462e-08,...
    2.4845248e-08,1.2753463e-08];


rng(0);
tmpSampleNum = 4;
tmpSampleIndex = floor(size(synAvgPSD_crack,2)/tmpSampleNum)/2+1:floor(size(synAvgPSD_crack,2)/tmpSampleNum):size(synAvgPSD_crack,2);
synPSD_dislo_err = max(randn(sum(synLabel==1),tmpSampleNum+2).*0.5+1,0.01);
synPSD_dislo_err = mat2cell(synPSD_dislo_err,ones(size(synPSD_dislo_err,1),1),tmpSampleNum+2);
synPSD_dislo_err = cellfun(@(x)interp1([0,tmpSampleIndex,size(synAvgPSD_crack,2)+1],x,...
    [1:size(synAvgPSD_crack,2)]),synPSD_dislo_err,'UniformOutput',false);
synPSD_dislo_err = cell2mat(synPSD_dislo_err);

synPSD_crack_err = max(randn(sum(synLabel==2),tmpSampleNum+2).*0.5+1,0.01);
synPSD_crack_err = mat2cell(synPSD_crack_err,ones(size(synPSD_crack_err,1),1),tmpSampleNum+2);
synPSD_crack_err = cellfun(@(x)interp1([0,tmpSampleIndex,size(synAvgPSD_crack,2)+1],x,...
    [1:size(synAvgPSD_crack,2)]),synPSD_crack_err,'UniformOutput',false);
synPSD_crack_err = cell2mat(synPSD_crack_err);

switch PSDstrategy
    case 1
% Strategy 1 
synPSD_comb = zeros(numel(synLabel),size(synAvgPSD_crack,2));
    case 2
% Strategy 2 
synPSD_comb = min(abs(randn(numel(synLabel),tmpSampleNum+2)*0.35),0.99);
synPSD_comb = mat2cell(synPSD_comb,ones(size(synPSD_comb,1),1),tmpSampleNum+2);
synPSD_comb = cellfun(@(x)interp1([0,tmpSampleIndex,size(synAvgPSD_crack,2)+1],x,...
    [1:size(synAvgPSD_crack,2)]),synPSD_comb,'UniformOutput',false);
synPSD_comb = cell2mat(synPSD_comb);


diffPerc = trapz(1:size(synPSD_comb,2),synPSD_comb,2)./size(synPSD_comb,2);
synPSD_comb(diffPerc>=0.5,:) = synPSD_comb(diffPerc>=0.5,:) ./2;

end

synPSD_Dataset = zeros(size(sampleTime,1),numel(freq));
synPSD_Dataset(synLabel==1,:) = (synAvgPSD_dislo + synPSD_comb(synLabel==1,:) .* (synAvgPSD_crack - synAvgPSD_dislo)) .* synPSD_dislo_err;
synPSD_Dataset(synLabel==2,:) = (synAvgPSD_crack + synPSD_comb(synLabel==2,:) .* (synAvgPSD_dislo - synAvgPSD_crack)) .* synPSD_crack_err ;

synPSD_Dataset = synPSD_Dataset./trapz(freq,synPSD_Dataset,2);
synPSD_Dataset = log10(synPSD_Dataset);

%%

[XTrain,PCAProcessor] = getPCADataProcess(synPSD_Dataset,0.95);
[XTrain,norm_mu,norm_scale] = normalize(XTrain,1);
[numSample,numFeature] = size(XTrain);
YTrain = categorical(synLabel);

%%
save(fullfile(exportPath,['synDataset_PSDstrategy',num2str(PSDstrategy),'_Trend',num2str(trendType),'.mat']));

end