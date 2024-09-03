%%
clear
warning('off');
%% set env
addpath '..\..\..\common';
addpath '..\..\';
addpath '..\..\synData';
%%
PSDstrategy = 2;
for trendType = [1,2,3]
load(['..\..\..\data\synData\synDataset_PSDstrategy',num2str(PSDstrategy),'_Trend',num2str(trendType),'.mat']);

%% 
%%
h5File = ['..\..\trainedModel\synData\summaries\SynData_PSD-',num2str(trendType),'\clusters.h5'];

idx = h5read(h5File,'/epoch_00699/hot');
idx = reshape(idx,[],numel(sampleTime));
clusIdx = mean(idx,1);

%%
lossFile1 = ['..\..\trainedModel\synData\summaries\SynData_PSD-',num2str(trendType),'\loss_clustering.txt'];
lossFile2 = ['..\..\trainedModel\synData\summaries\SynData_PSD-',num2str(trendType),'\loss_reconstruction.txt'];
load(lossFile1);
load(lossFile2);

loss = loss_clustering + loss_reconstruction;

%%

predLabel = idx' + 1;

if mean(synLabel == predLabel) < mean(synLabel == (3-predLabel))
    predLabel = 3 - predLabel;
end

%%
plotFlag = true;
if plotFlag
    %%
colorRGB1 = [  0, 43,128]/255;
colorRGB2 = [255,165,  0]/255;
colorRGB3 = [  0, 87, 55]/255;

colorRGB1_dark = [  0,  0,  0]/255;
colorRGB2_dark = [  0,  0,  0]/255;
colorRGB3_dark = [  0,  0,  0]/255;

colorRGBErrArea = 0.7.*[1 1 1];
alphaErrArea = 0.4;

colorRGBSSC = [  0,  0,  0]/255;

panel_Wid = 18;
subFig_Wid = 13.6;
subFig_Hig = subFig_Wid *0.6;
panel_gapWid = 2.2;
panel_gapHig = 1.7;
panelConn_gapHig = 0.5;
inter_Hid = 0.1;
TickLength = [0.01,0.02];

panel_Hig1 = panel_gapHig+subFig_Hig+0*inter_Hid + 0.7;

%%
% tInt = 1200;
% ratioPred = avgPercentageCalculation(res_pri,double(predLabel),tInt,'overlap',1000);
% ratioPesudo = avgPercentageCalculation(res_pri,double(synLabel),tInt,'overlap',1000);

tInt = 200;
overlapInt = 50;
stTime = 0:tInt - overlapInt:sampleTime(end);
edTime = stTime + tInt;
ratioPred = sum(double(predLabel).*(sampleTime >= stTime & sampleTime <edTime))./sum(sampleTime >= stTime & sampleTime <edTime) - 1;
ratioPesudo = sum(double(synLabel).*(sampleTime >= stTime & sampleTime <edTime))./sum(sampleTime >= stTime & sampleTime <edTime) - 1;


fig = figure('Units','centimeters','Position',[2,2,panel_Wid,panel_Hig1]);
axbg = axes(fig,'Units','centimeters','Position',[panel_gapWid panel_gapHig subFig_Wid subFig_Hig],...
    'Color', 'none','Box','off',...
    'XAxisLocation','top','YAxisLocation','right',...
    'LineWidth',2,'TickLength', TickLength,...
    'XTick',[],'YTick',[]);
ax1 = axes(fig,'Units','centimeters','Position',axbg.Position,...
    'Color', 'none','Box','off',...
    'TickDir','out',...
    'Layer','top',...
    'LineWidth',axbg.LineWidth,'TickLength', axbg.TickLength,...
    'FontName','Arial','FontSize',12,'FontWeight','bold');
xlabel('Time (s)','FontName','Arial','FontSize',14,'FontWeight','bold');
ylabel({['Percentage of'],['crack inducing signals (%)']},'FontName','Arial','FontSize',14,'FontWeight','bold');
ax1.YLim = [0,100];
hold on;
p0 = plot(synTime,exCrackPercen.*100,'-','LineWidth',2,'color',[1 0 0]);
pPesudo = plot(stTime(2:end-1),ratioPesudo(2:end-1).*100,'o-','LineWidth',2,'color',colorRGBErrArea,'MarkerFaceColor',[1,1,1]);
pPred = plot(stTime(2:end-1),ratioPred(2:end-1).*100,'o-','LineWidth',2,'color',colorRGB2,'MarkerFaceColor',[1,1,1]);

hl = legend([p0,pPesudo,pPred],...
    {'Ideal percentage','True percentage','The proposed method'},...
    'FontName','Arial','FontSize',12,'FontWeight','bold',...
    'Location','northwest','box','off');

Acc = mean(synLabel == double(predLabel));
figure;
plotconfusion(double(synLabel'==[1:2]'),double(double(predLabel)'==[1:2]'));

Precision = sum(double(predLabel)==2 & synLabel==2) ./ sum(double(predLabel)==2);
Recall = sum(double(predLabel)==2 & synLabel==2) ./ sum(synLabel==2);
F1Score = 2.*Precision.*Recall./(Precision+Recall)

end
exportPath = '..\..\trainedModel\synData';
save(fullfile(exportPath,['res_synDataset_ScaNet_Trend',num2str(trendType),'.mat']),...
    'sampleTime','synLabel',...
    'predLabel',...
    'PSDstrategy');


end
