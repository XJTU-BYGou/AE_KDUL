%%
clear
addpath '.\synData';
addpath '.\trainedModel';
addpath '..\common';
synDataGenerator;
%%
clear
%%
PSDstrategy = 2;
dateStr = '';

for trendType = [1,2,3]
load(['..\data\synData\synDataset_PSDstrategy',num2str(PSDstrategy),'_Trend',num2str(trendType),'.mat']);
%%

modelIndex = 1;
% 1 The proposed NN
cName{modelIndex} = 'NN';
load(['.\trainedModel\synData\res_synDataset_proposedNN_Trend',num2str(trendType),'.mat'],...
    'net','predLabel','predScore','Info');
predLabelCell{modelIndex} = double(predLabel);

% Peer machine learning method
modelIndex = 2;
% 6 unsupervised Cluster
cName{modelIndex} = 'Clustering';
load(['.\trainedModel\synData\res_synDataset_unspervised_Trend',num2str(trendType),'.mat']);
predLabelCell{modelIndex} = predLabelKmeans;

modelIndex = 3;
% 7 unsupervised ScaNet
cName{modelIndex} = 'ScaNet';
load(['.\trainedModel\synData\res_synDataset_ScaNet_Trend',num2str(trendType),'.mat'], 'predLabel');
predLabelCell{modelIndex} = double(predLabel);

modelIndex = 4;
% 9 supervised NN with focal loss
cName{modelIndex} = 'IDNN';
load(['.\trainedModel\synData\res_synDataset_supervised_Trend',num2str(trendType),'.mat'], 'predLabel');
predLabelCell{modelIndex} = double(predLabel);


%%
tInt = 500;
overlapInt = 400;
ratioCell = cellfun(@(x)avgPercentageCalculation(sampleTime,double(x),tInt,'overlap',overlapInt),...
    predLabelCell,'UniformOutput',false);
ratioReal = avgPercentageCalculation(sampleTime,synLabel,tInt,'overlap',overlapInt);
accMat = cellfun(@(x)mean(synLabel==reshape(double(x),[],1)),predLabelCell);

switch trendType
    case 1
Precision = cellfun(@(x)sum(reshape(double(x),[],1)==2 & synLabel==2) ./ sum(double(x)==2),predLabelCell);
Recall = cellfun(@(x)sum(reshape(double(x),[],1)==2 & synLabel==2) ./ sum(synLabel==2),predLabelCell);
F1Score = 2.*Precision.*Recall./(Precision+Recall);
    case 3
Precision = cellfun(@(x)sum(reshape(double(x),[],1)==1 & synLabel==1) ./ sum(double(x)==1),predLabelCell);
Recall = cellfun(@(x)sum(reshape(double(x),[],1)==1 & synLabel==1) ./ sum(synLabel==1),predLabelCell);
F1Score = 2.*Precision.*Recall./(Precision+Recall);
    otherwise
Precision = cellfun(@(x)sum(reshape(double(x),[],1)==2 & synLabel==2) ./ sum(double(x)==2),predLabelCell);
Recall = cellfun(@(x)sum(reshape(double(x),[],1)==2 & synLabel==2) ./ sum(synLabel==2),predLabelCell);
F1Score = 2.*Precision.*Recall./(Precision+Recall);
end


RMSE_Ratio = cellfun(@(x)sqrt(mean((ratioReal.Percentage - x.Percentage).^2.*10000)),ratioCell);

cName{1} = 'The proposed method';
cName{2} = 'Kmeans';
cName{3} = 'ScaNet';
cName{4} = 'IDNN';

%%

colorRGB1 = [  0, 43,128]/255;
colorRGB2 = [255,165,  0]/255;
colorRGB3 = [  0, 87, 55]/255;

colorRGB1_dark = [  0,  0,  0]/255;
colorRGB2_dark = [  0,  0,  0]/255;
colorRGB3_dark = [  0,  0,  0]/255;

colorRGBErrArea = 0.7.*[1 1 1];
alphaErrArea = 0.3;

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

colorRGBOrder(1,:) = [216,118, 90]/255;
colorRGBOrder(2,:) = [124,175,226]/255;
colorRGBOrder(3,:) = [179,150,189]/255;
colorRGBOrder(4,:) = [227,196,120]/255;

lineStyleOrder(1:4) = {'-'};

plotIndex = [2,3,4,1];

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
ylabel({['Ratio of Type II signals (%)']},'FontName','Arial','FontSize',14,'FontWeight','bold');
xlabel('Time','FontName','Arial','FontSize',14,'FontWeight','bold');
ax1.XLim = [-200,1.37e4];
ax1.XTick = [0:4e3:2e4];
ax1.XAxis.MinorTick = 'on';  ax1.XAxis.MinorTickValues = [0:0.05:1];
ax1.YLim = [-3,103];
ax1.YTick = [0:20:200]; 
ax1.YAxis.MinorTick = 'on';  ax1.YAxis.MinorTickValues = [0:5:200];
ax1.YLabel.Units = ax1.Units;  ax1.YLabel.Position = [-1,ax1.Position(4)/2,0];
hold on;
clear er pl
for i = 1:numel(plotIndex)
er(i) = patch([ratioCell{plotIndex(i)}.CenTime,flip(ratioCell{plotIndex(i)}.CenTime)],...
    [ratioCell{plotIndex(i)}.ErrorUpp,flip(ratioCell{plotIndex(i)}.ErrorLow)]*100,...
    colorRGBOrder(plotIndex(i),:),'FaceAlpha',alphaErrArea,'EdgeColor','none');
end
% p0 = plot(synTime,exCrackPercen.*100,'-','LineWidth',3,'color',[1 0 0]);
plabel = plot(ax1,ratioReal.CenTime,ratioReal.Percentage*100,'-',...
    'LineWidth',2,'color',[0 0 0],'MarkerFaceColor',[1,1,1]);
for i = 1:numel(plotIndex)
pl(i) = plot(ax1,ratioCell{plotIndex(i)}.CenTime,ratioCell{plotIndex(i)}.Percentage*100,'-',...
    'LineStyle',lineStyleOrder{plotIndex(i)},'LineWidth',2,...
    'Color',colorRGBOrder(plotIndex(i),:),'MarkerFaceColor',[1,1,1]);
end

if trendType == 3
hl = legend([pl,plabel],[cName(plotIndex),{'Actual percentage'}],...
    'FontName','Arial','FontSize',12,'FontWeight','bold',...
    'Location','southeast','box','off',...
    'NumColumns',1,'Orientation','horizontal');
else
hl = legend([pl,plabel],[cName(plotIndex),{'Actual percentage'}],...
    'FontName','Arial','FontSize',12,'FontWeight','bold',...
    'Location','northwest','box','off',...
    'NumColumns',1,'Orientation','horizontal');
end


%%
plotIndex = [2,3,4,1];

rebF1 = F1Score(plotIndex);
rebAcc = accMat(plotIndex);
rebCName = cName(plotIndex);
rebRMSE = RMSE_Ratio(plotIndex);

randPred = randsample(2,numel(synLabel),true);
accBaseMat = 0.7; % mean(synLabel==randPred);

switch trendType
    case 1
Precision = sum(randPred==2 & synLabel==2) ./ sum(randPred==2);
Recall = sum(randPred==2 & synLabel==2) ./ sum(synLabel==2);
F1BaseScore = 2.*Precision.*Recall./(Precision+Recall);
    case 3
Precision = sum(randPred==1 & synLabel==1) ./ sum(double(randPred)==1);
Recall = sum(randPred==1 & synLabel==1) ./ sum(synLabel==1);
F1BaseScore = 2.*Precision.*Recall./(Precision+Recall);
    otherwise
Precision = sum(randPred==2 & synLabel==2) ./ sum(double(randPred)==2);
Recall = sum(randPred==2 & synLabel==2) ./ sum(synLabel==2);
F1BaseScore = 2.*Precision.*Recall./(Precision+Recall);
end


colorRGB1 = [114,174,254]/255;
colorRGB2 = [180,218,168]/255;

colorRGB1_dark = [  0,  0,  0]/255;
colorRGB2_dark = [255, 64, 64]/255;
colorRGB3_dark = [  0,  0,  0]/255;

colorRGBErrArea = 0.7.*[1 1 1];
alphaErrArea = 1;

colorRGBSSC = [  0,  0,  0]/255;

panel_Wid = 18;
subFig_Wid = 13.6;
subFig_Hig = subFig_Wid *0.6;
panel_gapWid = 2.3;
panel_gapHig = 2.8;
panelConn_gapHig = 0.5;
inter_Hid = 0.1;
TickLength = [0.01,0.02];

panel_Hig2 = panel_gapHig+subFig_Hig+0*inter_Hid + 0.7;

fig = figure('Units','centimeters','Position',[4,4,panel_Wid,panel_Hig2]);

ax1 = axes(fig,'Units','centimeters','Position',[panel_gapWid panel_gapHig subFig_Wid subFig_Hig],...
    'Color', 'none','Box','off',...
    'TickDir','out',...
    'Layer','top',...
    'LineWidth',2,'TickLength', TickLength,...
    'FontName','Arial','FontSize',12,'FontWeight','bold');
ylabel({'RMSE'}...
    ,'FontName','Arial','FontSize',14,'FontWeight','bold');
ax1.XLim = [1,numel(plotIndex)*2+2]-0.5; %[0,length(rebInteresValue)+1]*2;
ax1.XTick = [1:numel(plotIndex)]*2;
ax1.XTickLabel = [];
ax1.XAxis.MinorTick = 'off';
ax1.XTickLabelRotation = 45;
% ax1.YLim = [-0.00,0.72];
% ax1.YTick = [-0.5:0.1:1.5];
ax1.YAxis.MinorTick = 'on';  %ax1.YAxis.TickLabels = num2str(ax1.YTick','%.1f');
ax1.YLabel.Units = ax1.Units;  ax1.YLabel.Position = [-1,ax1.Position(4)/2,0];
hold on;

ax12 = axes(fig,'Units','centimeters','Position',ax1.Position,...
    'Color', 'none','Box','off',...
    'TickDir','out',...
    'Layer','top',...
    'XAxisLocation','top','YAxisLocation','right',...
    'LineWidth',ax1.LineWidth,'TickLength', ax1.TickLength,...
    'FontName','Arial','FontSize',12,'FontWeight','bold',...
    'XTick',[],'YTick',[]);
ax12.YAxis.Color = [0 0 0];
hold on;

axXTick = axes(fig,'Units','centimeters','Position',ax1.Position,...
    'Color', 'none','Box','off',...
    'TickDir','out',...
    'Layer','top',...
    'LineWidth',ax1.LineWidth,'TickLength', ax1.TickLength,...
    'FontName','Arial','FontSize',12,'FontWeight','bold',...
    'XTick',[],'YTick',[]);
hold on;
axXTick.XLim = ax1.XLim;
axXTick.XTick = [1:numel(plotIndex)]*2;
axXTick.XTickLabel = rebCName;
axXTick.XTickLabelRotation = 0;

hold on;
rmseBar = bar(ax1,[1:numel(plotIndex)]*2,rebRMSE,...
    'BarWidth',0.35,'FaceColor',colorRGB1,'EdgeColor','none','FaceAlpha',alphaErrArea);

lgd = legend(axXTick,[rmseBar],...
    {'RMSE'},...
    'FontName','Arial','FontSize',12,'FontWeight','bold',...
    'location','northwest','box','off');

%%
tmp = struct('cName',cName,'RMSE',RMSE_Ratio,'ratioCell',ratioCell,'ratioReal',ratioReal);
resData{trendType} = tmp;
end

%%
exportPath = '.\export';
mkdir(exportPath);
save(fullfile(exportPath,'data_FigS4.mat'),...
    'resData');