clear
warning('off');
%% set env
addpath '..\..\..\common';
addpath '..\..\..\data';

%%
load('testing_data.mat')

CNTS = [res_pri.Counts];
cntsFilterFlag = CNTS>10 & [res_pri.Time]<6200;
res_pri = res_pri(cntsFilterFlag);
res_tra = res_tra(cntsFilterFlag);
vecT = vecT(cntsFilterFlag);
vecTR = vecTR(cntsFilterFlag);

%%
freq = [0:2e4:1e6];
PSD_Dataset = wave2psd(vecTR,fs,freq);
PSD_Dataset = log10(PSD_Dataset);

%% Padding & Cut

batchShape = 51;
wfTime = arrayfun(@(x)single(x.Time+ [1:batchShape]./fs),res_pri,'UniformOutput',false);

wfData = PSD_Dataset;
wfTime = cell2mat(wfTime);

wfData = reshape(wfData,[],1,batchShape);
wfTime = reshape(wfTime,[],1,batchShape);

% wfData = wfData + randn(size(wfData)).*1e-2;

save('.\testingData_scaNet.mat','wfData','wfTime','-v7.3');

%%

h5File = '..\..\trainedModel\summaries\PSSres_clus2_application\clusters.h5';
idx = h5read(h5File,'/epoch_00200/hot');
idx = reshape(idx,[],numel(res_pri));
%%
predLabel = idx' + 1;
res_pri_dislocation = res_pri(double(predLabel)==1);
res_pri_crack = res_pri(double(predLabel)==2);
[superJerkIDvec_crack,expMLEstimator_crack,enyExpFlag_crack,err_crack] = autoSuperjerkEstimtor(res_pri_crack,false,false);
enyExpFlag_crack([4,5,7]) = false;
%% Save result
exportPath = '..\..\trainedModel';
mkdir(exportPath)
save(fullfile(exportPath,'res_scaNet_testPSS.mat'),'predLabel','h5File');

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
subFig_Hig = subFig_Wid *0.5;
panel_gapWid = 2.2;
panel_gapHig = 1.5;
panelConn_gapHig = 0.5;
inter_Hid = 0.1;
TickLength = [0.01,0.02];

panel_Hig1 = panel_gapHig+3*subFig_Hig+2*inter_Hid + 0.5;

%%
fig = figure('Units','centimeters','Position',[2,2,panel_Wid,panel_Hig1]);
axbg = axes(fig,'Units','centimeters',...
    'Position',[panel_gapWid panel_gapHig+2*subFig_Hig+2*inter_Hid subFig_Wid subFig_Hig],...
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
ylabel(['Stress (MPa)'],'FontName','Arial','FontSize',14,'FontWeight','bold');
ax1.XLim = [0,6.2e3];  ax1.XAxis.TickValues = [];  ax1.XTickLabel = [];
ax1.XMinorTick = 'off';
ax1.YLim = [-2,66];  ax1.YAxis.TickValues = [0:10:90];
ax1.YMinorTick = 'on';  ax1.YAxis.MinorTickValues = [0:5:95];
ax1.YLabel.Units = ax1.Units;  ax1.YLabel.Position = [-1,ax1.Position(4)/2,0];
hold on;
ss = plot(ax1,ssc_Time(ssc_Time<=6060),ssc_Stress(ssc_Time<=6060),'-',...
    'LineWidth',3,'Color',colorRGBSSC);


axbg = axes(fig,'Units','centimeters',...
    'Position',[panel_gapWid panel_gapHig+1*subFig_Hig+1*inter_Hid subFig_Wid subFig_Hig],...
    'Color', 'none','Box','off',...
    'XAxisLocation','top','YAxisLocation','right',...
    'LineWidth',2,'TickLength', TickLength,...
    'XTick',[],'YTick',[]);
ax2 = axes(fig,'Units','centimeters','Position',axbg.Position,...
    'Color', 'none','Box','off',...
    'TickDir','out',...
    'Layer','top',...
    'LineWidth',axbg.LineWidth,'TickLength', axbg.TickLength,...
    'FontName','Arial','FontSize',12,'FontWeight','bold');
ylabel({'Energy (aJ)'},'FontName','Arial','FontSize',14,'FontWeight','bold');
ax2.XLim = [0,6.2e3];  ax2.XTickLabel = [];  ax2.XAxis.TickValues = [];
ax2.XMinorTick = 'off';
ax2.YScale = 'log';
ax2.YAxis.Limits = [2e-1,4e5];  ax2.YAxis.TickValues = 10.^[-2:1:15];
ax2.YAxis.MinorTick = 'on';  % ax2.YAxis.MinorTickValues = 10.^[-2:15];
ax2.YAxis.MinorTickValues = reshape([2;4;6;8]*10.^[-3:12],1,[]);
ax2.YLabel.Units = ax2.Units;  ax2.YLabel.Position = [-1.1,ax2.Position(4)/2,0];
hold on;
s2 = stem(ax2,[res_pri_crack.Time],[res_pri_crack.Eny],'-','Marker','none','Color',colorRGB2,'LineWidth',3,'MarkerFaceColor',colorRGB2);
semilogy(ax2,[res_pri_crack([1,superJerkIDvec_crack]).Time],[res_pri_crack([1,superJerkIDvec_crack]).Eny],'--','color',[0 0 0],'LineWidth',1.2);
stem(ax2,[res_pri_crack([1,superJerkIDvec_crack]).Time],[res_pri_crack([1,superJerkIDvec_crack]).Eny],'--','color',[0 0 0],'LineWidth',1.2,'Marker','none');

ht = text(300,ax2.YAxis.Limits(2)/1.2,['Crack signals'],...
    'FontSize',12,'FontWeight','bold','FontName','Arial','FontAngle','normal',...
    'HorizontalAlignment','left','VerticalAlignment','top');


axbg = axes(fig,'Units','centimeters',...
    'Position',[panel_gapWid panel_gapHig subFig_Wid subFig_Hig],...
    'Color', 'none','Box','off',...
    'XAxisLocation','top','YAxisLocation','right',...
    'LineWidth',2,'TickLength', TickLength,...
    'XTick',[],'YTick',[]);
ax3 = axes(fig,'Units','centimeters','Position',axbg.Position,...
    'Color', 'none','Box','off',...
    'TickDir','out',...
    'Layer','top',...
    'LineWidth',axbg.LineWidth,'TickLength', axbg.TickLength,...
    'FontName','Arial','FontSize',12,'FontWeight','bold');
xlabel('Time (s)','FontName','Arial','FontSize',14,'FontWeight','bold');
ylabel({['Energy Exponent, ',char(949)]},'FontName','Arial','FontSize',14,'FontWeight','bold');

ax3.XLim = [0,6.2e3];  ax3.XAxis.TickValues = [0:2000:6000];
ax3.XMinorTick = 'on';  ax3.XAxis.MinorTickValues = [0:500:8000];
ax3.XLabel.Units = ax3.Units;  ax3.XLabel.Position = [ax3.Position(3)/2,-0.7,0];
ax3.YAxis.Limits = [1.0,2.9];  ax3.YAxis.TickValues = [1:0.4:5];
ax3.YAxis.TickLabels = num2str(ax3.YTick','%.1f');
ax3.YAxis.MinorTick = 'on';  ax3.YAxis.MinorTickValues = [1:0.1:5];
ax3.YLabel.Units = ax3.Units;  ax3.YLabel.Position = [-1,ax3.Position(4)/2,0];
hold on;

errorbar(ax3,[res_pri_crack([1,superJerkIDvec_crack(enyExpFlag_crack)]).Time],[nan,expMLEstimator_crack(enyExpFlag_crack)],...
    [nan,err_crack(enyExpFlag_crack)],...
    '--','color',colorRGB2_dark,'LineWidth',2,'MarkerSize',10,'MarkerFaceColor',colorRGB2,'MarkerEdgeColor',colorRGB2_dark);
plot(ax3,[res_pri_crack([1,superJerkIDvec_crack(enyExpFlag_crack)]).Time],[nan,expMLEstimator_crack(enyExpFlag_crack)],...
    '^','color',colorRGB2_dark,'LineWidth',1,'MarkerSize',10,'MarkerFaceColor',colorRGB2,'MarkerEdgeColor',colorRGB2_dark);

plot(ax3,ax3.XLim.*[1,1],4/3.*[1,1],'k--','LineWidth',1);
ht = text(300,ax3.YAxis.Limits(2)-0.02,['Power-law exponents estimated for crack singlas'],...
    'FontSize',12,'FontWeight','bold','FontName','Arial','FontAngle','normal',...
    'HorizontalAlignment','left','VerticalAlignment','top');

htThr = text(300,1.34,[char(949),'=1.33'],...
    'FontSize',12,'FontWeight','bold','FontName','Arial','FontAngle','italic',...
    'HorizontalAlignment','left','VerticalAlignment','bottom');
