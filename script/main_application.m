clear
warning('off');
%% set env
addpath '..\common';
addpath '..\data';
addpath '.\model';
addpath '.\trainedModel';

%% Import data & mdl
load('..\data\testing_data.mat')
load('.\trainedModel\res_NN_2.mat','net','fs','PCAProcessor','norm_mu','norm_scale');

%% 
data_preprocess_testing;
%%
predTestLabel = classify(net,XTest);
predTestScore = predict(net,XTest);

res_pri_dislocation = res_pri(double(predTestLabel)==1);
res_pri_crack = res_pri(double(predTestLabel)==2);
%%
[superJerkIDvec_crack,expMLEstimator_crack,enyExpFlag_crack,err_crack] = superjerkEstimtor_insuit(res_pri_crack,false);

%% Save result
exportPath = '.\export';
mkdir(exportPath)
save(fullfile(exportPath,'data_Fig4.mat'),...
    'res_pri','ssc_Time','ssc_Stress',...
    'res_pri_crack','superJerkIDvec_crack','expMLEstimator_crack','enyExpFlag_crack','err_crack');

%%
plotFlag = true;
if plotFlag
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
ax1 = axes(fig,'Units','centimeters',...
    'Position',[panel_gapWid panel_gapHig+2*subFig_Hig+0*inter_Hid subFig_Wid subFig_Hig],...
    'Color', 'none','Box','off',...
    'TickDir','out',...
    'Layer','top',...
    'LineWidth',2,'TickLength',TickLength,...
    'FontName','Arial','FontSize',12,'FontWeight','bold');
ylabel(['Energy (aJ)'],'FontName','Arial','FontSize',14,'FontWeight','bold');
ax1.XLim = [0,6.2e3];
ax1.XMinorTick = 'off';
ax1.XTick = [];  ax1.XTickLabel = [];
ax1.YScale = 'log';
ax1.YAxis.Limits = [2e-1,4e6];  ax1.YAxis.TickValues = 10.^[-2:1:12];
ax1.YAxis.MinorTick = 'on';     %ax1.YAxis.MinorTickValues = 10.^[-2:1:12];
ax1.YAxis.MinorTickValues = reshape([2;4;6;8]*10.^[-3:12],1,[]);
ax1.YLabel.Units = ax1.Units;    ax1.YLabel.Position = [-1.2,ax1.Position(4)/2,0];
hold on;
clear et
et = stem(ax1,[res_pri.Time],[res_pri.Eny],...
    'Marker','none','Color',colorRGB3,'LineWidth',3,'MarkerFaceColor',colorRGB1);

ax12 = axes('Units','centimeters','Position', ax1.Position, ...
    'XAxisLocation', 'top', 'YAxisLocation', 'right', ...
    'Color', 'none','Box','off',...
    'TickDir','out',...
    'Layer','top',...
    'LineWidth',ax1.LineWidth,'TickLength', ax1.TickLength,...
    'FontName','Arial','FontSize',12,'FontWeight','bold');
ylabel(['Stress (MPa)'],'FontName','Arial','FontSize',14,'FontWeight','bold');
ax12.XLim = ax1.XLim;  ax12.XTick = [];  ax12.XAxis.TickValues = [];
ax12.XMinorTick = 'off';
ax12.XAxis.Color = [0 0 0];
ax12.YAxis.Limits = [-2,66];  ax12.YAxis.TickValues = [0:10:100];
ax12.YAxis.Color = colorRGBSSC;
ax12.YAxis.MinorTick = 'on';  ax12.YAxis.MinorTickValues = [0:5:100];
ax12.YLabel.Units = ax12.Units;  ax12.YLabel.Position = [ax1.Position(1)+ax1.Position(3)-1.3,ax1.Position(4)/2,0];
hold on;
ss = plot(ax12,ssc_Time,ssc_Stress,'-','LineWidth',3,'Color',colorRGBSSC);

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
ax3.YAxis.Limits = [1.0,2.5];  ax3.YAxis.TickValues = [1:0.4:5];
ax3.YAxis.TickLabels = num2str(ax3.YTick','%.1f');
ax3.YAxis.MinorTick = 'on';  ax3.YAxis.MinorTickValues = [1:0.1:5];
ax3.YLabel.Units = ax3.Units;  ax3.YLabel.Position = [-1,ax3.Position(4)/2,0];
hold on;

patch(ax3,'Faces',[1,2,3,4],'Vertices',[[0.*[1,1],[res_pri_crack([superJerkIDvec_crack(7)]).Time].*[1,1]]',[ax3.YLim,ax3.YLim(end:-1:1)]'],'FaceColor',[1 0 1],'FaceAlpha',.03,'EdgeColor','none')
patch(ax3,'Faces',[1,2,3,4],'Vertices',[[[res_pri_crack([superJerkIDvec_crack(7)]).Time].*[1,1],ax3.XLim(2).*[1,1]]',[ax3.YLim,ax3.YLim(end:-1:1)]'],'FaceColor','red','FaceAlpha',.1,'EdgeColor','none')
patch(ax3,'Faces',[1,2,3,4],'Vertices',[[[res_pri_crack([superJerkIDvec_crack(9)]).Time].*[1,1],ax3.XLim(2).*[1,1]]',[ax3.YLim,ax3.YLim(end:-1:1)]'],'FaceColor','red','FaceAlpha',.3,'EdgeColor','none')

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
end
%% scanet res
load('.\trainedModel\res_scaNet_testPSS.mat','predLabel');

res_pri_dislocation = res_pri(double(predLabel)==1);
res_pri_crack = res_pri(double(predLabel)==2);
[superJerkIDvec_crack,expMLEstimator_crack,enyExpFlag_crack,err_crack] = ...
    autoSuperjerkEstimtor(res_pri_crack,false,false);
enyExpFlag_crack([4,5,7]) = false;
%% Save result
exportPath = '.\export';
mkdir(exportPath)
save(fullfile(exportPath,'data_FigS6.mat'),...
    'res_pri','ssc_Time','ssc_Stress',...
    'res_pri_crack','superJerkIDvec_crack','expMLEstimator_crack','enyExpFlag_crack','err_crack');
