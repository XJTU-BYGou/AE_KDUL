function [res] = calculateAvgFeature(res_pri,timeInt,varargin)
% [res] = calculateAvgFeature(res_pri,timeInt,varargin)
%
%

% default setting
samplingMod = 'window';
overlapInt = 0;
plotFlag = false;

for i = 1:length(varargin)/2
    switch lower(varargin{i*2-1})
        case 'samplingmod'
            samplingMod = varargin{i*2};
        case 'sampling'
            samplingMod = varargin{i*2};
        case 'overlap'
            overlapInt = varargin{i*2};
        case 'plotflag'
            plotFlag = varargin{i*2};
    end
end



T = [res_pri.Time];
if strcmp(samplingMod,'neighbor')
    res.Time = [res_pri.Time];
    res.TimeSt = arrayfun(@(x)x-timeInt/2,res.Time);
    res.TimeEd = arrayfun(@(x)x+timeInt/2,res.Time);
    
    res.AvgEny = arrayfun(@(x)mean([res_pri(T>=x-timeInt/2&T<x+timeInt/2).Eny]),res.Time);
    res.AvgDur = arrayfun(@(x)mean([res_pri(T>=x-timeInt/2&T<x+timeInt/2).Dur]),res.Time);
    res.AvgRiseTime = arrayfun(@(x)mean([res_pri(T>=x-timeInt/2&T<x+timeInt/2).RiseT]),res.Time);
    res.AvgAmp = arrayfun(@(x)mean([res_pri(T>=x-timeInt/2&T<x+timeInt/2).Amp]),res.Time);
    res.Activity = arrayfun(@(x)sum(T>=x-timeInt/2&T<x+timeInt/2)/timeInt,res.Time);
    
    
    res.AvgRMS = arrayfun(@(x)mean([res_pri(T>=x-timeInt/2&T<x+timeInt/2).RMS]),res.Time);
    res.AvgFreq = arrayfun(@(x)mean([res_pri(T>=x-timeInt/2&T<x+timeInt/2).Counts]...
        ./[res_pri(T>=x-timeInt/2&T<x+timeInt/2).Dur]),res.Time).*1e3;
    
%     res.MLEres = arrayfun(@(x)powerlawExponentMLEstimator([res_pri(T>=x-timeInt/2&T<x+timeInt/2).Eny]),res.Time);
    [res.MLEres,res.MLErecord] = arrayfun(@(x)powerlawExponentMLEstimator([res_pri(T>=x-timeInt/2&T<x+timeInt/2).Eny]),...
        res.Time);
elseif strcmp(samplingMod,'window')
    res.Time = timeInt/2:timeInt-overlapInt:max([res_pri.Time])+timeInt/2;
    res.TimeSt = arrayfun(@(x)x-timeInt/2,res.Time);
    res.TimeEd = arrayfun(@(x)x+timeInt/2,res.Time);
    
    res.AvgEny = arrayfun(@(x)max([mean([res_pri(T>=x-timeInt/2&T<x+timeInt/2).Eny]),single(0)]),res.Time);
    res.AvgDur = arrayfun(@(x)max([mean([res_pri(T>=x-timeInt/2&T<x+timeInt/2).Dur]),single(0)]),res.Time);
    res.AvgRiseTime = arrayfun(@(x)max([mean([res_pri(T>=x-timeInt/2&T<x+timeInt/2).RiseT]),single(0)]),res.Time);
    res.AvgAmp = arrayfun(@(x)max([mean([res_pri(T>=x-timeInt/2&T<x+timeInt/2).Amp]),single(0)]),res.Time);
    res.Activity = arrayfun(@(x)max([sum(T>=x-timeInt/2&T<x+timeInt/2)/timeInt,single(0)]),res.Time);
    
    res.AvgRMS = arrayfun(@(x)max([mean([res_pri(T>=x-timeInt/2&T<x+timeInt/2).RMS]),single(0)]),res.Time);
    res.AvgFreq = arrayfun(@(x)max([mean([res_pri(T>=x-timeInt/2&T<x+timeInt/2).Counts]...
        ./[res_pri(T>=x-timeInt/2&T<x+timeInt/2).Dur]),nan]),res.Time).*1e3;
    
%     res.MLEres = arrayfun(@(x)powerlawExponentMLEstimator([res_pri(T>=x-timeInt/2&T<x+timeInt/2).Eny]),res.Time);
    [res.MLEres,res.MLErecord] = arrayfun(@(x)powerlawExponentMLEstimator([res_pri(T>=x-timeInt/2&T<x+timeInt/2).Eny]),...
        res.Time);
end

if plotFlag
    panel_Wid = 15;
    subFig_Wid = 11.4;
    subFig_Hig = subFig_Wid *0.6;
    panel_gapWid = 2.2;
    panel_gapHig = 1.7;
    inter_Hid = 0.1;
    TickLength = [0.01,0.02];

    panel_Hig1 = panel_gapHig+3.*subFig_Hig+2*inter_Hid + 0.7;
    
    fig = figure('Units','centimeters','Position',[2,2,panel_Wid,panel_Hig1]);
    axbg = axes(fig,'Units','centimeters',...
        'Position',[panel_gapWid panel_gapHig+2.*subFig_Hig+2*inter_Hid subFig_Wid subFig_Hig],...
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
    ylabel({['Energy (aJ)']},'FontName','Arial','FontSize',14,'FontWeight','bold');
    ax1.XTickLabel = [];
    ax1.YScale = 'log';
    hold on;
    stem([res_pri.Time],[res_pri.Eny],'Marker','none','LineWidth',2);
    
    axbg = axes(fig,'Units','centimeters',...
        'Position',[panel_gapWid panel_gapHig+1.*subFig_Hig+1*inter_Hid subFig_Wid subFig_Hig],...
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
    ylabel({['Average Energy (aJ)']},'FontName','Arial','FontSize',14,'FontWeight','bold');
    ax2.XTickLabel = [];
    ax2.YScale = 'log';
    hold on;
    plot(ax2,res.Time,res.AvgEny,'-','LineWidth',2);
    
%     axbg = axes(fig,'Units','centimeters',...
%         'Position',[panel_gapWid panel_gapHig+0.*subFig_Hig+0*inter_Hid subFig_Wid subFig_Hig],...
%         'Color', 'none','Box','off',...
%         'XAxisLocation','top','YAxisLocation','right',...
%         'LineWidth',2,'TickLength', TickLength,...
%         'XTick',[],'YTick',[]);
%     ax3 = axes(fig,'Units','centimeters','Position',axbg.Position,...
%         'Color', 'none','Box','off',...
%         'TickDir','out',...
%         'Layer','top',...
%         'LineWidth',axbg.LineWidth,'TickLength', axbg.TickLength,...
%         'FontName','Arial','FontSize',12,'FontWeight','bold');
%     ylabel({['Activity']},'FontName','Arial','FontSize',14,'FontWeight','bold');
%     xlabel('Time (s)','FontName','Arial','FontSize',14,'FontWeight','bold');
%     ax3.YScale = 'log';
%     hold on;
%     plot(ax3,res.Time,res.Activity,'-','LineWidth',2);
    
    axbg = axes(fig,'Units','centimeters',...
        'Position',[panel_gapWid panel_gapHig+0.*subFig_Hig+0*inter_Hid subFig_Wid subFig_Hig],...
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
    ylabel({['Average Duration']},'FontName','Arial','FontSize',14,'FontWeight','bold');
    xlabel('Time (s)','FontName','Arial','FontSize',14,'FontWeight','bold');
    hold on;
    plot(ax3,res.Time,res.AvgDur,'-','LineWidth',2);
    
    linkaxes([ax1,ax2,ax3],'x');
end
end