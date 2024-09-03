function [psd_dataset] = wave2psd(vecTR,fs,freq,varargin)
%% 
% [psd_dataset] = wave2psd(vecTR,fs,freq,varargin)
% 

%% Parameter Setting
% Default
window = [];
noverlap = [];
normalize = true;

for i = 1:length(varargin)/2
    switch lower(varargin{i*2-1})
        case 'window'
            window = varargin{i*2};
        case 'noverlap'
            noverlap = varargin{i*2};
        case 'normalize'
            normalize = varargin{i*2};
    end
end

if length(window) == 1
window = repmat(window,1,numel(vecTR));
end


psd_dataset = [];
for i = 1:numel(vecTR)

% [psdx,f] = pwelch(vecTR{i},[],[],freq,fs);
% tmp_psd = interp1([f;2e7],[psdx;0],freq);
% tmp_psd = log10(tmp_psd);
if length(window) > 0
[psdx,~] = pwelch(vecTR{i},window(i),noverlap,freq,fs);
else
[psdx,~] = pwelch(vecTR{i},[],[],freq,fs);
end
if normalize
    tmp_psd = psdx/trapz(freq,psdx);
else
    tmp_psd = psdx;
end

psd_dataset = [psd_dataset;tmp_psd];
end

end