function [fig,drawData] = aeMLPlotor(data,label,x0)

if ((nargin > 3) || (nargin < 1))
   error('Too many or too few input arguments.');
elseif(nargin == 1)
    label = [];
    x0 = [];
elseif(nargin == 2)
    x0 = [];
end

errbar_flag = 1;

%BestR.l
% gcolor = {[0,0,1],[1 0 0],[255 165 0]/255,[0,0,0],[1,0,1],[0,1,1],[1,1,0],[166,124,64]/255,[217,217,25]/255};
% gcolor = {[0,43,128]/255,[255 127 0]/255,[0,87,55]/255,[176 48 96]/255,[95 158 160]/255,[0,1,1],[1,1,0],[166,124,64]/255,[217,217,25]/255};
% gshape = {'o','d','s','^','x','d','o','^','s','.','x','d'};

gcolor = {[0,136,204]/255,[255,64,140]/255,[0,43,128]/255,[255 127 0]/255,[0,87,55]/255,[176 48 96]/255,[95 158 160]/255,[0,1,1],[1,1,0],[166,124,64]/255,[217,217,25]/255};
gshape = {'d','o','s','^','x','d','o','^','s','.','x','d'};


if isempty(label)
    label = ones(1,numel(data));
end

%
for i = 1:max(label)
    [~,record] = powerlawExponentMLEstimator(data(label == i),x0);
    epsilon = record.Alpha;
    errbar = record.Err;
    E0 = record.Xmin;
    
    if ~isempty(E0)
    if errbar_flag == 0
    fig{i} = semilogx(E0,epsilon,gshape{i},'Color',gcolor{i},'Linewidth',1.5,'MarkerFaceColor',[1,1,1]); 
    hold on;
    elseif errbar_flag == 1
        if length(epsilon) ~= length(E0)
            E0(length(epsilon)+1:end) = [];
        end
    fig{i} = errorbar(E0,epsilon,errbar,[gshape{i}],'Color',gcolor{i},'Linewidth',1.5,'MarkerFaceColor',[1,1,1]);
    hold on;
    set(gca, 'Xscale',  'log')
    end
    end
    
    drawData.E0{i} = reshape(E0,1,[]);
    drawData.epsilon{i} = reshape(epsilon,1,[]);
    drawData.errbar{i} = reshape(errbar,1,[]);
end
hold off
% ylim([1 2.4])


end

function [epsilon,errbar,Ex,id] = aeMLEstimator(Eny,E0)
% Get the epsilon with the ep.
%   此处显示详细说明
if nargin < 1 || nargin > 2
    error('Too many or too few input arguments.');
elseif nargin == 1
    E0 = [];
end
    
if length(Eny) < 2
    %error('Too few Eny data.')
    epsilon = [];
    id = [];
else


if isempty(E0) 
    [Eny,id] = sort(Eny);
for i = length(Eny)-1:-1:1
    tmp = log(Eny(i:end)/Eny(i));
    epsilon(i) = 1 + 1/mean(tmp);
    errbar(i) = sqrt(var(tmp)./length(tmp))./mean(tmp).^2;
end
id = id(1:end-1);
Ex = Eny(1:end-1);
else
%% Special E0
id = [];
epsilon = [];
errbar = [];
for i = length(E0):-1:1
    if ~isempty(Eny(Eny>=E0(i)))
    tmp = log(Eny(Eny>=E0(i))/E0(i));
    if ~isempty(tmp)
        epsilon = [1 + 1/mean(tmp),epsilon];
        errbar = [sqrt(var(tmp)./length(tmp))./mean(tmp).^2,errbar];
        id = [i,id];
    end
    end
end
Ex = E0(id);
end
end

end

