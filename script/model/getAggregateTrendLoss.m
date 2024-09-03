function loss = getAggregateTrendLoss(timeTransMat,input,varargin)
delta = 1e-8;
reptNum = 20;
w2 = 0.5;
for i = 1:length(varargin)/2
    switch lower(varargin{i*2-1})
        case 'w'
        w2 = varargin{i*2};
        case 'delta'
        delta = varargin{i*2};
        case 'reptnum'
        reptNum = varargin{i*2};
    end
end
w1 = 1 - w2;

% Symmetric form
C_p1 = -timeTransMat.Mat /2 * (log(max(1 - input(2,:)',delta)) + log(max(input(1,:)',delta)));
C_p2 = -timeTransMat.Mat /2 * (log(max(1 - input(1,:)',delta)) + log(max(input(2,:)',delta)));

% Acceleration by matrix calculation
% Default region 
N = timeTransMat.N;
Num = timeTransMat.Num;
timelen = timeTransMat.Timelen;

% Sampling
interruptNum = randsample([2:2:N],reptNum,true) - 1;

L = 0;
sumLen = 0;
for i = 1:reptNum
    interruptMat = zeros(interruptNum(i)+1,N);
    interruptIndex = [0,sort(randsample([1:N-1],interruptNum(i))),N];
    % Merge region
    %   for example. r1+r2 -> r1 
    %   make sure  si<ei<sj<ej in auxiliary function and  the probabilities
    %   of sampling are similar for early and later regions.
    for j = 1:interruptNum(i)+1
        interruptMat(j,interruptIndex(j)+1:interruptIndex(j+1)) = 1;
    end
    R1 = interruptMat * C_p1 ./ (interruptMat * Num);
    R2 = interruptMat * C_p2 ./ (interruptMat * Num);
    % Time intervals
    Rlen = interruptMat * timelen;

    RIndex = [1:numel(R1)];
    for j = 1:(interruptNum(i)+1)/2
        R1Index(j) = RIndex(1);
        RIndex(1) = [];
        tmpIndex = randsample(numel(RIndex),1);
        R2Index(j) = RIndex(tmpIndex);
        RIndex(tmpIndex) = [];
        % Trend Tcj-Tci <=> Tcj+Tdi
        tmpLen = Rlen(R1Index(j)) + Rlen(R2Index(j));
         L = L + (w1.*R1(R1Index(j)) +...
             w2.*R2(R2Index(j))).*tmpLen;
        sumLen = sumLen + tmpLen;
    end
end
% Normalize
loss = L ./ sumLen; 
        
end