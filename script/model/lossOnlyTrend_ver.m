classdef lossOnlyTrend_ver < nnet.layer.ClassificationLayer
    properties
        timeMat_p1
        timeMat_p2
        preIndex
        timeMat
        timeInt
        
        Ver
        alpha = 1;
        gamma = 0;
        w = 0.5;
    end
    
    
    methods
        function layer = lossOnlyTrend_ver(name,timeMat1,timeMat2,preIndex,timeMat,gamma,w,ver,timeInt)
    
            % Set layer name.
            layer.Name = name;

            % Set layer description.
            layer.Description = 'time transfer loss';
            
            % Set inputs.
            timeMat1 = reshape(timeMat1,1,[]);
            layer.timeMat_p1 = timeMat1;
            timeMat2 = reshape(timeMat2,1,[]);
            layer.timeMat_p2 = timeMat2;     
            layer.preIndex = preIndex;
            layer.timeMat = timeMat;
            
            layer.gamma = gamma;
            layer.w = w;
            layer.Ver = ver;
            
            if nargin < 9
                layer.timeInt = [];
            else
                layer.timeInt = timeInt;
            end
        end
        
        function loss = forwardLoss(layer, Y, T)

            intNum = max(poissrnd(12),5);

            % Define the timeTransMat
            timeTransMat_l = getTimeTransMat(layer.timeMat,intNum,[],layer.timeInt);
            l = getAggregateTrendLoss(timeTransMat_l,Y,'w',layer.w);  
        end
           
    end
end