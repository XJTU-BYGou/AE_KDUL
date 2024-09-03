classdef focalLossClassificationLayer < nnet.layer.ClassificationLayer
    % Example custom classification layer with sum of squares error loss.
    properties
        % (Optional) Layer properties.
        
        alpha = [0.5;0.5];
        gamma = 0;
        w = 0.5;
        epsilon = 1e-8;
        % Layer properties go here.
    end
    
    
    methods
        function layer = focalLossClassificationLayer(name,w,gamma)
            % layer = sseClassificationLayer(name) creates a sum of squares
            % error classification layer and specifies the layer name.
    
            % Set layer name.
            layer.Name = name;

            % Set layer description.
            layer.Description = 'time transfer loss';
            
            % Set inputs.
%             layer.alpha = alpha;     
            layer.w = w;
            layer.gamma = gamma;
            
            
        end
        
        function loss = forwardLoss(layer, Y, T)
            % loss = forwardLoss(layer, Y, T) returns the SSE loss between
            % the predictions Y and the training targets T.

            alpha = [layer.w;1-layer.w];
            loss = -sum(alpha.*(1-Y + layer.epsilon).^layer.gamma.*T.*log(Y + layer.epsilon),'all');

        end
    end
end