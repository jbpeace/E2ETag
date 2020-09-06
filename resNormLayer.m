%{
    resNormLayer.m
    Written by: John Brennan Peace
%}
classdef resNormLayer < nnet.layer.Layer
    properties
        resMean
        resStd
    end
    methods
        function layer = resNormLayer(name)
            % Set layer name.
            layer.Name = name;
            layer.resMean = reshape(single([0.485, 0.456, 0.406]),[1 1 3]);
            layer.resStd = reshape(single([0.229, 0.224, 0.225]),[1 1 3]);
            % Set layer description.
            layer.Description = 'Resnet input normalization';
        end
        
        
        function [Z1] = predict(layer, X1)
            % Forward input data through the layer at prediction time and
            % output the result.
            %
            % Inputs:
            %         layer       - Layer to forward propagate through
            %         X1, ..., Xn - Input data
            % Outputs:
            %         Z1, ..., Zm - Outputs of layer forward function
            
            % Layer forward function for prediction goes here.  
            
            %X1 should be [r,c,3],range(0,1)
            Z1 = (X1-layer.resMean)./layer.resStd;
        end
    end
end