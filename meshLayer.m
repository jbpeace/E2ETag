%{
    meshLayer.m
    Written by: John Brennan Peace
%}
classdef meshLayer < nnet.layer.Layer

    properties
        % (Optional) Layer properties.
        % Layer properties go here.
    end

    methods
        function layer = meshLayer(name)
            % This function must have the same name as the class.
            layer.Name = name;
            layer.Description = 'CoordConv Layer';
            % Layer constructor function goes here.
        end
        
        function [Z1] = predict(layer, X1)
            % Forward input data through the layer at prediction time and
            % output the result.
            %
            % Inputs:
            %         layer       - Layer to forward propagate through
            %         X1, ..., Xn - Input data
            %         Z1, ..., Zm - Outputs of layer forward function
            
            % Layer forward function for prediction goes here.
            M = X1;
            [M1,M2] = meshgrid(1:size(X1,1),1:size(X1,2));
            M(:,:,1,:) = repmat(M1,[1 1 1 size(X1,4)]);
            M(:,:,2,:) = repmat(M2,[1 1 1 size(X1,4)]);
            M = (2*M./size(X1,1))-1;
            M(:,:,3,:) = [];
            Z1 = cat(3, X1, M);
            

        end
            % Outputs:


        function [dLdX1] =  backward(layer, X1, Z1, dLdZ1, memory)
            % (Optional) Backward propagate the derivative of the loss  
            % function through the layer.
            %
            % Inputs:
            %         layer             - Layer to backward propagate through
            %         X1, ..., Xn       - Input data
            %         Z1, ..., Zm       - Outputs of layer forward function            
            %         dLdZ1, ..., dLdZm - Gradients propagated from the next layers
            %         memory            - Memory value from forward function
            % Outputs:
            %         dLdX1, ..., dLdXn - Derivatives of the loss with respect to the
            %                             inputs
            %         dLdW1, ..., dLdWk - Derivatives of the loss with respect to each
            %                             learnable parameter
            dLdX1 = dLdZ1(:,:,1:3,:);
        end
    end
end