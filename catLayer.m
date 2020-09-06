%{
    catLayer.m
    Written by: John Brennan Peace
%}
classdef catLayer < nnet.layer.Layer

    properties
        % (Optional) Layer properties.
        % Layer properties go here.
    end

    methods
        function layer = catLayer(name)
            % This function must have the same name as the class.
            layer.Name = name;
            layer.NumInputs = 3;
            layer.InputNames = {'object','classes','affine'};
            layer.Description = 'Concatenate Localization/Classification/Transformation';
            % Layer constructor function goes here.
        end
        
        function [Z1] = predict(layer, X1, X2, X3)
            % Forward input data through the layer at prediction time and
            % output the result.
            %
            % Inputs:
            %         layer       - Layer to forward propagate through
            %         X1, ..., Xn - Input data
            % Outputs:
            %         Z1, ..., Zm - Outputs of layer forward function
            
            % Layer forward function for prediction goes here.
            Z1 = cat(3, X1, X2, X3);
            

        end


        function [dLdX1, dLdX2, dLdX3] =  backward(layer, X1, X2, X3, Z1, dLdZ1, memory)
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
            dLdX1 = dLdZ1(:,:,1,:);
            dLdX2 = dLdZ1(:,:,2:31,:);
            dLdX3 = dLdZ1(:,:,32:end,:);
        end
    end
end