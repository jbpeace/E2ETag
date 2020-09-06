%{
    splitSoftmaxLayer.m
    Written by: John Brennan Peace
%}
classdef splitSoftmaxLayer < nnet.layer.Layer

    properties
        % (Optional) Layer properties.

        % Layer properties go here.

    end

    methods
        function layer = splitSoftmaxLayer(name)
            % (Optional) Create a myLayer.
            % This function must have the same name as the class.
            layer.Name = name;
%             layer.NumInputs = 2;
            layer.NumOutputs = 3;
            layer.OutputNames = {'object','classes','affine'};
            layer.Description = 'Split Layers, Classification goes to Softmax';
            % Layer constructor function goes here.
        end
        
        function [Z1, Z2, Z3] = predict(layer, X1)
            % Forward input data through the layer at prediction time and
            % output the result.
            %
            % Inputs:
            %         layer       - Layer to forward propagate through
            %         X1, ..., Xn - Input data
            % Outputs:
            %         Z1, ..., Zm - Outputs of layer forward function
            
            % Layer forward function for prediction goes here.
            Z1 = X1(:,:,1,:);
            Z2 = X1(:,:,2:31,:);
            Z3 = X1(:,:,32:end,:);
            

        end


        function [dLdX1] =  backward(layer, X1, Z1, Z2, Z3, dLdZ1,dLdZ2,dLdZ3,memory)
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
            dLdX1 = cat(3, dLdZ1, dLdZ2, dLdZ3);
        end
    end
end