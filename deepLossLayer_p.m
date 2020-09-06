%{
    deepLossLayer_p.m
    Written by: John Brennan Peace
%}
classdef deepLossLayer_p < nnet.layer.RegressionLayer
    % Example custom regression layer with mean-absolute-error loss.
    properties
       m_len
    end
    methods
        function layer = deepLossLayer_p(name)
            % layer = maeRegressionLayer(name) creates a
            % mean-absolute-error regression layer and specifies the layer
            % name.
            
            % Set layer name.
            layer.Name = name;
            layer.m_len = 63.5;
%             layer.corners = [-layer.m_len,-layer.m_len,1; ...
%                 -layer.m_len,layer.m_len,1; ...
%                 layer.m_len,-layer.m_len,1; ...
%                 layer.m_len,layer.m_len,1];
            % Set layer description.
            layer.Description = 'Deep Losses';
        end
        
        function [loss] = forwardLoss(layer, Y, T)
            % loss = forwardLoss(layer, Y, T) returns the MAE loss between
            % the predictions Y and the training targets T.
            % Calculate MAE.
            corners = [-63.5,-63.5,1; ...
            -63.5,63.5,1; ...
            63.5,-63.5,1; ...
            63.5,63.5,1];
            corners = single(corners);
            N = size(T,4);
            nonan = 1e-9;
            Yo = Y(:,:,1,:);
            Yc = Y(:,:,2:31,:);
            Ya = Y(:,:,32:end,:);
            To = T(:,:,1,:);
            Tc = T(:,:,2:31,:);
            Ta = T(:,:,32:end,:);
            ind = (To>0.1);
            num_el = sum(ind,'all');
            loss_ce = -sum(sum(sum(sum(Tc.*log(max(Yc,nonan))))))/(30*num_el);
            loss_obj = sum(sum(sum((To-Yo).^2)))/sum(ind,'all');
            loss_aff = 0;
%             for i = 1:N
%                 Ti = To(:,:,:,i);
%                 [~,maxind] = max(Ti(:)); 
%                 col = ceil(maxind/layer.grid);
%                 row = maxind - (col-1)*layer.grid;
%                 
%                 out_aff = Ya(row,col,:,i);
%                 out_H = reshape(out_aff,[2,2]);
%                 out_H = horzcat(out_H,[0;0]);
%                 out_H = vertcat(out_H,[0 0 1]);
%                 out_corner = layer.corners*out_H;
%                 out_corner(:,3) = [];
%                 t_aff = Ta(row,col,:,i);
%                 t_H = reshape(t_aff,[2,2]);
%                 t_H = horzcat(t_H,[0;0]);
%                 t_H = vertcat(t_H,[0 0 1]);
%                 t_corner = layer.corners*t_H;
%                 t_corner(:,3) = [];
%                 scale = std(t_corner);
%                 out_corner = out_corner./scale;
%                 t_corner = t_corner./scale;
%                 loss_aff = loss_aff + ((out_corner-t_corner).^2);
%                 
%             end
            nel = 0;
            grid_size = size(To,1);
            for i = 1:N
                Ti = To(:,:,:,i);
                [maxcent,maxind] = maxk(Ti(:),5);
                col = ceil(maxind/grid_size);
                row = maxind - (col-1)*grid_size;
%                 [row,col] = find(Ti > 0.1); 
                t_aff = Ta(row(1),col(1),:,i);
                t_aff = reshape(t_aff,[2 3]);
                t_H = vertcat(t_aff,[0 0 1]);
                t_corner = corners*t_H;
                t_corner = t_corner./(t_corner(:,3));
                t_corner(:,3) = [];
                scale = std(t_corner);
                t_corner = t_corner./scale;
                
                for j = 1:sum(maxcent>0)
                    out_aff = Ya(row(j),col(j),:,i);
                    out_aff = reshape(out_aff,[2 3]);
                    out_H = vertcat(out_aff,[0 0 1]);
                    out_corner = corners*out_H;
                    out_corner = out_corner./(out_corner(:,3));
                    out_corner(:,3) = [];
                    out_corner = out_corner./scale;
                
                    diff = abs(out_corner-t_corner);
                    loss_aff = loss_aff + diff;
                    nel = nel + 1;
                end
%                 loss_aff = loss_aff + ((out_corner-t_corner).^2);
                
            end
            loss_aff = sum(loss_aff,'all')/(8*nel);
%             loss_aff = sum(sum(sum(sum(ind.*((Ta-Ya).^2)))))/(4*num_el);

%             rat = loss_ce + loss_obj + loss_aff;
%             ce_rat = (loss_obj + loss_aff)/rat;
%             obj_rat = (loss_ce + loss_aff)/rat;
%             aff_rat = (loss_obj + loss_ce)/rat;
%             loss = loss_ce*ce_rat + loss_obj*obj_rat + loss_aff*aff_rat;
            loss = 100*loss_ce + 50*loss_obj + loss_aff;
            if isnan(extractdata(loss_aff))
                a = 3;
            elseif isnan(extractdata(loss_obj))
                a = 3;
            end
%             loss = -sum(sum(T.*log(max(Y,nonan)) + (1-T).*log(max(1-Y,nonan))))/numel(T);
            
        end
        
%         function dLdY = backwardLoss(layer, Y, T)
%             % (Optional) Backward propagate the derivative of the loss 
%             % function.
%             %
%             % Inputs:
%             %         layer - Output layer
%             %         Y     – Predictions made by network
%             %         T     – Training targets
%             %
%             % Output:
%             %         dLdY  - Derivative of the loss with respect to the 
%             %                 predictions Y        
%             
%             N = size(Y,4);
%             indicator = T(:,:,1,:);
%             t_cat = T(:,:,2:end,:);
%             y_obj = Y(:,:,1,:);
%             y_cat = Y(:,:,2:end,:);
%             
% %             Y_obj = 1/size(Y,4);
% %             y_cat = T(:,:,1,:).*(T(:,:,2:end,:)./Y(:,:,2:end,:));
% %             dLdY = cat(3, y_obj, y_cat);
%             
% %             y_cat = sqrt(indicator.*abs(Y(:,:,2:end,:)-T(:,:,2:end,:)))/N;
% %             dLdCat = indicator.*(y_cat-t_cat);
%             
% %             wasGpu = 0;
% %             if isa(dLdCat,'gpuArray')
% %                 dLdCat = gather(dLdCat);
% %                 wasGpu = 1;
% %             end
% %             if isa(dLdCat,'dlarray')
% %                 a = extractdata(dLdCat);
% %             else 
% %                 a = dLdCat;
% %             end
% %             i = isnan(a);
% %             dLdCat(i) = 0;
% %             if wasGpu
% %                 dLdCat = gpuArray(dLdCat);
% %             end
%             dLdCat = 2*indicator.*(t_cat - y_cat)/N;
%             dLdObj = 2*(indicator-y_obj)/N;
% %             dLdY = cat(3, dLdObj, dLdCat);
% %             test = 0.*Y(:,:,1,:);
%             dLdY = cat(3, dLdObj, dLdCat);
%             
%             
%         end
    end
end