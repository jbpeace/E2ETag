%{
    get_tag_data.m
    Written by: John Brennan Peace
%}
function [out] = get_tag_data(x, output, imsize, maxima_thresh, ...
    num_det, imageFolder)
gridsize = 80;
% imsize = 640;

out_obj = output(:,:,1,:);
out_class = output(:,:,2:31,:);
out_proj = output(:,:,32:end,:);

N = size(output,4);

detections = zeros(N,1,'single');
code = zeros(N,1,'single');
pred_corners = zeros(N,num_det,4,2,'single');

centroids = zeros(N,num_det,2,'single');
% detect_margins = zeros(N,num_det,'single');
% grid_positions = zeros(N,num_det,2,'single');

test_corners = [-63.5,-63.5,1; 63.5,-63.5,1; 63.5,63.5,1; -63.5,63.5,1];
for i = 1:N
    
    m_class = out_class(:,:,:,i);
    out_obj_n = out_obj(:,:,i);
    out_proj_n = out_proj(:,:,:,i);
    [max_obj, idx_obj] = maxk(out_obj_n(:),5);
    raw_max = max_obj;
    raw_idx = idx_obj;
    thresh_obj = (max_obj>maxima_thresh);
    max_obj = thresh_obj.*max_obj;
    idx_obj = thresh_obj.*idx_obj;
    
    max_obj(max_obj==0) = [];
    idx_obj(idx_obj==0) = [];
    
    col = ceil(idx_obj/gridsize);
    row = idx_obj - (col-1)*gridsize;
    
    sub_obj = [col,row];
    
    out_corners = zeros(size(max_obj,1),4,2,'single');    
    raw_corners = zeros(size(max_obj,1),2,4,'single');
    
    if isempty(max_obj)
%         num_det = 0;
        num_det = 0;
        class = 0;
        m_corners = zeros(1,4,2,'single');
        m_cent = zeros(1,2,'single');
        img = draw_pred(x(:,:,:,i),m_corners, out_obj_n, raw_max, raw_idx, m_class, 1);
        imwrite(img,fullfile(imageFolder,sprintf('%05d.png',i)),'png');
    else
        for j = 1:size(max_obj,1)
            proj = out_proj_n(sub_obj(j,2),sub_obj(j,1),:);
            proj = reshape(proj,[2 3]);
            proj_H = vertcat(proj,[0 0 1]);
            corners = test_corners*proj_H;
            corners = corners./corners(:,3);
            corners(:,3) = [];
            raw_corners(j,:,:) = corners';
            out_corners(j,:,:) = (corners' + 8*[sub_obj(j,1);sub_obj(j,2)])';
        end
        clust = eye(size(max_obj,1),size(max_obj,1),'single');
        if i == 236
           a=3; 
        end
        for j = 1:size(max_obj,1)
            for k = 1:size(max_obj,1)
                if j~=k
                    jcor = squeeze(out_corners(j,:,:));
                    kcor = squeeze(out_corners(k,:,:));
                    out_poly = polyshape(jcor(:,2),jcor(:,1));
                    out_poly2 = polyshape(kcor(:,2),kcor(:,1));
                    a_int = area(intersect(out_poly,out_poly2));
                    a_uni = area(union(out_poly,out_poly2));
                    iou = a_int/a_uni;
                    if iou > 0
                        clust(j,k) = 1; 
                    end
                end
                
            end
        end
        
        clust_idx = zeros(size(idx_obj,1),2,'single');
        clust_idx(:,2) = clust_idx(:,2) + 1;
        clust_idx(1,:) = [1,0];
        for j = 2:size(clust,1)
            for k = 1:size(clust,2)
                if (j~=k) && (clust(j,k)==1)
                    a = [clust_idx(j,1),k,j];
                    clust_idx(j,1) = min(a(a~=0));
                    clust_idx(k,2) = 0;
                    if (clust_idx(k,1)~=0) && (k<j)
                        clust_idx(j,1) = min(clust_idx(k,1),clust_idx(j,1));
                    end
                end
            end            
        end

     
        clust_idx(:,1) = (clust_idx(:,1) == 0).*(1:size(idx_obj,1))' + clust_idx(:,1);
        [~,det,~] = unique(clust_idx(:,1), 'rows', 'first');
        
        det = sort(det);
%         num_det = size(det,1);
%         if num_det > 1
%             a = 3;
%         end
        num_det = size(det,1);
        m_idx_obj = idx_obj(det);
        m_max_obj = max_obj(det);
        m_sub_obj = sub_obj(det,:);
        m_obj = out_obj_n;
        m_proj = zeros(num_det,6,'single');
        m_corners = zeros(num_det,4,2,'single');
        m_cent = zeros(num_det,2,'single');
        if num_det == 0
        else
        for j = 1:num_det
            m_proj(j,:) = reshape(out_proj_n(sub_obj(det(j),2),sub_obj(det(j),1),:),[1 6]);
            row = m_sub_obj(j,2);
            col = m_sub_obj(j,1);
            y0 = out_obj_n(row,col);
            yn = out_obj_n(max(1,row-1),col);
            yp = out_obj_n(min(size(out_obj_n,1),row+1),col);
            if (yp > y0)
                yp = yp - y0;
            end
            if (yn > y0)
                yn = yn - y0;  
            end
            rShift = (yn-yp)/(2*(yp + yn - 2*y0));
            yn = out_obj_n(row,max(1,col-1));
            yp = out_obj_n(row,min(size(out_obj_n,2),col+1));        
            if (yp > y0)
                yp = yp - y0;
            end
            if (yn > y0)
                yn = yn - y0;  
            end
            cShift = (yn-yp)/(2*(yp + yn - 2*y0));
            row_d = max(1,min(size(out_obj_n,1),row + rShift));
            col_d = max(1,min(size(out_obj_n,2),col + cShift));
            
            sc = imsize/size(out_obj_n,1);
            pos = [col_d,row_d]*sc;
            pos = pos - ((sc-1)/2);
            real_corners = raw_corners(det(j),:,:) + pos;
            m_corners(j,:,:) = squeeze(real_corners)';
            m_cent(j,:) = pos;
        end
        end
        
        [~, id] = lastwarn;
        warning('off', id)
        [img,class,c_prob] = draw_pred(x(:,:,:,i),m_corners,m_obj,m_max_obj, m_idx_obj, m_class, 0);
%         if c_prob < class_thresh
%             num_det = 0;
%             class = 0;
%             m_corners = zeros(1,4,2,'single');
%             m_cent = zeros(1,2,'single');
%             
%         end
        imwrite(img,fullfile(imageFolder,sprintf('%05d.png',i)),'png');
        
    end   
    if size(m_corners,1) > 1
        if num_det < 2
            a = 3;
        end
        
    end
    dex = max(num_det,1);
    detections(i) = num_det;
    code(i) = class;
    pred_corners(i,1:dex,:,:) = m_corners;
    centroids(i,1:dex,:) = m_cent;
end

out = struct('detections',detections,'code',code,'corners',pred_corners,...
    'centroids',centroids);
end
