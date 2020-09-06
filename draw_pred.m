%{
    draw_pred.m
    Written by: John Brennan Peace
%}
function [img,c_pred,c_prob] = draw_pred(img, marker, m_obj, m_max_obj, m_idx_obj,...
    m_class, no_det)
    imsize = 640;
    gridsize = 80;
    im1 = im2single(img);
    im2 = repmat(rgb2gray(im1), [1 1 3]);
    obj_pred = m_obj;
    obj_highest = 0*m_obj;
    for j = 1:size(m_max_obj,1)
        obj_highest(m_idx_obj(j)) = m_max_obj(j);
    end
    obj_pred = imresize(obj_pred, [imsize imsize], 'nearest');
    obj_highest = imresize(obj_highest,[imsize imsize], 'nearest');
    obj_pred = obj_pred - obj_highest;
    if no_det == 0
    xv = marker(1,:,1);
    yv = marker(1,:,2);
    [xp,yp] = meshgrid(1:imsize,1:imsize);
    [in,~] = inpolygon(xp,yp,xv,yv);
    idx_grid = imresize(in, [gridsize gridsize], 'nearest');
    idx_in = find(idx_grid);
    col = ceil(idx_in/gridsize);
    row = idx_in - (col-1)*gridsize;
    class_prob = zeros(1,1,30,'single');
    for j = 1:size(idx_in,1)
        class_prob = class_prob + m_class(row(j),col(j),:);
    end
    class_prob = class_prob/size(idx_in,1);
    [class_prob,class_pred] = maxk(class_prob,3);
    class_prob = squeeze(class_prob);
    class_pred = squeeze(class_pred);
    class_hm = zeros(gridsize,gridsize,3,'single');
    for s = 1:gridsize
        for t = 1:gridsize
            [val,class] = max(m_class(s,t,:));
            if class == class_pred(1)
                class_hm(s,t,1) = val;
            elseif class == class_pred(2)
                class_hm(s,t,2) = val;
            elseif class == class_pred(3)
                class_hm(s,t,3) = val;
            end
        end
    end
    class_hm = imresize(class_hm,[640 640],'nearest');
    for i = 1:size(marker,1)
        cval1 = 'red';
        cval2 = 'green';
        cval3 = 'blue';
        cval4 = 'yellow';
        if (marker(i,1,1) < marker(i,2,1)) && (marker(i,1,2) < marker(i,2,2)) 
           ap = 'RightBottom';
        elseif (marker(i,1,1) < marker(i,2,1)) && (marker(i,1,2) > marker(i,2,2))
           ap = 'RightTop';
        elseif (marker(i,1,1) > marker(i,2,1)) && (marker(i,1,2) < marker(i,2,2))
           ap = 'LeftBottom';
        elseif (marker(i,1,1) > marker(i,2,1)) && (marker(i,1,2) > marker(i,2,2))
           ap = 'LeftTop';
        end
        
        im1 = insertShape(im1,'Line',[marker(i,1,1) marker(i,1,2) marker(i,2,1) marker(i,2,2)],...
            'color',cval1,'LineWidth',2);
        im1 = insertShape(im1,'Line',[marker(i,2,1) marker(i,2,2) marker(i,3,1) marker(i,3,2)],...
            'color',cval2,'LineWidth',2);
        im1 = insertShape(im1,'Line',[marker(i,3,1) marker(i,3,2) marker(i,4,1) marker(i,4,2)],...
            'color',cval3,'LineWidth',2);
        im1 = insertShape(im1,'Line',[marker(i,4,1) marker(i,4,2) marker(i,1,1) marker(i,1,2)],...
            'color',cval4,'LineWidth',2);  
%         im1 = insertText(im1,[marker(i,1,1) marker(i,1,2)],sprintf('%0.2f',m_max_obj(i)),...
%             'FontSize',18,'BoxColor','Green','AnchorPoint',ap);
        
        
%         im2 = insertShape(im2,'Line',[marker(i,1,1) marker(i,1,2) marker(i,2,1) marker(i,2,2)],...
%             'color',cval1,'LineWidth',2);
%         im2 = insertShape(im2,'Line',[marker(i,2,1) marker(i,2,2) marker(i,3,1) marker(i,3,2)],...
%             'color',cval2,'LineWidth',2);
%         im2 = insertShape(im2,'Line',[marker(i,3,1) marker(i,3,2) marker(i,4,1) marker(i,4,2)],...
%             'color',cval3,'LineWidth',2);
%         im2 = insertShape(im2,'Line',[marker(i,4,1) marker(i,4,2) marker(i,1,1) marker(i,1,2)],...
%             'color',cval4,'LineWidth',2);  
%         im2 = insertText(im2,[marker(i,1,1) marker(i,1,2)],sprintf('%0.2f',m_max_obj(i)),...
%             'FontSize',18,'BoxColor','Green','AnchorPoint',ap);
    end

        im3 = im2single(img);
    
        im2(:,:,1) = im2(:,:,1) + 0.5*(obj_pred/m_max_obj(1)) - 0.5*(obj_highest/m_max_obj(1));    
        im2(:,:,2) = im2(:,:,2) + 0.5*(obj_highest/m_max_obj(1)) - 0.5*(obj_pred/m_max_obj(1));   
        im2(:,:,3) = im2(:,:,3) - 0.5*((obj_pred+obj_highest)/m_max_obj(1));

        class_hm1 = class_hm(:,:,1);
        im3(:,:,3) = im3(:,:,3) + 0.5*(class_hm1);
        im3(:,:,2) = im3(:,:,2) - 0.5*(class_hm1);
        im3(:,:,1) = im3(:,:,1) - 0.5*(class_hm1);
        
        class_hm2 = class_hm(:,:,2);
        im3(:,:,3) = im3(:,:,3) + 0.5*(class_hm2);
        im3(:,:,2) = im3(:,:,2) - 0.5*(class_hm2);
        im3(:,:,1) = im3(:,:,1) + 0.5*(class_hm2);
        
        class_hm3 = class_hm(:,:,3);
        im3(:,:,3) = im3(:,:,3) + 0.5*(class_hm3);
        im3(:,:,2) = im3(:,:,2) + 0.5*(class_hm3);
        im3(:,:,1) = im3(:,:,1) - 0.5*(class_hm3);
        value = 'Centers';
        position = [1 1];
%         im2 = insertText(im2,position,value,'FontSize',20);
        im2 = insertText(im2,position,sprintf('Center: %0.2f',m_max_obj(i)),'FontSize',20, ...
            'BoxColor','green','BoxOpacity',1);
    else
        im3 = im2;
        im2(:,:,1) = im2(:,:,1) + 0.5*(obj_pred/m_max_obj(1)) - 0.5*(obj_highest/m_max_obj(1));    
        im2(:,:,2) = im2(:,:,2) + 0.5*(obj_highest/m_max_obj(1)) - 0.5*(obj_pred/m_max_obj(1));   
        im2(:,:,3) = im2(:,:,3) - 0.5*((obj_pred+obj_highest)/m_max_obj(1));
        class_pred = [-1, -1, -1];
        class_prob = [0, 0, 0];
        position = [1 1];
        im2 = insertText(im2,position,sprintf('Center: %0.2f',m_max_obj(1)),'FontSize',20, ...
            'BoxColor','green','BoxOpacity',1);
    end

    im3 = insertText(im3,position,sprintf('Class: %0.0f, Prob: %0.4f',...
        [class_pred(1) class_prob(1)]),'FontSize',20,'BoxColor','blue','TextColor','white','BoxOpacity',1);
    im3 = insertText(im3,position+[0,36],sprintf('Class: %0.0f, Prob: %0.4f',...
        [class_pred(2) class_prob(2)]),'FontSize',20,'BoxColor','magenta','BoxOpacity',1);
    im3 = insertText(im3,position+[0,72],sprintf('Class: %0.0f, Prob: %0.4f',...
        [class_pred(3) class_prob(3)]),'FontSize',20,'BoxColor','cyan','BoxOpacity',1);
    c_pred = class_pred(1);
    c_prob = class_prob(1);
    img = imtile({im2, im3, im1},'GridSize',[1 3]);
%     figure(1);
%     imshow(img);
end