%{
    pred_output.m
    Written by: John Brennan Peace
%}
function output = pred_output(x,net)
%Predict network output with images loaded into RAM

gridsize = 80;
output = zeros(gridsize,gridsize,37,size(x,4),'single');
if size(x,4) < 64
    pro_size = ceil(size(x,4)/2);
else
    pro_size = 64;
end
fprintf('Predicting network output...\n');
tic
for i = 1:pro_size:size(x,4)-pro_size
    output(:,:,:,i:i+pro_size) = predict(net,x(:,:,:,i:i+pro_size));
end
output(:,:,:,i+pro_size+1:size(x,4)) = predict(net,x(:,:,:,i+pro_size+1:size(x,4)));
toc
end