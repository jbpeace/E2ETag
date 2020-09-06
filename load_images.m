%{
    load_images.m
    Written by: John Brennan Peace
%}
function [x,imsize] = load_images(input_folder)
%Load images from input into RAM

dir_input = dir(input_folder);
imsize = size(imread(fullfile(input_folder,dir_input(3).name)),1);
N = length(dir_input);
x = zeros(imsize,imsize,3,N-2,'single');
tic
fprintf('Loading images into ram...\n');
for i = 3:N
    imgfile = dir_input(i).name;
    img = im2single(imread(fullfile(input_folder,imgfile)));
    x(:,:,:,i-2) = img;
end
toc
end