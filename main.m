%{
    main.m
    Written by: John Brennan Peace
%}
 
input_folder = 'input\images';  %resize images to [640,640,3] before running
output_folder = 'output\images';

data = 'data.mat';

maxima_thresh = 0.5;
num_det = 5;

load('net_bmvc.mat'); %load detector network

[x,imsize] = load_images(input_folder);

output = pred_output(x,net);

out_data = get_tag_data(x, output, imsize, maxima_thresh, num_det, output_folder);

save data out_data