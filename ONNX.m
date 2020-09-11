clear variables;
close all;
load('net_bmvc.mat'); %load detector network
net = layerGraph(net);
net = removeLayers(net,{'split2','cat'});


%---- Replace the resnorm layer ----%
net = addLayers(net,depthConcatenationLayer(3,'Name','depth_concat'));
resNormLayer = convolution2dLayer(1,3,'Name','resNorm');
temp = zeros(1,1,3,3,'single');
temp(1,1,1,1) = 1/0.229;
temp(1,1,2,2) = 1/0.224;
temp(1,1,3,3) = 1/0.225;
resNormLayer.Weights = temp;
resNormLayer.WeightLearnRateFactor = 0;
temp = zeros(1,1,3,'single');
temp(1:3) = -[0.485, 0.456, 0.406]./[0.229, 0.224, 0.225];
resNormLayer.Bias = temp;
resNormLayer.BiasLearnRateFactor = 0;
net = replaceLayer(net,'resNorm',resNormLayer);


%---- Replace the mesh layer ----%
net = removeLayers(net,'mesh');
createOnesLayer = convolution2dLayer(640,2,'Name','create_ones_vector');
createOnesLayer.Weights = zeros(640,640,3,2,'single');
createOnesLayer.WeightLearnRateFactor = 0;
createOnesLayer.Bias = ones(1,1,2,'single');
createOnesLayer.BiasLearnRateFactor = 0;
net = addLayers(net,createOnesLayer);
net = connectLayers(net,'resNorm','create_ones_vector');
meshLayer = transposedConv2dLayer(640,2,'Name','mesh');
temp = zeros(640,640,2,2,'single');
[M1,M2] = meshgrid(1:640,1:640);
temp(:,:,1,1) = (2*M1./640)-1;
temp(:,:,2,2) = (2*M2./640)-1;
meshLayer.Weights = temp;
meshLayer.WeightLearnRateFactor = 0;
meshLayer.Bias = zeros(1,1,2,'single');
meshLayer.BiasLearnRateFactor = 0;
net = addLayers(net, meshLayer);
net = connectLayers(net,'create_ones_vector','mesh');
net = addLayers(net, depthConcatenationLayer(2,'Name','mesh_concat'));
net = connectLayers(net,'resNorm','mesh_concat/in1');
net = connectLayers(net,'mesh','mesh_concat/in2');
net = connectLayers(net,'mesh_concat','preconv');


%---- Replace the split layers ----%
split1 = convolution2dLayer(1,1,'Name','split1');
temp = zeros(1,1,37,1,'single');
temp(1,1,1,1) = 1;
split1.Weights = temp;
split1.WeightLearnRateFactor = 0;
split1.Bias = 0;
split1.BiasLearnRateFactor = 0;
net = addLayers(net,split1);
net = connectLayers(net,'fm_conv2','split1');
net = connectLayers(net,'split1','depth_concat/in1');
split2 = convolution2dLayer(1,30,'Name','split2');
temp = zeros(1,1,37,30,'single');
for k = 2:31
    temp(1,1,k,k-1) = 1;
end
split2.Weights = temp;
split2.WeightLearnRateFactor = 0;
split2.Bias = zeros(1,1,30,'single');
split2.BiasLearnRateFactor = 0;
net = addLayers(net,split2);
net = connectLayers(net,'fm_conv2','split2');
net = connectLayers(net,'split2','softmax');
net = connectLayers(net,'softmax','depth_concat/in2');
split3 = convolution2dLayer(1,6,'Name','split3');
temp = zeros(1,1,37,6,'single');
for k = 32:37
    temp(1,1,k,k-31) = 1;
end
split3.Weights = temp;
split3.WeightLearnRateFactor = 0;
split3.Bias = zeros(1,1,6,'single');
split3.BiasLearnRateFactor = 0;
net = addLayers(net,split3);
net = connectLayers(net,'fm_conv2','split3');
net = connectLayers(net,'split3','depth_concat/in3');
net = connectLayers(net,'depth_concat','output');
net = assembleNetwork(net);
save('net_bmvc_onnx.mat', 'net'); %load detector network
exportONNXNetwork(net,'E2ETag.onnx')