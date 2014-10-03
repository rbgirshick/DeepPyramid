function cnn = init_cnn_model(varargin)
% cnn = init_cnn_model
% Initialize a CNN with caffe
%
% Optional arguments
% net_file   network binary file
% def_file   network prototxt file
% use_gpu    set to false to use CPU (default: true)
% use_caffe  set to false to avoid using caffe (default: true)
%            useful for running on the cluster (must use cached pyramids!)

% AUTORIGHTS
% ---------------------------------------------------------
% Copyright (c) 2014, Ross Girshick
%
% This file is part of the DeepPyramid code and is available
% under the terms of the Simplified BSD License provided in
% LICENSE. Please retain this notice and LICENSE if you use
% this file (or any portion of it) in your project.
% ---------------------------------------------------------

% ------------------------------------------------------------------------
% Options
ip = inputParser;

% network binary file
ip.addParamValue('net_file', ...
    './data/caffe_nets/ilsvrc_2012_train_iter_310k', ...
    @isstr);

% network prototxt file
ip.addParamValue('def_file', ...
    './model-defs/pyramid_cnn_output_conv5_scales_7_plane_1713.prototxt', ...
    @isstr);

% Set use_gpu to false to use the CPU
ip.addParamValue('use_gpu', true, @islogical);

% Set use_caffe to false to avoid using caffe
% (must be used in conjunction with cached features!)
ip.addParamValue('use_caffe', true, @islogical);

ip.parse(varargin{:});
opts = ip.Results;
% ------------------------------------------------------------------------

cnn.binary_file = opts.net_file;
cnn.definition_file = opts.def_file;
cnn.init_key = -1;
cnn.mu = get_channelwise_mean;
cnn.pyra.dimx = 1713;
cnn.pyra.dimy = 1713;
cnn.pyra.stride = 16;
cnn.pyra.num_levels = 7;
cnn.pyra.num_channels = 256;
cnn.pyra.scale_factor = sqrt(2);

if opts.use_caffe
  cnn.init_key = ...
      caffe('init', cnn.definition_file, cnn.binary_file);
  caffe('set_phase_test');
  if opts.use_gpu
    caffe('set_mode_gpu');
  else
    caffe('set_mode_cpu');
  end
end


% ------------------------------------------------------------------------
function mu = get_channelwise_mean()
% ------------------------------------------------------------------------
% load the ilsvrc image mean
data_mean_file = 'ilsvrc_2012_mean.mat';
assert(exist(data_mean_file, 'file') ~= 0);
% input size business isn't likley necessary, but we're doing it
% to be consistent with previous experiments
ld = load(data_mean_file);
mu = ld.image_mean; clear ld;
input_size = 227;
off = floor((size(mu,1) - input_size)/2)+1;
mu = mu(off:off+input_size-1, off:off+input_size-1, :);
mu = sum(sum(mu, 1), 2) / size(mu, 1) / size(mu, 2);
