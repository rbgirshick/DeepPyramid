function [pyra, im_pyra] = deep_pyramid(im, cnn_model, cache_opts)
% pyra = deep_pyramid(im, cnn_model, cache_opts)
%
% im: a color image
% cnn_model: a handle to the model loaded into caffe
% cache_opts [optional]
%   .cache_dir: directory where cache is
%   .image_id: file name (without extension) inside cache directory
%   .debug [optional]: print info about cache hit/miss

% AUTORIGHTS
% ---------------------------------------------------------
% Copyright (c) 2014, Ross Girshick
%
% This file is part of the DeepPyramid code and is available
% under the terms of the Simplified BSD License provided in
% LICENSE. Please retain this notice and LICENSE if you use
% this file (or any portion of it) in your project.
% ---------------------------------------------------------

% Load from cache if cache_opts is specified
if exist('cache_opts', 'var') && ~isempty(cache_opts)
  cache_dir = cache_opts.cache_dir;
  image_id = cache_opts.image_id;
  debug = isfield(cache_opts, 'debug') & cache_opts.debug;

  cache_file = fullfile(cache_dir, [image_id '.mat']);
  if exist(cache_file, 'file')
    if debug
      warning('Loaded cache file %s.', cache_file);
    end
    ld = load(cache_file);
    pyra = ld.pyra;
    assert(pyra.padx == 0);
    assert(pyra.pady == 0);
    assert(ndims(pyra.feat) == 4);
    return;
  elseif debug
    warning('Cache file %s not found.', cache_file);
  end
end

[pyra, im_pyra] = feat_pyramid(im, cnn_model);

imsize = size(im);
pyra.imsize = imsize(1:2);
pyra.num_levels = cnn_model.pyra.num_levels;
pyra.stride = cnn_model.pyra.stride;

pyra.valid_levels = true(pyra.num_levels, 1);
pyra.padx = 0;
pyra.pady = 0;


% ------------------------------------------------------------------------
function [pyra, im_pyra] = feat_pyramid(im, cnn_model)
% ------------------------------------------------------------------------
% Compute a feature pyramid with caffe

if cnn_model.init_key ~= caffe('get_init_key')
  error('You probably need to load the cnn model into caffe.');
end

% we're assuming the input height and width are the same
assert(cnn_model.pyra.dimx == cnn_model.pyra.dimy);
% compute output width and height
sz_w = (cnn_model.pyra.dimx - 1) / cnn_model.pyra.stride + 1;
sz_h = sz_w;
sz_l = cnn_model.pyra.num_levels;
sz_c = cnn_model.pyra.num_channels;

%th = tic;
[batch, scales, level_sizes] = image_pyramid(im, cnn_model);
%fprintf('prep: %.3fs\n', toc(th));

%th = tic;
feat = caffe('forward', {batch});
% standard song and dance of swapping width and height between caffe and matlab
feat = permute(reshape(feat{1}, [sz_w sz_h sz_c sz_l]), [2 1 3 4]);
%fprintf('fwd: %.3fs\n', toc(th));

pyra.feat = feat;
pyra.scales = scales;
pyra.level_sizes = level_sizes;

im_pyra = batch;

% ------------------------------------------------------------------------
function [batch, scales, level_sizes] = image_pyramid(im, cnn_model)
% ------------------------------------------------------------------------
% Construct an image pyramid that's ready for feeding directly into caffe
% forward

batch_width = cnn_model.pyra.dimx;
batch_height = cnn_model.pyra.dimy;
num_levels = cnn_model.pyra.num_levels;
stride = cnn_model.pyra.stride;

im = single(im);
% Convert to BGR
im = im(:,:,[3 2 1]);
% Subtract mean (mean of the image mean--one mean per channel)
im = bsxfun(@minus, im, cnn_model.mu);

% scale = base scale relative to input image
im_sz = size(im);
if im_sz(1) > im_sz(2)
  height = batch_height;
  width = NaN;
  scale = height / im_sz(1);
else
  height = NaN;
  width = batch_width;
  scale = width / im_sz(2);
end
im_orig = im;

batch = zeros(batch_width, batch_height, 3, num_levels, 'single');
alpha = cnn_model.pyra.scale_factor;
scales = scale*(alpha.^-(0:num_levels-1))';
level_sizes = zeros(num_levels, 2);
for i = 0:num_levels-1
  if i == 0
    im = imresize(im_orig, [height width], 'bilinear');
  else
    im = imresize(im_orig, scales(i+1), 'bilinear');
  end
  im_sz = size(im);
  im_sz = im_sz(1:2);
  level_sizes(i+1, :) = ceil((im_sz - 1) / stride + 1);
  % Make width the fastest dimension (for caffe)
  im = permute(im, [2 1 3]);
  batch(1:im_sz(2), 1:im_sz(1), :, i+1) = im;
end
