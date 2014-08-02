function [pyra, im_pyra] = deep_pyramid(im, cnn_model, padx, pady, cache_opts)
% pyra = deep_pyramid(im, cnn_model, padx, pady, cache_opts)
%
% im: a color image
% cnn_model: a handle to the model loaded into caffe
% padx, pady: amount of extra padding to add around each pyramid level
% cache_opts [optional]
%   .cache_dir: directory where cache is
%   .image_id: file name (without extension) inside cache directory
%   .debug [optional]: print info about cache hit/miss


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
    assert(pyra.padx == padx);
    assert(pyra.pady == pady);
    return;
  elseif debug
    warning('Cache file %s not found.', cache_file);
  end
end

if ~exist('padx', 'var') || isempty(padx)
  padx = 0;
end

if ~exist('pady', 'var') || isempty(pady)
  pady = 0;
end

% TODO(rbg) allow of num_levels not 7
num_levels = 7;

% Scale factor to bring the norm of a 10x10 patch of features
% somewhat close to what I've used in R-CNN (so I can use a similar 
% C = 0.001 value in SVMs)
% Z = 50;
% Z = 1;

[pyra, im_pyra] = feat_pyramid(im, cnn_model, num_levels);

imsize = size(im);
pyra.imsize = imsize(1:2);
pyra.num_levels = num_levels;

%% add padding
%for i = 1:pyra.num_levels
%  pyra.feat{i} = padarray(pyra.feat{i} * 1/Z, [pady padx 0], 0);
%end
pyra.valid_levels = true(pyra.num_levels, 1);
pyra.padx = padx;
pyra.pady = pady;


% ------------------------------------------------------------------------
function [pyra, im_pyra] = feat_pyramid(im, cnn_model, num_levels)
% ------------------------------------------------------------------------
% Compute a feature pyramid with caffe

if cnn_model.init_key ~= caffe('get_init_key')
  error('You probably need to load the cnn model into caffe.');
end

% TODO(rbg) yes, hardcoded numbers of joy
batch_width = 1713;
batch_height = 1713;
sz_w = (batch_width - 1) / 16 + 1;
sz_h = sz_w;

%th = tic;
[batch, scales, level_sizes] = ...
    image_pyramid(im, cnn_model, batch_height, batch_width, num_levels);
%fprintf('prep: %.3fs\n', toc(th));

%th = tic;
feat = caffe('forward', {batch});
% standard song and dance of swapping width and height between caffe and matlab
feat = ...
    permute(reshape(feat{1}, [sz_w sz_h 256 num_levels]), [2 1 3 4]);
%fprintf('fwd: %.3fs\n', toc(th));

% crop out feat map levels
%for i = 1:num_levels
%  pyra.feat{i} = feat(1:level_sizes(i,1), 1:level_sizes(i,2), :, i);
%end
pyra.feat = feat;
pyra.scales = scales;
pyra.level_sizes = level_sizes;

im_pyra = batch;

% ------------------------------------------------------------------------
function [batch, scales, level_sizes] = ...
    image_pyramid(im, cnn_model, batch_height, batch_width, num_levels)
% ------------------------------------------------------------------------
% Construct an image pyramid that's ready for feeding directly into caffe
% forward

im = single(im);
% Convert to BGR
im = im(:,:,[3 2 1]);
% Subtract mean (mean of the image mean--one mean per channel)
im = bsxfun(@minus, im, cnn_model.mu);

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
alpha = 2^(1/2);
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
  level_sizes(i+1, :) = ceil((im_sz - 1) / 16 + 1);
  % Make width the fastest dimension (for caffe)
  im = permute(im, [2 1 3]);
  batch(1:im_sz(2), 1:im_sz(1), :, i+1) = im;
end
