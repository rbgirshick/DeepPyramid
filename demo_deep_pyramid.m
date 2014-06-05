function pyra = demo_deep_pyramid()

if exist('caffe') ~= 3
  error('You must add matcaffe to your path.');
end

if ~exist('data/caffe_nets/ilsvrc_2012_train_iter_310k')
  error('You need the CNN model in %s.', ...
      'data/caffe_nets/ilsvrc_2012_train_iter_310k');
end

% real use settings (compute features using the GPU)
%USE_GPU = true;
%USE_CACHE = false;
%USE_CAFFE = true;

% fast demo settings
USE_GPU = false;
USE_CACHE = true;
USE_CAFFE = false;

im = imread('000084.jpg');
bbox = [263 145 381 225];

cnn = init_cnn_model('use_gpu', USE_GPU, 'use_caffe', USE_CAFFE);

if USE_CACHE
  cache_opts.cache_dir = '.';
  cache_opts.image_id = 'cached_pyra';
  cache_opts.debug = true;
else
  cache_opts = [];
end
padx = 0;
pady = 0;

th = tic;
pyra = deep_pyramid(im, cnn, padx, pady, cache_opts);
fprintf('deep_pyramid took %.3fs\n', toc(th));

fprintf('Press any key to loop through feature channels.\n');
for channel = 1:256
  vis_pyramid(im, pyra, bbox, channel);
  pause;
end


function vis_pyramid(im, pyra, bbox, channel)

pyra_boxes = im_to_pyra_coords(pyra, bbox);

clf;

rows = 2;
cols = 4;
subplot(rows, cols, 1);
imagesc(im);
axis image;
rectangle('Position', bbox_to_xywh(bbox), 'EdgeColor', 'g');
title(sprintf('input image feature %d', channel));

max_val = 0;
for level = 1:pyra.num_levels
  f = pyra.feat{level}(:,:,channel);
  max_val = max(max_val, max(f(:)));
end

ld = load('green_colormap');
colormap(ld.map); clear ld;

for level = 1:pyra.num_levels
  subplot(rows, cols, level+1);
  imagesc(pyra.feat{level}(:,:,channel), [0 max_val]);
  axis image;
  rectangle('Position', bbox_to_xywh(pyra_boxes{level}), 'EdgeColor', 'r');
  title(sprintf('level %d', level));

  % project pyramid box back to image and display as red
  im_bbox = pyra_to_im_coords(pyra, [pyra_boxes{level} level]);
  subplot(rows, cols, 1);
  rectangle('Position', bbox_to_xywh(im_bbox), 'EdgeColor', 'r');
  %text(im_bbox(1), im_bbox(2), sprintf('%d', level));
end

function xywh = bbox_to_xywh(bbox)
xywh = [bbox(1) bbox(2) bbox(3)-bbox(1)+1 bbox(4)-bbox(2)+1];
