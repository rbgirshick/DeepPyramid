function pyra_boxes = im_to_pyra_coords(pyra, boxes)
% boxes is N x 4 where each row is a box in the image specified
% by [x1 y1 x2 y2].
%
% Output is a cell array where cell i holds the pyramid boxes
% coming from the image box

% AUTORIGHTS
% ---------------------------------------------------------
% Copyright (c) 2014, Ross Girshick
%
% This file is part of the DeepPyramid code and is available
% under the terms of the Simplified BSD License provided in
% LICENSE. Please retain this notice and LICENSE if you use
% this file (or any portion of it) in your project.
% ---------------------------------------------------------

boxes = boxes - 1;
for level = 1:pyra.num_levels
  level_boxes = bsxfun(@times, boxes, pyra.scales(level));
  level_boxes = round(level_boxes / pyra.stride);
  level_boxes = level_boxes + 1;
  level_boxes(:, [1 3]) = level_boxes(:, [1 3]) + pyra.padx;
  level_boxes(:, [2 4]) = level_boxes(:, [2 4]) + pyra.pady;
  pyra_boxes{level} = level_boxes;
end
