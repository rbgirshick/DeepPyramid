function im_boxes = pyra_to_im_coords(pyra, boxes)
% boxes is N x 5 where each row is a box in the format [x1 y1 x2 y2 pyra_level]
% where (x1, y1) is the upper-left corner of the box in pyramid level pyra_level
% and (x2, y2) is the lower-right corner of the box in pyramid level pyra_level
% Assumes 1-based indexing.

sbin = 16;
% pyramid to im scale factors for each scale
scales = sbin ./ pyra.scales;
% pyramid to im scale factors for each pyra level in boxes
scales = scales(boxes(:, end));

% Remove padding from pyramid boxes
boxes(:, [1 3]) = boxes(:, [1 3]) - pyra.padx;
boxes(:, [2 4]) = boxes(:, [2 4]) - pyra.pady;

im_boxes = bsxfun(@times, (boxes(:, 1:4) - 1), scales) + 1;
