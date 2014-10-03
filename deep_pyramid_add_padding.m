function pyra = deep_pyramid_add_padding(pyra, padx, pady, uniform_output)
% pyra = deep_pyramid_add_padding(pyra, padx, pady, uniform_output)
%
% pyra: a pyramid computed using deep_pyramid.m
% padx, pady: amount of extra padding to add around each pyramid level
% uniform_output: true => each pyramid level is embedded in the upper-left
%     corner of each 3D plane inside a 4D array;
%     false => each pyramid level is stored as a 3D array with the height
%     and width of that level (same as in the DPM HOG feature pyramid code)

% AUTORIGHTS
% ---------------------------------------------------------
% Copyright (c) 2014, Ross Girshick
%
% This file is part of the DeepPyramid code and is available
% under the terms of the Simplified BSD License provided in
% LICENSE. Please retain this notice and LICENSE if you use
% this file (or any portion of it) in your project.
% ---------------------------------------------------------

pyra.padx = padx;
pyra.pady = pady;

if ~exist('uniform_output', 'var') || isempty(uniform_output)
  uniform_output = false;
end

if uniform_output
  pyra.feat = padarray(pyra.feat, [pady padx 0 0], 0);
else
  % This mimics the output of the DPM featpyramid.m code
  feat = pyra.feat;
  pyra.feat = cell(pyra.num_levels, 1);
  for i = 1:pyra.num_levels
    % crop out feat map levels
    pyra.feat{i} = feat(1:pyra.level_sizes(i, 1), ...
                        1:pyra.level_sizes(i, 2), :, i);
    % add padding
    pyra.feat{i} = padarray(pyra.feat{i}, [pady padx 0], 0);
  end
end

