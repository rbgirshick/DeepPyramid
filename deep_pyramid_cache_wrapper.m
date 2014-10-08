function pyra = deep_pyramid_cache_wrapper(im, cnn_model, cache_opts)
% pyra = deep_pyramid_cache_wrapper(im, cnn_model, cache_opts)
%
% Load a feature pyramid from a cache on disk, optionally saving if
% there's a cache miss.
%
% cache_opts.cache_file
%           .debug
%           .write_on_miss
%           .exists

% AUTORIGHTS
% ---------------------------------------------------------
% Copyright (c) 2014, Ross Girshick
%
% This file is part of the DeepPyramid code and is available
% under the terms of the Simplified BSD License provided in
% LICENSE. Please retain this notice and LICENSE if you use
% this file (or any portion of it) in your project.
% ---------------------------------------------------------

debug = isfield(cache_opts, 'debug') && cache_opts.debug;
write_on_miss = isfield(cache_opts, 'write_on_miss') && ...
                cache_opts.write_on_miss;

cache_file = cache_opts.cache_file;
if exist(cache_file, 'file')
  ld = load(cache_file);
  pyra = ld.pyra;
  if debug
    warning('Loaded cache file %s.', cache_file);
  end
else
  if debug
    warning('Cache file %s not found.', cache_file);
  end
  if write_on_miss
    pyra = deep_pyramid(im, cnn_model);
    save(cache_opts.cache_file, 'pyra');
    if debug
      warning('Cache file % saved.', cache_file);
    end
  end
end
