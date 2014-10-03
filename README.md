## DeepPyramid

DeepPyramid is a simple toolkit for building feature pyramids from deep convolutional networks.
The DeepPyramid data structure is nearly identical to the HOG feature pyramid created by the featpyramid.m function in the [voc-dpm](https://github.com/rbgirshick/voc-dpm) code.

### References

This code was used in our [tech report](http://arxiv.org/pdf/1409.5403v2.pdf) about the relationship between deformable part models and convolutional networks.

    @article{girshick14dpdpm,
        author    = {Ross Girshick and Forrest Iandola and Trevor Darrell and Jitendra Malik},
        title     = {Deformable Part Models are Convolutional Neural Networks},
        journal   = {CoRR},
        year      = {2014},
        volume    = {abs/1409.5403},
        url       = {http://arxiv.org/abs/1409.5403},
        year      = {2014}
    }


### Installation

0. **Prerequisites** 
  0. MATLAB (tested with 2014a on 64-bit Linux)
  0. Caffe's [prerequisites](http://caffe.berkeleyvision.org/installation.html#prequequisites)
0. **Install Caffe** (this is the most complicated part)
  0. Follow the [Caffe installation instructions](http://caffe.berkeleyvision.org/installation.html)
  0. Let's call the place where you installed caffe `$CAFFE_ROOT` (you can run `export CAFFE_ROOT=$(pwd)`)
  0. **Important:** Make sure to compile the Caffe MATLAB wrapper, which is not built by default: `make matcaffe`
  0. **Important:** Make sure to run `cd $CAFFE_ROOT/data/ilsvrc12 && ./get_ilsvrc_aux.sh` to download the ImageNet image mean
  0. DeepPyramid has been tested with master and dev at the time of this writing
0. **Get DeepPyramid**
  0. `git clone https://github.com/rbgirshick/DeepPyramid.git`
  0. If you haven't installed R-CNN, you'll need to download its [models](http://www.cs.berkeley.edu/~rbg/r-cnn-release1-data.tgz)
  1. Copy R-CNN's non-finetuned ImageNet network `<rcnnpath>/data/caffe_nets/ilsvrc_2012_train_iter_310k` to `<deeppyramidpath>/data/caffe_nets/ilsvrc_2012_train_iter_310k` (or just create a symlink).

### Usage

1. Run matlab from inside the DeepPyramid code directory
2. Add the `matcaffe` mex function to your path (`addpath /path/to/caffe/matlab/caffe`)
3. Run the demo `demo_deep_pyramid`

### Uses

DeepPyramid can be used for implementing DPMs on deep convolutional network features, rather than HOG features. It can also be used whenever you need a dense multiscale pyramid of image features.

### Caveats

The implementation is designed to be simple and as a result is very inefficient. There are a variety of ways to speed it up, and they will be done in the feature. For now, it takes about 0.5 to 0.6 seconds to compute a feature pyramid on an NVIDIA Titan GPU, which is acceptable.
