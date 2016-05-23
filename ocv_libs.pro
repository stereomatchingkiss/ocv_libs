#-------------------------------------------------
#
# Project created by QtCreator 2015-07-04T18:01:11
#
#-------------------------------------------------

TARGET = ocv_libs
TEMPLATE = lib

CONFIG += c++11
CONFIG -= dll
CONFIG += shared static

msvc:QMAKE_CXXFLAGS_RELEASE += /openmp
gcc:QMAKE_CXXFLAGS_RELEASE += -fopenmp

INCLUDEPATH += ..

include(../pri/arma.pri)
include(../pri/boost.pri)
include(../pri/cv.pri)
include(../pri/dlibs.pri)
include(../pri/eigen.pri)
include(../pri/hdf5.pri)
include(../pri/tbb.pri)

SOURCES += qt/mat_and_qimage.cpp \
    core/histogram.cpp \
    qt/io_img.cpp \
    core/wavelet_transform.cpp \        
    ml/utility/feature_scaling.cpp \    
    ml/utility/activation.cpp \
    core/crop_images.cpp \
    core/bitplane.cpp \
    core/merge_rectangle.cpp \
    cmd/command_prompt_utility.cpp \
    core/attribute.cpp \
    file/utility.cpp \
    core/block_binary_pixel_sum.cpp \
    core/augment_data.cpp \
    cbir/color_descriptor.cpp \
    cbir/f2d_descriptor.cpp \
    cbir/features_indexer.cpp \
    core/montage.cpp \
    saliency/edge_boxes.cpp \
    utility/hsv_range_observer.cpp

HEADERS += qt/mat_and_qimage.hpp \
    core/histogram.hpp \
    qt/io_img.hpp \
    core/for_each.hpp \
    core/utility.hpp \
    core/wavelet_transform.hpp \
    ml/utility/gradient_checking.hpp \
    ml/deep_learning/autoencoder.hpp \
    ml/utility/feature_scaling.hpp \
    ml/deep_learning/propagation.hpp \
    ml/utility/activation.hpp \
    ../3rdLibs/opencv/dev/opencv/modules/core/src/bufferpool.impl.hpp \
    ../3rdLibs/opencv/dev/opencv/modules/core/src/directx.inc.hpp \
    ../3rdLibs/opencv/dev/opencv/modules/core/src/gl_core_3_1.hpp \
    ../3rdLibs/opencv/dev/opencv/modules/core/src/precomp.hpp \    
    profile/measure.hpp \
    eigen/eigen.hpp \
    ml/deep_learning/softmax.hpp \
    core/crop_images.hpp \
    core/bitplane.hpp \
    core/merge_rectangle.hpp \    
    cmd/command_prompt_utility.hpp \
    core/attribute.hpp \
    file/utility.hpp \
    ml/utility/split_train_test.hpp \
    core/perspective_transform.hpp \
    core/block_binary_pixel_sum.hpp \
    tiny_cnn/trainer.hpp \
    core/augment_data.hpp \
    ml/utility/shuffle_data.hpp \
    tiny_cnn/image_converter.hpp \
    cbir/color_descriptor.hpp \
    cbir/f2d_descriptor.hpp \
    cbir/features_indexer.hpp \
    cbir/code_book_builder.hpp \
    cbir/visualize_feature.hpp \
    core/montage.hpp \
    cbir/bovw.hpp \
    cbir/bovw_hist_creator.hpp \
    core/inverted_index.hpp \
    cbir/searcher.hpp \
    arma/type_traits.hpp \
    arma/dist_metric.hpp \
    cbir/build_inverted_index.hpp \
    cbir/spatial_verifier.hpp \
    saliency/edge_boxes.hpp \
    saliency/utility.hpp \
    normalize/mean_cvmat_normalization.hpp \
    utility/hsv_range_observer.hpp \
    dlib/cv_to_dlib.hpp
