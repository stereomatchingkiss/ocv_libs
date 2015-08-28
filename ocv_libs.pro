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

include(../pri/boost.pri)
include(../pri/cv_dev_cuda.pri)
include(../pri/eigen.pri)

SOURCES += qt/mat_and_qimage.cpp \
    core/histogram.cpp \
    qt/io_img.cpp \
    core/wavelet_transform.cpp \    
    ml/deep_learning/autoencoder.cpp \
    ml/utility/feature_scaling.cpp \
    ml/deep_learning/network_layer.cpp \
    ml/utility/activation.cpp

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
    ml/deep_learning/network_layer.hpp \
    profile/measure.hpp \
    eigen/eigen.hpp \
    ml/deep_learning/softmax.hpp
