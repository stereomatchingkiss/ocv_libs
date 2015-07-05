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

include(cv_boost_libs.pri)

SOURCES += qt/mat_and_qimage.cpp \
    core/histogram.cpp \
    qt/io_img.cpp

HEADERS += qt/mat_and_qimage.hpp \
    core/histogram.hpp \
    qt/io_img.hpp
