TBB_PATH = $$PWD/../3rdLibs/tbb/tbb43_20150611oss_win/tbb43_20150611oss

INCLUDEPATH += $${TBB_PATH}/include

LIB_SUFFIX = a

win32-msvc*{

  LIB_SUFFIX = lib

  CONFIG(debug, debug|release) {
    LIBS += $${TBB_PATH}/bin/ia32/vc12/tbb_debug.lib
	LIBS += $${TBB_PATH}/bin/ia32/vc12/tbbmalloc_debug.lib
  } else {    
    LIBS += $${TBB_PATH}/bin/ia32/vc12/tbb.lib
	LIBS += $${TBB_PATH}/bin/ia32/vc12/tbbmalloc.lib
  } #config end

} #win32 end
