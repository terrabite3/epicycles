CMAKE_MINIMUM_REQUIRED ( VERSION 3.8)

project("jungreis")

# add_definitions("-DCMAKE_WINDOWS_EXPORT_ALL_SYMBOLS")

add_library(${PROJECT_NAME} SHARED
# jungreis.h
jungreis.cpp
)
# set_target_properties(${PROJECT_NAME} PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS true)

# include(GenerateExportHeader)
# generate_export_header(${PROJECT_NAME})