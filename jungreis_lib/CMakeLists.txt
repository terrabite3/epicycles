CMAKE_MINIMUM_REQUIRED ( VERSION 3.8)

project("jungreis")

add_library(${PROJECT_NAME} SHARED
jungreis.cpp
)

add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND cmake -E copy $<TARGET_FILE:${PROJECT_NAME}> ${CMAKE_CURRENT_LIST_DIR}/..
)
