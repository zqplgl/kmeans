FILE(GLOB SRC *.cpp)
#add_executable(kmeans ${SRC})
add_library(kmeans SHARED ${SRC})
target_link_libraries(kmeans math_function)