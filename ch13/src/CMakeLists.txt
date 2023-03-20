add_library(myslam SHARED
        frame.cpp
        mappoint.cpp
        map.cpp
        camera.cpp
        config.cpp
        feature.cpp
        frontend.cpp
        backend.cpp
        viewer.cpp
        visual_odometry.cpp
        dataset.cpp)#将需要的文件链接生成一个共享库

target_link_libraries(myslam
        ${THIRD_PARTY_LIBS})#链接依赖库和自己的库
