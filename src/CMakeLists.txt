add_library(spsub_div SHARED
        spatical_subdivision.cpp
        spacenear_matcher.cpp
        datas_map.cpp)


target_link_libraries(spsub_div ${THIRD_PARTY_LIBS})