file(GLOB examples_SRCS "*.cpp")

foreach(example_src ${examples_SRCS})
  get_filename_component(example ${example_src} NAME_WE)
  add_executable(${example} ${example_src})
  if(EIGEN_STANDARD_LIBRARIES_TO_LINK_TO)
    target_link_libraries(${example} ${EIGEN_STANDARD_LIBRARIES_TO_LINK_TO})
  endif()
  target_link_libraries(${example} Eigen3::Eigen)
  add_custom_command(
    TARGET ${example}
    POST_BUILD
    COMMAND ${example}
    ARGS >${CMAKE_CURRENT_BINARY_DIR}/${example}.out
  )
  add_dependencies(all_examples ${example})
endforeach()
