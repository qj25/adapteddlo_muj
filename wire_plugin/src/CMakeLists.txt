# Copyright 2022 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set(MUJOCO_ELASTICITY_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/../..
                              ${CMAKE_CURRENT_SOURCE_DIR}/../../src
)

# Add Eigen3
add_compile_definitions(EIGEN_MPL2_ONLY)
findorfetch(
  USE_SYSTEM_PACKAGE
  OFF
  PACKAGE_NAME
  Eigen3
  LIBRARY_NAME
  eigen
  GIT_REPO
  https://gitlab.com/libeigen/eigen
  GIT_TAG
  464c1d097891a1462ab28bf8bb763c1683883892
  TARGETS
  Eigen3::Eigen
  EXCLUDE_FROM_ALL
)

set(MUJOCO_ELASTICITY_SRCS
    cable.cc
    cable.h
    elasticity.cc
    elasticity.h
    register.cc
    shell.cc
    shell.h
    wire_qst.cc
    wire_qst.h
    wire_utils.cc
    wire_utils.h
)

# Create the target first
add_library(elasticity SHARED)
target_sources(elasticity PRIVATE ${MUJOCO_ELASTICITY_SRCS})

# Then add include directories and link libraries
target_include_directories(elasticity PRIVATE ${MUJOCO_ELASTICITY_INCLUDE} ${EIGEN3_INCLUDE_DIR})
target_link_libraries(elasticity PRIVATE mujoco Eigen3::Eigen)

target_compile_options(
  elasticity
  PRIVATE ${AVX_COMPILE_OPTIONS}
          ${MUJOCO_MACOS_COMPILE_OPTIONS}
          ${EXTRA_COMPILE_OPTIONS}
          ${MUJOCO_CXX_FLAGS}
)
target_link_options(
  elasticity
  PRIVATE
  ${MUJOCO_MACOS_LINK_OPTIONS}
  ${EXTRA_LINK_OPTIONS}
)