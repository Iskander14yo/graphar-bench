# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# Built automatically by run_all.sh. To build manually from the repo root:
#   cmake -S graphar-bench/convert -B graphar-bench/convert/build -DCMAKE_BUILD_TYPE=Release
#   cmake --build graphar-bench/convert/build --parallel

cmake_minimum_required(VERSION 3.14)
project(ogbn_to_gar CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type")

# ── Arrow & Parquet (system-installed) ──────────────────────────────────────
find_package(Arrow REQUIRED)
find_package(Parquet REQUIRED)

# ── GraphAr library (built in cpp/build, not yet installed) ─────────────────
set(GRAPHAR_CPP_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../cpp")
set(GRAPHAR_BUILD_DIR "${GRAPHAR_CPP_ROOT}/build")

if (NOT EXISTS "${GRAPHAR_BUILD_DIR}/libgraphar.so")
  message(FATAL_ERROR
    "libgraphar.so not found at ${GRAPHAR_BUILD_DIR}.\n"
    "Build the C++ library first:\n"
    "  cd cpp/build && cmake .. && make -j\$(nproc)")
endif()

add_library(graphar SHARED IMPORTED)
set_target_properties(graphar PROPERTIES
  IMPORTED_LOCATION "${GRAPHAR_BUILD_DIR}/libgraphar.so"
  INTERFACE_INCLUDE_DIRECTORIES "${GRAPHAR_CPP_ROOT}/src"
)

# ── Converter binary ─────────────────────────────────────────────────────────
add_executable(ogbn_to_gar ogbn_to_gar.cc)

target_include_directories(ogbn_to_gar PRIVATE
  "${GRAPHAR_CPP_ROOT}/src"
  "${GRAPHAR_CPP_ROOT}/thirdparty"        # for result/result.hpp
)

target_link_libraries(ogbn_to_gar PRIVATE
  graphar
  Arrow::arrow_shared
  Parquet::parquet_shared
)

# RPATH so the binary finds libgraphar.so at runtime without LD_LIBRARY_PATH
set_target_properties(ogbn_to_gar PROPERTIES
  BUILD_RPATH "${GRAPHAR_BUILD_DIR}"
  INSTALL_RPATH "${GRAPHAR_BUILD_DIR}"
)
