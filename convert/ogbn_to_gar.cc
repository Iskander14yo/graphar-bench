/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

// Converts OGB-style data to GraphAR format.
//
// Reads Arrow IPC files written by 02_load_gar.py:
//   vertices.arrow  columns: id (int64), f000..fFFF (float32), label (int64)
//   edges.arrow     columns: src_id (int64), dst_id (int64)
//
// Output: GraphAR directory tree at --output-dir.
//
// Usage:
//   ogbn_to_gar --output-dir <dir> --name <dataset> --data-dir <dir>
//               --vertex-chunk <V> --edge-chunk <E>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "arrow/api.h"
#include "arrow/io/file.h"
#include "arrow/ipc/reader.h"

#include "graphar/api/arrow_writer.h"
#include "graphar/fwd.h"
#include "graphar/general_params.h"
#include "graphar/graph_info.h"

namespace fs = std::filesystem;

// ─────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────

#define CHECK_OK(expr, msg)                                   \
  do {                                                        \
    auto _s = (expr);                                         \
    if (!_s.ok()) {                                           \
      std::cerr << "Error [" << (msg) << "]: "               \
                << _s.message() << "\n";                      \
      std::exit(1);                                           \
    }                                                         \
  } while (0)

#define CHECK_RESULT(var, expr, msg)                          \
  auto _r_##var = (expr);                                     \
  if (_r_##var.has_error()) {                                 \
    std::cerr << "Error [" << (msg) << "]: "                  \
              << _r_##var.status().message() << "\n";         \
    std::exit(1);                                             \
  }                                                           \
  auto var = std::move(_r_##var.value());

#define ARROW_CHECK(expr, msg)                                \
  do {                                                        \
    auto _r = (expr);                                         \
    if (!_r.ok()) {                                           \
      std::cerr << "Arrow error [" << (msg) << "]: "         \
                << _r.ToString() << "\n";                     \
      std::exit(1);                                           \
    }                                                         \
  } while (0)

#define ARROW_ASSIGN(var, expr, msg)                          \
  auto _ar_##var = (expr);                                    \
  if (!_ar_##var.ok()) {                                      \
    std::cerr << "Arrow error [" << (msg) << "]: "            \
              << _ar_##var.status().ToString() << "\n";       \
    std::exit(1);                                             \
  }                                                           \
  auto var = std::move(_ar_##var).ValueUnsafe();

// Read an Arrow IPC file into a Table.
std::shared_ptr<arrow::Table> read_ipc_table(const std::string& path) {
  ARROW_ASSIGN(infile, arrow::io::ReadableFile::Open(path), "open " + path);
  ARROW_ASSIGN(reader, arrow::ipc::RecordBatchFileReader::Open(infile),
               "ipc open " + path);
  std::vector<std::shared_ptr<arrow::RecordBatch>> batches;
  for (int i = 0; i < reader->num_record_batches(); ++i) {
    ARROW_ASSIGN(batch, reader->ReadRecordBatch(i), "read batch");
    batches.push_back(batch);
  }
  ARROW_ASSIGN(table, arrow::Table::FromRecordBatches(batches),
               "table from batches");
  return table;
}

// Replicate Python's _compute_vertex_groups so the schema matches exactly.
// GraphAR uses property-name concatenation as a filesystem directory name,
// so keep every group ≤ max_len characters.
std::vector<std::vector<std::string>> compute_vertex_groups(int feat_dim,
                                                            int max_len = 240) {
  std::vector<std::vector<std::string>> groups;
  groups.push_back({"id"});

  std::vector<std::string> current;
  int current_len = 0;
  for (int i = 0; i < feat_dim; ++i) {
    char buf[8];
    std::snprintf(buf, sizeof(buf), "f%03d", i);
    std::string name(buf);
    int added = static_cast<int>(name.size()) + (current.empty() ? 0 : 1);
    if (!current.empty() && current_len + added > max_len) {
      groups.push_back(current);
      current.clear();
      current_len = 0;
      added = static_cast<int>(name.size());
    }
    current.push_back(name);
    current_len += added;
  }
  if (!current.empty()) groups.push_back(current);
  groups.push_back({"label"});
  return groups;
}

// ─────────────────────────────────────────────
// Argument parsing
// ─────────────────────────────────────────────

struct Args {
  std::string output_dir;
  std::string dataset_name;
  std::string data_dir;
  int64_t     vertex_chunk_size = 0;
  int64_t     edge_chunk_size   = 0;
};

static Args parse_args(int argc, char* argv[]) {
  Args a;
  for (int i = 1; i < argc - 1; ++i) {
    std::string flag = argv[i];
    std::string val  = argv[i + 1];
    if      (flag == "--output-dir")    { a.output_dir        = val;              ++i; }
    else if (flag == "--name")          { a.dataset_name      = val;              ++i; }
    else if (flag == "--data-dir")      { a.data_dir          = val;              ++i; }
    else if (flag == "--vertex-chunk")  { a.vertex_chunk_size = std::stoll(val);  ++i; }
    else if (flag == "--edge-chunk")    { a.edge_chunk_size   = std::stoll(val);  ++i; }
  }
  if (a.output_dir.empty() || a.dataset_name.empty() || a.data_dir.empty() ||
      a.vertex_chunk_size <= 0 || a.edge_chunk_size <= 0) {
    std::cerr << "Usage: ogbn_to_gar"
              << " --output-dir <dir> --name <dataset> --data-dir <dir>"
              << " --vertex-chunk <V> --edge-chunk <E>\n";
    std::exit(1);
  }
  return a;
}

// ─────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────

int main(int argc, char* argv[]) {
  auto args = parse_args(argc, argv);

  fs::create_directories(args.output_dir);
  std::string prefix = args.output_dir;
  if (prefix.back() != '/') prefix += '/';

  const int64_t VCS = args.vertex_chunk_size;
  const int64_t ECS = args.edge_chunk_size;

  // ─────────────────────────────────────────────
  // 1. Read vertices.arrow — schema tells us V and F
  // ─────────────────────────────────────────────
  std::cout << "[1/4] Reading vertices.arrow...\n";
  auto v_table = read_ipc_table(args.data_dir + "/vertices.arrow");
  const int64_t V = v_table->num_rows();
  // feature columns are everything except id (first) and label (last)
  const int F = v_table->num_columns() - 2;
  assert(F > 0);
  std::cout << "      " << V << " vertices, " << F << " features\n";

  const int64_t n_vchunks = (V + VCS - 1) / VCS;

  // ─────────────────────────────────────────────
  // 2. Build GraphAR schema
  // ─────────────────────────────────────────────
  std::cout << "[2/4] Building schema...\n";
  auto version = graphar::InfoVersion::Parse("gar/v1").value();
  auto groups  = compute_vertex_groups(F);

  graphar::PropertyGroupVector vertex_pgs;
  for (const auto& group : groups) {
    std::vector<graphar::Property> props;
    for (const auto& name : group) {
      if (name == "id")
        props.emplace_back(name, graphar::int64(), /*primary=*/true, /*nullable=*/false);
      else if (name == "label")
        props.emplace_back(name, graphar::int64(), false, false);
      else
        props.emplace_back(name, graphar::float32(), false, false);
    }
    vertex_pgs.push_back(
        graphar::CreatePropertyGroup(props, graphar::FileType::PARQUET));
  }

  auto vertex_info = graphar::CreateVertexInfo(
      "node", VCS, vertex_pgs, {}, "vertex/node/", version);
  CHECK_OK(vertex_info->Save(prefix + "node.vertex.yaml"), "save vertex yml");

  auto adj_src = graphar::CreateAdjacentList(
      graphar::AdjListType::ordered_by_source, graphar::FileType::PARQUET);
  auto adj_dst = graphar::CreateAdjacentList(
      graphar::AdjListType::ordered_by_dest, graphar::FileType::PARQUET);

  auto edge_info = graphar::CreateEdgeInfo(
      "node", "edge", "node",
      ECS, VCS, VCS,
      /*directed=*/true,
      {adj_src, adj_dst},
      {},
      "edge/node_edge_node/",
      version);
  CHECK_OK(edge_info->Save(prefix + "node_edge_node.edge.yaml"), "save edge yml");

  auto graph_info = graphar::CreateGraphInfo(
      args.dataset_name, {vertex_info}, {edge_info}, {}, prefix, version);
  CHECK_OK(graph_info->Save(prefix + args.dataset_name + ".graph.yml"),
           "save graph yml");

  // ─────────────────────────────────────────────
  // 3. Write vertices
  // ─────────────────────────────────────────────
  std::cout << "[3/4] Writing " << n_vchunks << " vertex chunk(s)...\n";
  CHECK_RESULT(v_writer,
               graphar::VertexPropertyWriter::Make(vertex_info, prefix),
               "make VertexPropertyWriter");
  // WriteTable slices the full table into vertex chunks internally.
  CHECK_OK(v_writer->WriteTable(v_table, 0), "WriteTable vertices");
  CHECK_OK(v_writer->WriteVerticesNum(V), "WriteVerticesNum");

  // Release vertex table before loading edges
  v_table.reset();

  // ─────────────────────────────────────────────
  // 4. Read edges.arrow and write adj lists
  // ─────────────────────────────────────────────
  std::cout << "[4/4] Reading edges.arrow...\n";
  auto e_table_raw = read_ipc_table(args.data_dir + "/edges.arrow");
  const int64_t E = e_table_raw->num_rows();
  std::cout << "      " << E << " edges\n";

  // The file is written in batches, so each column has multiple chunks.
  // Merge into one contiguous chunk so raw_values() covers all rows.
  ARROW_ASSIGN(e_table, e_table_raw->CombineChunks(), "combine edge chunks");
  e_table_raw.reset();

  auto src_col = std::static_pointer_cast<arrow::Int64Array>(
      e_table->column(0)->chunk(0));
  auto dst_col = std::static_pointer_cast<arrow::Int64Array>(
      e_table->column(1)->chunk(0));
  const int64_t* src_ptr = src_col->raw_values();
  const int64_t* dst_ptr = dst_col->raw_values();

  std::vector<int64_t> idx(E);
  std::iota(idx.begin(), idx.end(), 0);

  // Build adj-list Arrow table for one vertex chunk from a sorted index slice.
  auto build_adj_table = [&](int64_t lo,
                              int64_t hi) -> std::shared_ptr<arrow::Table> {
    int64_t n = hi - lo;
    arrow::Int64Builder sb, db;
    sb.Reserve(n).ok();
    db.Reserve(n).ok();
    for (int64_t k = lo; k < hi; ++k) {
      int64_t e = idx[k];
      sb.UnsafeAppend(src_ptr[e]);
      db.UnsafeAppend(dst_ptr[e]);
    }
    std::shared_ptr<arrow::Array> sa, da;
    sb.Finish(&sa).ok();
    db.Finish(&da).ok();
    return arrow::Table::Make(
        arrow::schema({
            arrow::field(graphar::GeneralParams::kSrcIndexCol, arrow::int64()),
            arrow::field(graphar::GeneralParams::kDstIndexCol, arrow::int64()),
        }),
        {sa, da});
  };

  // Sort idx by key_ptr, sweep vertex chunks in one pass.
  auto write_adj = [&](graphar::AdjListType adj_type, const int64_t* key_ptr,
                       const char* label) {
    std::cout << "      Sorting by " << label << "...\n";
    std::sort(idx.begin(), idx.end(),
              [&](int64_t a, int64_t b) { return key_ptr[a] < key_ptr[b]; });

    CHECK_RESULT(writer,
                 graphar::EdgeChunkWriter::Make(edge_info, prefix, adj_type),
                 "make EdgeChunkWriter");

    int64_t ei = 0;
    for (int64_t ci = 0; ci < n_vchunks; ++ci) {
      int64_t v0 = ci * VCS;
      int64_t v1 = std::min(v0 + VCS, V);
      int64_t lo = ei;
      while (ei < E && key_ptr[idx[ei]] < v1) ++ei;
      auto table = build_adj_table(lo, ei);
      CHECK_OK(writer->SortAndWriteAdjListTable(table, ci, 0), "write adj");
      CHECK_OK(writer->WriteEdgesNum(ci, table->num_rows()), "write edges num");
    }
    CHECK_OK(writer->WriteVerticesNum(V), "write vertices num (edges)");
    std::cout << "      " << label << " done.\n";
  };

  write_adj(graphar::AdjListType::ordered_by_source, src_ptr, "source");
  write_adj(graphar::AdjListType::ordered_by_dest,   dst_ptr, "dest");

  std::cout << "Graph written to: " << prefix << "\n";
  return 0;
}
