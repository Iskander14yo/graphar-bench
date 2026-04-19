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
// Reads Arrow IPC files written by 02_load_gar.py.
//
// Small datasets (CSV format): single edges.arrow with src_id + dst_id columns.
// Large datasets (binary format): vertices.arrow streamed in VCS-row batches,
//   edges_src.arrow (src_id) + edges_dst.arrow (dst_id) streamed to avoid a
//   single 26 GB allocation.
//
// Output: GraphAR directory tree at --output-dir.
//
// Usage:
//   ogbn_to_gar --output-dir <dir> --name <dataset> --data-dir <dir>
//               --vertex-chunk <V> --edge-chunk <E>
//               [--num-vertices N --feat-dim F]   # binary format only

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

// Read an Arrow IPC file into a Table (loads all batches eagerly).
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
  // Optional: provided for large binary datasets so vertices can be streamed
  // without loading the full Arrow table to discover V and F.
  int64_t     num_vertices = 0;
  int         feat_dim     = 0;
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
    else if (flag == "--num-vertices")  { a.num_vertices      = std::stoll(val);  ++i; }
    else if (flag == "--feat-dim")      { a.feat_dim          = std::stoi(val);   ++i; }
  }
  if (a.output_dir.empty() || a.dataset_name.empty() || a.data_dir.empty() ||
      a.vertex_chunk_size <= 0 || a.edge_chunk_size <= 0) {
    std::cerr << "Usage: ogbn_to_gar"
              << " --output-dir <dir> --name <dataset> --data-dir <dir>"
              << " --vertex-chunk <V> --edge-chunk <E>"
              << " [--num-vertices N --feat-dim F]\n";
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
  // 1. Determine V and F
  // ─────────────────────────────────────────────
  // For large binary datasets Python passes --num-vertices and --feat-dim so
  // that we can build the schema without loading the full vertex table.
  // For small CSV datasets V and F are derived from the Arrow file.

  const bool stream_vertices = (args.num_vertices > 0 && args.feat_dim > 0);
  int64_t V = args.num_vertices;
  int     F = args.feat_dim;
  // Non-streaming: must load vertices.arrow *before* building vertex YAML so F is
  // correct. (Previously F stayed 0 and node.vertex.yaml listed only id+label.)
  std::shared_ptr<arrow::Table> v_table_eager;

  if (!stream_vertices) {
    std::cout << "[1/4] Reading vertices.arrow...\n";
    v_table_eager = read_ipc_table(args.data_dir + "/vertices.arrow");
    V = v_table_eager->num_rows();
    F = static_cast<int>(v_table_eager->num_columns()) - 2;
    if (v_table_eager->num_columns() < 2 || F < 0) {
      std::cerr << "vertices.arrow: expected columns id, …features…, label (got "
                << v_table_eager->num_columns() << ").\n";
      std::exit(1);
    }
    std::cout << "      " << V << " vertices, " << F << " features\n";
  } else {
    std::cout << "[1/4] Streaming vertices.arrow (" << V << " rows, F=" << F << ")...\n";
    // F is already set from args; open the reader to validate schema.
    ARROW_ASSIGN(v_probe, arrow::io::ReadableFile::Open(args.data_dir + "/vertices.arrow"),
                 "open vertices");
    ARROW_ASSIGN(v_probe_reader, arrow::ipc::RecordBatchFileReader::Open(v_probe),
                 "ipc vertices probe");
    // F from schema: num_fields - 2 (id + label)
    int F_from_schema = v_probe_reader->schema()->num_fields() - 2;
    if (F_from_schema != F) {
      std::cerr << "Warning: --feat-dim=" << F << " but schema has "
                << F_from_schema << " feature columns; using schema value.\n";
      F = F_from_schema;
    }
  }

  // ─────────────────────────────────────────────
  // 2. Build GraphAR schema
  // ─────────────────────────────────────────────
  std::cout << "[2/4] Building schema (F=" << F << ")...\n";
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

  auto make_edge_info = [&](int64_t /*v_count*/) {
    return graphar::CreateEdgeInfo(
        "node", "edge", "node",
        ECS, VCS, VCS,
        /*directed=*/true,
        {adj_src},
        {},
        "edge/node_edge_node/",
        version);
  };

  // ─────────────────────────────────────────────
  // 3. Write vertices
  // ─────────────────────────────────────────────
  CHECK_RESULT(v_writer,
               graphar::VertexPropertyWriter::Make(vertex_info, prefix),
               "make VertexPropertyWriter");

  if (stream_vertices) {
    // Each Python Arrow batch = exactly VCS rows (aligned on purpose), so
    // batch_index == chunk_index and we call WriteTable once per batch.
    ARROW_ASSIGN(v_infile, arrow::io::ReadableFile::Open(args.data_dir + "/vertices.arrow"),
                 "open vertices");
    ARROW_ASSIGN(v_reader, arrow::ipc::RecordBatchFileReader::Open(v_infile),
                 "ipc vertices");

    int num_batches = v_reader->num_record_batches();
    std::cout << "      " << num_batches << " batch(es) to write\n";

    for (int i = 0; i < num_batches; ++i) {
      ARROW_ASSIGN(batch, v_reader->ReadRecordBatch(i), "read vertex batch");
      ARROW_ASSIGN(table, arrow::Table::FromRecordBatches({batch}),
                   "vertex batch to table");
      // Chunk index equals batch index because Python wrote exactly VCS rows
      // per batch (except possibly the last).
      CHECK_OK(v_writer->WriteTable(table, i), "WriteTable vertices");
      std::cout << "      chunk " << (i + 1) << "/" << num_batches << "\r" << std::flush;
    }
    std::cout << "\n";
  } else {
    int64_t n_vchunks = (V + VCS - 1) / VCS;
    std::cout << "      Writing " << n_vchunks << " vertex chunk(s)...\n";
    CHECK_OK(v_writer->WriteTable(v_table_eager, 0), "WriteTable vertices");
    v_table_eager.reset();
  }

  CHECK_OK(v_writer->WriteVerticesNum(V), "WriteVerticesNum");
  const int64_t n_vchunks = (V + VCS - 1) / VCS;

  // ─────────────────────────────────────────────
  // 4. Read edges and write adj lists
  // ─────────────────────────────────────────────
  auto edge_info = make_edge_info(V);
  CHECK_OK(edge_info->Save(prefix + "node_edge_node.edge.yaml"), "save edge yml");
  auto graph_info = graphar::CreateGraphInfo(
      args.dataset_name, {vertex_info}, {edge_info}, {}, prefix, version);
  CHECK_OK(graph_info->Save(prefix + args.dataset_name + ".graph.yml"),
           "save graph yml");

  // Detect whether we have split edge files (binary format) or a combined
  // edges.arrow (CSV format).
  const std::string src_path = args.data_dir + "/edges_src.arrow";
  const std::string dst_path = args.data_dir + "/edges_dst.arrow";
  const std::string combined_path = args.data_dir + "/edges.arrow";
  const bool split_edges = fs::exists(src_path) && fs::exists(dst_path);

  int64_t E = 0;
  const int64_t* src_ptr = nullptr;
  const int64_t* dst_ptr = nullptr;

  // Keep these in scope until the sort+write is done.
  std::shared_ptr<arrow::Table> src_combined, dst_combined;
  std::shared_ptr<arrow::Table> e_combined;

  if (split_edges) {
    std::cout << "[4/4] Loading edges_src.arrow + edges_dst.arrow...\n";
    auto src_raw = read_ipc_table(src_path);
    auto dst_raw = read_ipc_table(dst_path);
    E = src_raw->num_rows();
    assert(dst_raw->num_rows() == E);
    std::cout << "      " << E << " edges\n";

    ARROW_ASSIGN(src_c, src_raw->CombineChunks(), "combine src chunks");
    ARROW_ASSIGN(dst_c, dst_raw->CombineChunks(), "combine dst chunks");
    src_raw.reset();
    dst_raw.reset();
    src_combined = src_c;
    dst_combined = dst_c;

    src_ptr = std::static_pointer_cast<arrow::Int64Array>(
        src_combined->column(0)->chunk(0))->raw_values();
    dst_ptr = std::static_pointer_cast<arrow::Int64Array>(
        dst_combined->column(0)->chunk(0))->raw_values();
  } else {
    std::cout << "[4/4] Loading edges.arrow...\n";
    auto e_table_raw = read_ipc_table(combined_path);
    E = e_table_raw->num_rows();
    std::cout << "      " << E << " edges\n";

    // The file is written in batches, so columns are chunked.
    // Merge into contiguous arrays for raw_values().
    ARROW_ASSIGN(e_c, e_table_raw->CombineChunks(), "combine edge chunks");
    e_table_raw.reset();
    e_combined = e_c;

    src_ptr = std::static_pointer_cast<arrow::Int64Array>(
        e_combined->column(0)->chunk(0))->raw_values();
    dst_ptr = std::static_pointer_cast<arrow::Int64Array>(
        e_combined->column(1)->chunk(0))->raw_values();
  }

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
            arrow::field(graphar::GeneralParams::kSrcIndexCol, arrow::int64(),
                         /*nullable=*/false),
            arrow::field(graphar::GeneralParams::kDstIndexCol, arrow::int64(),
                         /*nullable=*/false),
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

  std::cout << "Graph written to: " << prefix << "\n";
  return 0;
}
