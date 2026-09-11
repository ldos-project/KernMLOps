// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#pragma once

#include <cerrno>
#include <climits>
#include <cstdlib>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/JSON.h>
#include <llvm/Support/raw_ostream.h>

// This hook is included by BCC's single bpf_module.cc translation unit. Keep the
// same implementation as the offline harness without modifying upstream CMake.
#include "original_reduce.cpp"

namespace ebpf {

inline int BPFModule::kernmlops_reduce() {
  const char *evidence = std::getenv("KERNMLOPS_BCC_EVIDENCE");
  const char *selection = std::getenv("KERNMLOPS_BCC_REDUCE");
  const char *export_only = std::getenv("KERNMLOPS_BCC_EXPORT_ONLY");
  if ((!evidence || !*evidence) && (!selection || !*selection) &&
      (!export_only || !*export_only))
    return 0;
  auto fail = [](const llvm::Twine &message) {
    llvm::errs() << "original-bcc: " << message << '\n';
    errno = EINVAL;
    return -1;
  };
  if (!evidence || evidence[0] != '/')
    return fail("an absolute KERNMLOPS_BCC_EVIDENCE prefix is required");
  if (llvm::verifyModule(*mod_, &llvm::errs()))
    return fail("invalid original frontend IR");

  llvm::json::Array maps;
  for (const TableDesc *table : tables_)
    maps.push_back(llvm::json::Object{
        {"name", table->name},
        {"fake_fd", table->fake_fd},
        {"type", table->type},
        {"key_size", static_cast<int64_t>(table->key_size)},
        {"value_size", static_cast<int64_t>(table->leaf_size)},
        {"max_entries", static_cast<int64_t>(table->max_entries)},
        {"flags", table->flags},
        {"key_desc", table->key_desc},
        {"value_desc", table->leaf_desc},
    });
  llvm::json::Object events;
  for (const auto &event : perf_events_) {
    llvm::json::Array fields;
    for (const auto &field : event.second)
      fields.push_back(field);
    events[event.first] = std::move(fields);
  }
  llvm::json::Object metadata{
      {"version", 1}, {"llvm_version", LLVM_VERSION_STRING},
      {"maps", std::move(maps)}, {"perf_events", std::move(events)},
      {"export_only", export_only && *export_only},
  };
  // Evidence is exclusive-create: never overwrite an earlier compilation.
  for (const char *suffix : {".before.bc", ".before.ll", ".before.json",
                             ".rewritten.c"}) {
    int fd = -1;
    std::string path = std::string(evidence) + suffix;
    std::error_code error = llvm::sys::fs::openFileForWrite(
        path, fd, llvm::sys::fs::CD_CreateNew);
    if (error)
      return fail(path + ": " + error.message());
    llvm::raw_fd_ostream output(fd, true);
    if (llvm::StringRef(suffix) == ".before.bc")
      llvm::WriteBitcodeToFile(*mod_, output);
    else if (llvm::StringRef(suffix) == ".before.ll")
      mod_->print(output, nullptr);
    else if (llvm::StringRef(suffix) == ".before.json")
      output << llvm::formatv("{0:2}\n", llvm::json::Value(llvm::json::Object(metadata)));
    else
      output << mod_src_;
    output.flush();
    if (output.has_error())
      return fail("cannot write " + path);
  }
  if (selection && *selection) {
    if (selection[0] != '/')
      return fail("KERNMLOPS_BCC_REDUCE must be an absolute JSON path");
    auto wanted = kernmlops::readOriginalJSON(selection);
    if (!wanted)
      return fail(llvm::toString(wanted.takeError()));
    auto after = kernmlops::reduceOriginal(*mod_, metadata, *wanted);
    if (!after)
      return fail(llvm::toString(after.takeError()));
    // Validate private descriptors before applying any size changes. Map
    // creation uses fake_fd_map_, whereas Python's table ABI uses TableDesc.
    for (const auto &value : *after->getArray("maps")) {
      const auto &map = *value.getAsObject();
      const auto name = *map.getString("name");
      const auto fd = *map.getInteger("fake_fd");
      auto fake = fake_fd_map_.find(fd);
      if (fake == fake_fd_map_.end() || std::get<1>(fake->second) != name ||
          *map.getInteger("value_size") > INT_MAX)
        return fail("reduced fake-fd map metadata mismatch");
      TableDesc *table = nullptr;
      for (auto *candidate : tables_)
        if (candidate->name == name)
          table = candidate;
      if (!table || table->fake_fd != fd || table->is_extern)
        return fail("reduced table metadata mismatch or external map");
    }
    for (const auto &value : *after->getArray("maps")) {
      const auto &map = *value.getAsObject();
      const auto name = *map.getString("name");
      const auto fd = *map.getInteger("fake_fd");
      std::get<3>(fake_fd_map_.at(fd)) = *map.getInteger("value_size");
      for (auto *table : tables_)
        if (table->name == name) {
          table->leaf_size = *map.getInteger("value_size");
          table->leaf_desc = map.getString("value_desc")->str();
        }
    }
    for (const auto &event : *after->getObject("perf_events")) {
      auto &fields = perf_events_[event.first.str()];
      fields.clear();
      for (const auto &field : *event.second.getAsArray())
        fields.push_back(field.getAsString()->str());
    }
    if (auto error = kernmlops::writeOriginalEvidence(*mod_, *after, evidence, "after"))
      return fail(llvm::toString(std::move(error)));
  }
  if (export_only && *export_only) {
    // Abort before MCJIT, map creation, and any Python automatic attachment.
    errno = ECANCELED;
    return -1;
  }
  return 0;
}

} // namespace ebpf
