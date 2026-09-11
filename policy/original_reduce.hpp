// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#pragma once

#include <llvm/ADT/StringRef.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/JSON.h>

namespace llvm { class Module; }

namespace kernmlops {

// Operates on the original, retargeted BCC frontend module, before map creation.
// On error the caller MUST discard the module; there is no unreduced fallback.
llvm::Expected<llvm::json::Object> reduceOriginal(
    llvm::Module &module, const llvm::json::Object &metadata,
    const llvm::json::Object &selection);

llvm::Expected<llvm::json::Object> readOriginalJSON(llvm::StringRef path);

// Exclusive-create PREFIX.STAGE.{bc,ll,json}; never overwrite evidence.
llvm::Error writeOriginalEvidence(llvm::Module &module,
                                 const llvm::json::Object &metadata,
                                 llvm::StringRef prefix, llvm::StringRef stage);

} // namespace kernmlops
