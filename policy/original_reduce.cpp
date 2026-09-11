// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "original_reduce.hpp"

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/StringSet.h>
#include <llvm/Analysis/ValueTracking.h>
#include <llvm/BinaryFormat/Dwarf.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/Config/llvm-config.h>
#include <llvm/IR/DIBuilder.h>
#include <llvm/IR/DebugInfo.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/InstIterator.h>
#include <llvm/IR/IntrinsicInst.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/TargetParser/Triple.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <string>
#include <vector>

#if LLVM_VERSION_MAJOR != 18
#error "Original BCC reduction requires LLVM 18"
#endif

namespace kernmlops {
namespace {
using namespace llvm;

Error invalid(const Twine &message) {
  return createStringError(inconvertibleErrorCode(), message);
}

uint64_t helperID(const CallBase &call) {
  auto *callee = dyn_cast<ConstantExpr>(call.getCalledOperand());
  if (!callee || callee->getOpcode() != Instruction::IntToPtr)
    return 0;
  auto *id = dyn_cast<ConstantInt>(callee->getOperand(0));
  return id && id->getValue().getActiveBits() <= 64 ? id->getZExtValue() : 0;
}

std::optional<int64_t> mapFD(Value *value) {
  auto *cast = dyn_cast<IntToPtrInst>(value);
  auto *call = cast ? dyn_cast<CallBase>(cast->getOperand(0)) : nullptr;
  if (!call || !call->getCalledFunction() ||
      call->getCalledFunction()->getName() != "llvm.bpf.pseudo" ||
      call->arg_size() != 2)
    return std::nullopt;
  auto *kind = dyn_cast<ConstantInt>(call->getArgOperand(0));
  auto *fd = dyn_cast<ConstantInt>(call->getArgOperand(1));
  if (!kind || !kind->isOne() || !fd || !fd->getValue().isSignedIntN(64))
    return std::nullopt;
  return fd->getSExtValue();
}

Error optimize(Module &module, bool prepare = false) {
  LoopAnalysisManager lam;
  FunctionAnalysisManager fam;
  CGSCCAnalysisManager cgam;
  ModuleAnalysisManager mam;
  PassBuilder pb;
  pb.registerModuleAnalyses(mam);
  pb.registerCGSCCAnalyses(cgam);
  pb.registerFunctionAnalyses(fam);
  pb.registerLoopAnalyses(lam);
  pb.crossRegisterProxies(lam, fam, cgam, mam);
  ModulePassManager pm;
  // Do not run SROA until output storage and original field identities are known.
  if (auto error = pb.parsePassPipeline(
          pm, prepare ? "globalopt,always-inline,function(mem2reg,instcombine,simplifycfg)"
                      : "always-inline,function(sroa,mem2reg,instcombine,simplifycfg,dse,adce,"
                        "loop(loop-deletion)),globaldce"))
    return error;
  pm.run(module, mam);
  return Error::success();
}

DIType *unqualified(DIType *type) {
  while (auto *derived = dyn_cast_or_null<DIDerivedType>(type)) {
    unsigned tag = derived->getTag();
    if (tag != dwarf::DW_TAG_typedef && tag != dwarf::DW_TAG_const_type &&
        tag != dwarf::DW_TAG_volatile_type && tag != dwarf::DW_TAG_restrict_type)
      break;
    type = derived->getBaseType();
  }
  return type;
}

Expected<DICompositeType *> debugRecord(const DebugInfoFinder &finder, StringRef name) {
  DICompositeType *result = nullptr;
  for (DIType *type : finder.types()) {
    auto *record = dyn_cast<DICompositeType>(type);
    if (!record || record->getTag() != dwarf::DW_TAG_structure_type ||
        record->isForwardDecl() || record->getName() != name)
      continue;
    if (result && result != record)
      return invalid("ambiguous debug record: " + name);
    result = record;
  }
  if (!result)
    return invalid("missing full debug record: " + name);
  return result;
}

Expected<SmallVector<DIDerivedType *>> members(
    StructType *type, DICompositeType *debug, const DataLayout &dl) {
  if (!type || type->isOpaque() || type->isPacked() || !type->isSized() ||
      !debug || debug->getSizeInBits() != dl.getTypeAllocSizeInBits(type) ||
      debug->getElements().size() != type->getNumElements())
    return invalid("LLVM/debug struct layout mismatch");
  SmallVector<DIDerivedType *> result;
  const StructLayout *layout = dl.getStructLayout(type);
  for (DINode *node : debug->getElements()) {
    auto *member = dyn_cast<DIDerivedType>(node);
    unsigned index = result.size();
    if (!member || member->getTag() != dwarf::DW_TAG_member ||
        member->isBitField() || member->getName().empty() ||
        member->getOffsetInBits() != layout->getElementOffsetInBits(index) ||
        member->getSizeInBits() != dl.getTypeAllocSizeInBits(type->getElementType(index)))
      return invalid("unsupported or mismatched struct member");
    result.push_back(member);
  }
  return result;
}

// Build a distinct DI type with the source name. Never RAUW the shared source
// record: the RSS hash and its raw-tracepoint producer must retain their ABI.
DICompositeType *cloneDebug(DIBuilder &dib, DICompositeType *old,
                           StructType *type, ArrayRef<DIDerivedType *> source,
                           ArrayRef<DIType *> bases, const DataLayout &dl) {
  auto *layout = dl.getStructLayout(type);
  auto *result = dib.createStructType(
      old->getScope(), old->getName(), old->getFile(), old->getLine(),
      layout->getSizeInBits(), layout->getAlignment().value() * 8,
      DINode::FlagZero, nullptr, dib.getOrCreateArray({}));
  SmallVector<Metadata *> fields;
  for (unsigned i = 0; i < source.size(); ++i)
    fields.push_back(dib.createMemberType(
        result, source[i]->getName(), source[i]->getFile(), source[i]->getLine(),
        dl.getTypeAllocSizeInBits(type->getElementType(i)),
        dl.getABITypeAlign(type->getElementType(i)).value() * 8,
        layout->getElementOffsetInBits(i), DINode::FlagZero, bases[i]));
  dib.replaceArrays(result, dib.getOrCreateArray(fields));
  dib.retainType(result);
  return result;
}

Expected<json::Value> descriptor(Type *type, DIType *debug, const DataLayout &dl) {
  debug = unqualified(debug);
  if (auto *integer = dyn_cast<IntegerType>(type)) {
    auto *basic = dyn_cast_or_null<DIBasicType>(debug);
    if (!basic || basic->getSizeInBits() != integer->getBitWidth() ||
        (basic->getEncoding() != dwarf::DW_ATE_signed &&
         basic->getEncoding() != dwarf::DW_ATE_unsigned &&
         basic->getEncoding() != dwarf::DW_ATE_signed_char &&
         basic->getEncoding() != dwarf::DW_ATE_unsigned_char))
      return invalid("only ordinary integer record fields are supported");
    return json::Value(basic->getName().str());
  }
  auto *record = dyn_cast<StructType>(type);
  auto *di = dyn_cast_or_null<DICompositeType>(debug);
  auto fields = members(record, di, dl);
  if (!fields)
    return fields.takeError();
  json::Array entries;
  uint64_t cursor = 0;
  auto pad = [&](uint64_t end) {
    if (end > cursor)
      entries.push_back(json::Array{"__pad_" + std::to_string(cursor), "char",
                                    json::Array{int64_t(end - cursor)}});
  };
  for (unsigned i = 0; i < fields->size(); ++i) {
    uint64_t offset = dl.getStructLayout(record)->getElementOffset(i);
    pad(offset);
    auto entry = descriptor(record->getElementType(i), (*fields)[i]->getBaseType(), dl);
    if (!entry)
      return entry.takeError();
    entries.push_back(json::Array{(*fields)[i]->getName().str(), std::move(*entry)});
    cursor = offset + dl.getTypeAllocSize(record->getElementType(i));
  }
  pad(dl.getTypeAllocSize(record));
  return json::Value(json::Array{di->getName().str(), std::move(entries), "struct_packed"});
}

struct Field {
  DIDerivedType *debug;
  IntegerType *type;
  unsigned index;
  bool isSigned;
};

struct Output {
  std::string name, program, record, map;
  int64_t fd = 0;
  bool mpmo = false;
  Function *entry = nullptr;
  StructType *original = nullptr, *compact = nullptr;
  DICompositeType *compactDebug = nullptr;
  SmallVector<Field> fields;
  uint64_t originalSize = 0, size = 0, dataOffset = 0, mapSize = 0;
};

Error rewriteMapGlobal(Module &module, StringRef name, StructType *oldLeaf,
                       StructType *newLeaf, DICompositeType *newLeafDebug,
                       DIBuilder &dib) {
  auto *global = module.getNamedGlobal(name);
  auto *oldType = global ? dyn_cast<StructType>(global->getValueType()) : nullptr;
  if (!oldType || oldType->getNumElements() < 2 ||
      oldType->getElementType(1) != oldLeaf || !global->hasInitializer())
    return invalid("unexpected BCC map global: " + name);
  // BCC rewrites accesses to llvm.bpf.pseudo before this hook. A residual access
  // to table_t would need its own relocation, and must not silently survive.
  for (User *user : global->users()) {
    auto *array = dyn_cast<ConstantArray>(user);
    if (!array)
      return invalid("unexpected live use of BCC map global: " + name);
    for (User *owner : array->users()) {
      auto *used = dyn_cast<GlobalVariable>(owner);
      if (!used || (used->getName() != "llvm.compiler.used" && used->getName() != "llvm.used"))
        return invalid("map global escapes compiler retention: " + name);
    }
  }
  SmallVector<DIGlobalVariableExpression *> expressions;
  global->getDebugInfo(expressions);
  if (expressions.size() != 1)
    return invalid("map global requires one debug declaration: " + name);
  auto *variable = expressions[0]->getVariable();
  auto *oldDebug = dyn_cast_or_null<DICompositeType>(unqualified(variable->getType()));
  auto fields = members(oldType, oldDebug, module.getDataLayout());
  if (!fields)
    return fields.takeError();
  SmallVector<Type *> types(oldType->elements());
  types[1] = newLeaf;
  auto *newType = StructType::create(module.getContext(), types,
                                    (oldType->getName() + ".selected").str());
  SmallVector<DIType *> bases;
  SmallVector<Constant *> values;
  for (unsigned i = 0; i < types.size(); ++i) {
    auto *value = global->getInitializer()->getAggregateElement(i);
    if (!value || (i == 1 && !value->isNullValue()))
      return invalid("unexpected initialized BCC leaf: " + name);
    values.push_back(i == 1 ? Constant::getNullValue(newLeaf) : value);
    bases.push_back(i == 1 ? newLeafDebug : (*fields)[i]->getBaseType());
  }
  auto *newDebug = cloneDebug(dib, oldDebug, newType, *fields, bases, module.getDataLayout());
  auto *replacement = new GlobalVariable(
      module, newType, global->isConstant(), global->getLinkage(),
      ConstantStruct::get(newType, values), "", global, global->getThreadLocalMode(),
      global->getAddressSpace(), global->isExternallyInitialized());
  replacement->copyAttributesFrom(global);
  replacement->takeName(global);
  // Change only this variable's declaration, including its CU reference. Do
  // not add a second ____btf_map_NAME definition or mutate the shared leaf DI.
  variable->replaceOperandWith(3, newDebug); // LLVM 18 DIVariable::getRawType().
  replacement->setMetadata(LLVMContext::MD_dbg, expressions[0]);
  global->replaceAllUsesWith(replacement);
  global->eraseFromParent();
  return Error::success();
}

Error capture(Module &module, Output &output) {
  const DataLayout &dl = module.getDataLayout();
  SmallVector<CallInst *> submits;
  for (Function &function : module)
    for (Instruction &instruction : instructions(function)) {
      auto *call = dyn_cast<CallInst>(&instruction);
      if (!call || helperID(*call) != 25)
        continue;
      if (call->arg_size() != 5)
        return invalid("unexpected perf_event_output signature");
      auto fd = mapFD(call->getArgOperand(1));
      if (&function != output.entry && fd == output.fd)
        return invalid("selected capture map is used by another program");
      if (&function != output.entry)
        continue;
      if (fd != output.fd)
        return invalid("every output call in the selected program must match its map");
      submits.push_back(call);
    }
  if (submits.empty())
    return invalid("no capture output in " + output.program);
  for (CallInst *submit : submits) {
    int64_t offset = 0;
    auto *storage = dyn_cast<AllocaInst>(GetPointerBaseWithConstantOffset(
        submit->getArgOperand(3), offset, dl));
    auto *size = dyn_cast<ConstantInt>(submit->getArgOperand(4));
    auto *count = storage ? dyn_cast<ConstantInt>(storage->getArraySize()) : nullptr;
    if (!storage || offset || storage->getAllocatedType() != output.original ||
        !count || !count->isOne() || !size || size->getValue().getActiveBits() > 64 ||
        size->getZExtValue() != output.originalSize)
      return invalid("capture must submit exactly one original stack record");
    if (output.originalSize + output.size > 512)
      return invalid("capture projection exceeds conservative stack budget");
    IRBuilder<> allocate(&*output.entry->getEntryBlock().getFirstInsertionPt());
    auto *payload = allocate.CreateAlloca(output.compact, nullptr, "selected_record");
    payload->setAlignment(dl.getABITypeAlign(output.compact));
    IRBuilder<> emit(submit);
    emit.CreateMemSet(payload, emit.getInt8(0), output.size, payload->getAlign());
    for (unsigned i = 0; i < output.fields.size(); ++i) {
      const Field &field = output.fields[i];
      Value *from = emit.CreateStructGEP(output.original, storage, field.index);
      Value *value = emit.CreateAlignedLoad(field.type, from, Align(1));
      emit.CreateStore(value, emit.CreateStructGEP(output.compact, payload, i));
    }
    submit->setArgOperand(3, payload);
    submit->removeParamAttr(3, Attribute::Dereferenceable);
    submit->removeParamAttr(3, Attribute::DereferenceableOrNull);
    submit->removeParamAttr(3, Attribute::Alignment);
    submit->setArgOperand(4, ConstantInt::get(size->getType(), output.size));
  }
  return Error::success();
}

Error mpmo(Module &module, Output &output, json::Object &map, DIBuilder &dib,
           const DebugInfoFinder &finder) {
  const DataLayout &dl = module.getDataLayout();
  std::string slotName = "kermit_slot_" + output.name;
  auto *slot = StructType::getTypeByName(module.getContext(), "struct." + slotName);
  auto slotDebug = debugRecord(finder, slotName);
  if (!slotDebug)
    return slotDebug.takeError();
  auto slotMembers = members(slot, *slotDebug, dl);
  if (!slotMembers)
    return slotMembers.takeError();
  unsigned dataIndex = slotMembers->size() - 1;
  if (slotMembers->size() < 2 || (*slotMembers)[0]->getName() != "valid" ||
      !slot->getElementType(0)->isIntegerTy(64) ||
      (*slotMembers)[dataIndex]->getName() != "data" ||
      slot->getElementType(dataIndex) != output.original ||
      map.getInteger("value_size") != int64_t(dl.getTypeAllocSize(slot)))
    return invalid("invalid original KERMIT slot layout/metadata");
  output.dataOffset = dl.getStructLayout(slot)->getElementOffset(dataIndex);
  SmallVector<Type *> types(slot->elements());
  SmallVector<DIType *> bases;
  for (auto *member : *slotMembers)
    bases.push_back(member->getBaseType());
  types[dataIndex] = output.compact;
  bases[dataIndex] = output.compactDebug;
  auto *compactSlot = StructType::create(module.getContext(), types,
                                        (slot->getName() + ".selected").str());
  if (dl.getStructLayout(compactSlot)->getElementOffset(dataIndex) != output.dataOffset)
    return invalid("compaction would move the slot prefix");
  output.mapSize = dl.getTypeAllocSize(compactSlot);
  auto *compactSlotDebug = cloneDebug(dib, *slotDebug, compactSlot, *slotMembers, bases, dl);

  DenseMap<Value *, int64_t> offsets;
  for (Function &function : module)
    for (Instruction &instruction : instructions(function)) {
      auto *call = dyn_cast<CallInst>(&instruction);
      if (!call)
        continue;
      if (&function == output.entry && helperID(*call) == 25)
        return invalid("capture output in an MPMO program");
      if (!call->arg_size() || mapFD(call->getArgOperand(0)) != output.fd)
        continue;
      if (&function != output.entry || helperID(*call) != 1 || call->arg_size() != 2)
        return invalid("output MPMO map has an unsupported user");
      offsets[call] = 0;
    }
  if (offsets.empty())
    return invalid("no lookup of selected MPMO output map");

  // Follow provenance, not the record type: RSS output and bridge have the same
  // source type. Null PHI arms are the original reserve-failure path.
  bool changed;
  do {
    changed = false;
    for (Instruction &instruction : instructions(output.entry)) {
      if (!instruction.getType()->isPointerTy() || offsets.count(&instruction))
        continue;
      std::optional<int64_t> offset;
      if (auto *gep = dyn_cast<GetElementPtrInst>(&instruction)) {
        auto base = offsets.find(gep->getPointerOperand());
        if (base == offsets.end())
          continue;
        APInt delta(dl.getIndexTypeSizeInBits(gep->getType()), 0);
        int64_t sum;
        if (!gep->accumulateConstantOffset(dl, delta) || !delta.isSignedIntN(64) ||
            __builtin_add_overflow(base->second, delta.getSExtValue(), &sum))
          return invalid("nonconstant/overflowing output pointer offset");
        offset = sum;
      } else if (auto *cast = dyn_cast<BitCastInst>(&instruction)) {
        auto base = offsets.find(cast->getOperand(0));
        if (base != offsets.end())
          offset = base->second;
      } else if (isa<PHINode>(instruction) || isa<SelectInst>(instruction)) {
        unsigned begin = isa<SelectInst>(instruction) ? 1 : 0;
        for (unsigned i = begin; i < instruction.getNumOperands(); ++i) {
          Value *incoming = instruction.getOperand(i);
          if (isa<ConstantPointerNull>(incoming))
            continue;
          auto base = offsets.find(incoming);
          // Infer through loop-carried PHIs, then validate every incoming arm
          // after the fixed point. Unknown arms are not accepted as provenance.
          if (base == offsets.end())
            continue;
          if (offset && *offset != base->second)
            return invalid("mixed output pointer offsets in PHI/select");
          offset = base->second;
        }
        if (offset && *offset != 0 && *offset != int64_t(output.dataOffset))
          return invalid("PHI/select of an interior output field is unsupported");
      }
      if (offset) {
        offsets[&instruction] = *offset;
        changed = true;
      }
    }
  } while (changed);

  for (auto item : offsets) {
    auto *instruction = dyn_cast<Instruction>(item.first);
    if (!instruction || (!isa<PHINode>(instruction) && !isa<SelectInst>(instruction)))
      continue;
    unsigned begin = isa<SelectInst>(instruction) ? 1 : 0;
    for (unsigned i = begin; i < instruction->getNumOperands(); ++i) {
      Value *incoming = instruction->getOperand(i);
      if (isa<ConstantPointerNull>(incoming))
        continue;
      auto base = offsets.find(incoming);
      if (base == offsets.end() || base->second != item.second)
        return invalid("mixed output pointer provenance in PHI/select");
    }
  }

  SmallVector<StoreInst *> remove;
  struct Relocation { StoreInst *store; Value *anchor; int64_t delta; };
  SmallVector<Relocation> relocations;
  unsigned payloadStores = 0;
  for (auto item : offsets) {
    Value *pointer = item.first;
    int64_t offset = item.second;
    for (User *user : pointer->users()) {
      if (offsets.count(user))
        continue;
      if (auto *compare = dyn_cast<ICmpInst>(user)) {
        if (compare->isEquality() &&
            (isa<ConstantPointerNull>(compare->getOperand(0)) ||
             isa<ConstantPointerNull>(compare->getOperand(1))))
          continue;
      }
      if (isa<DbgInfoIntrinsic>(user))
        continue;
      auto *store = dyn_cast<StoreInst>(user);
      auto *load = dyn_cast<LoadInst>(user);
      auto *rmw = dyn_cast<AtomicRMWInst>(user);
      auto *cas = dyn_cast<AtomicCmpXchgInst>(user);
      Type *accessType = nullptr;
      if (store && store->getPointerOperand() == pointer && store->getValueOperand() != pointer)
        accessType = store->getValueOperand()->getType();
      else if (load && load->getPointerOperand() == pointer)
        accessType = load->getType();
      else if (rmw && rmw->getPointerOperand() == pointer)
        accessType = rmw->getValOperand()->getType();
      else if (cas && cas->getPointerOperand() == pointer)
        accessType = cas->getCompareOperand()->getType();
      if (!accessType || !accessType->isIntegerTy())
        return invalid("unrecognized output pointer use (escape or mixed provenance)");
      uint64_t width = dl.getTypeStoreSize(accessType);
      if (offset < 0 || uint64_t(offset) + width > dl.getTypeAllocSize(slot))
        return invalid("out-of-bounds output slot access");
      if (uint64_t(offset) + width <= output.dataOffset)
        continue; // valid and optional CPU metadata, including all atomics.
      if (!store || store->isVolatile() || store->isAtomic() ||
          uint64_t(offset) < output.dataOffset)
        return invalid("output payload is not ordinary write-only scalar storage");
      uint64_t fieldOffset = offset - output.dataOffset;
      if (fieldOffset >= output.originalSize)
        return invalid("store into output slot tail padding");
      const StructLayout *oldLayout = dl.getStructLayout(output.original);
      unsigned index = oldLayout->getElementContainingOffset(fieldOffset);
      if (oldLayout->getElementOffset(index) != fieldOffset ||
          output.original->getElementType(index) != accessType)
        return invalid("output store does not exactly match an original field");
      ++payloadStores;
      auto kept = llvm::find_if(output.fields, [index](const Field &f) { return f.index == index; });
      if (kept == output.fields.end()) {
        remove.push_back(store);
        continue;
      }
      unsigned newIndex = kept - output.fields.begin();
      uint64_t newOffset = output.dataOffset + dl.getStructLayout(output.compact)->getElementOffset(newIndex);
      int64_t delta = 0;
      Value *anchor = GetPointerBaseWithConstantOffset(pointer, delta, dl);
      auto base = offsets.find(anchor);
      if (base == offsets.end() ||
          (base->second != 0 && base->second != int64_t(output.dataOffset)))
        return invalid("cannot anchor relocated output store");
      relocations.push_back({store, anchor, int64_t(newOffset) - base->second});
    }
  }
  if (!payloadStores)
    return invalid("no scalar stores into the selected output payload");
  for (StoreInst *store : remove)
    store->eraseFromParent();
  for (const auto &relocation : relocations) {
    IRBuilder<> builder(relocation.store);
    Value *to = builder.CreateGEP(builder.getInt8Ty(), relocation.anchor,
                                  builder.getInt64(relocation.delta), "selected_field");
    relocation.store->setOperand(1, to);
    relocation.store->setMetadata(LLVMContext::MD_tbaa, nullptr);
    relocation.store->setAlignment(dl.getABITypeAlign(relocation.store->getValueOperand()->getType()));
  }

  if (auto error = rewriteMapGlobal(module, output.map, slot, compactSlot, compactSlotDebug, dib))
    return error;
  if (auto error = rewriteMapGlobal(module, "____btf_map_" + output.map,
                                    slot, compactSlot, compactSlotDebug, dib))
    return error;
  auto desc = descriptor(compactSlot, compactSlotDebug, dl);
  if (!desc)
    return desc.takeError();
  map["value_size"] = int64_t(output.mapSize);
  map["value_desc"] = formatv("{0}", *desc).str();
  return Error::success();
}

bool overlaps(int64_t offset, uint64_t size, uint64_t start, uint64_t length) {
  return offset < 0 || size > UINT64_MAX - uint64_t(offset) ||
         (uint64_t(offset) < start + length && start < uint64_t(offset) + size);
}

bool readOrEscape(Value *pointer, int64_t offset, uint64_t start,
                  uint64_t length, const DataLayout &dl) {
  for (User *user : pointer->users()) {
    if (auto *gep = dyn_cast<GetElementPtrInst>(user)) {
      APInt delta(dl.getIndexTypeSizeInBits(gep->getType()), 0);
      int64_t next;
      if (!gep->accumulateConstantOffset(dl, delta) || !delta.isSignedIntN(64) ||
          __builtin_add_overflow(offset, delta.getSExtValue(), &next) ||
          readOrEscape(gep, next, start, length, dl))
        return true;
      continue;
    }
    if (auto *cast = dyn_cast<BitCastInst>(user)) {
      if (readOrEscape(cast, offset, start, length, dl))
        return true;
      continue;
    }
    if (auto *load = dyn_cast<LoadInst>(user)) {
      if (overlaps(offset, dl.getTypeStoreSize(load->getType()), start, length))
        return true;
      continue;
    }
    if (auto *store = dyn_cast<StoreInst>(user)) {
      if (store->getValueOperand() == pointer)
        return true;
      continue;
    }
    if (auto *set = dyn_cast<MemSetInst>(user)) {
      if (set->getRawDest() != pointer || set->isVolatile())
        return true;
      continue;
    }
    if (auto *copy = dyn_cast<MemTransferInst>(user)) {
      auto *size = dyn_cast<ConstantInt>(copy->getLength());
      if (copy->isVolatile() || (copy->getRawSource() == pointer &&
          (!size || overlaps(offset, size->getZExtValue(), start, length))))
        return true;
      continue;
    }
    if (auto *intrinsic = dyn_cast<IntrinsicInst>(user))
      if (isa<DbgInfoIntrinsic>(intrinsic) ||
          intrinsic->getIntrinsicID() == Intrinsic::lifetime_start ||
          intrinsic->getIntrinsicID() == Intrinsic::lifetime_end)
        continue;
    if (auto *call = dyn_cast<CallBase>(user))
      if ((helperID(*call) == 4 || helperID(*call) == 113) && call->arg_size() == 3 &&
          call->getArgOperand(0) == pointer && call->getArgOperand(2) != pointer)
        continue;
    return true;
  }
  return false;
}

bool pruneHelpers(Module &module) {
  SmallVector<CallInst *> dead;
  const DataLayout &dl = module.getDataLayout();
  for (Function &function : module)
    for (Instruction &instruction : instructions(function)) {
      auto *call = dyn_cast<CallInst>(&instruction);
      if (!call || !call->use_empty())
        continue;
      uint64_t id = helperID(*call);
      if (id == 5 || id == 8 || id == 14 || id == 15) {
        dead.push_back(call);
      } else if ((id == 4 || id == 113) && call->arg_size() == 3) {
        auto *size = dyn_cast<ConstantInt>(call->getArgOperand(1));
        int64_t offset = 0;
        auto *stack = dyn_cast<AllocaInst>(GetPointerBaseWithConstantOffset(
            call->getArgOperand(0), offset, dl));
        if (!stack || !size || offset < 0)
          continue;
        auto allocation = stack->getAllocationSize(dl);
        if (!allocation || allocation->isScalable() ||
            uint64_t(offset) > allocation->getFixedValue() ||
            size->getZExtValue() > allocation->getFixedValue() - offset)
          continue;
        if (!readOrEscape(stack, 0, offset, size->getZExtValue(), dl))
          dead.push_back(call);
      }
    }
  for (CallInst *call : dead)
    call->eraseFromParent();
  return !dead.empty();
}
} // namespace

llvm::Expected<llvm::json::Object> reduceOriginal(
    llvm::Module &module, const llvm::json::Object &metadata,
    const llvm::json::Object &selection) {
  using namespace llvm;
  if (!Triple(module.getTargetTriple()).isBPF() || verifyModule(module, &errs()))
    return invalid("a valid retargeted original BCC module is required");
  auto *outputs = selection.getObject("outputs");
  if (selection.getInteger("version") != 1 || selection.size() != 2 ||
      !outputs || outputs->empty() || metadata.getInteger("version") != 1 ||
      !metadata.getArray("maps") || !metadata.getObject("perf_events"))
    return invalid("expected version 1 selection outputs and BCC metadata");
  if (module.debug_compile_units().empty())
    return invalid("full original frontend debug information is required");
  json::Object result = metadata;
  auto &maps = *result.getArray("maps");
  StringSet<> mapNames, programs;
  std::map<int64_t, std::string> fds;
  for (const auto &value : maps) {
    auto *map = value.getAsObject();
    if (!map || !map->getString("name") || !map->getInteger("fake_fd") ||
        *map->getInteger("fake_fd") >= 0 ||
        *map->getInteger("fake_fd") < std::numeric_limits<int>::min() ||
        !map->getInteger("value_size") ||
        *map->getInteger("value_size") <= 0 || !map->getString("value_desc") ||
        !mapNames.insert(*map->getString("name")).second ||
        !fds.emplace(*map->getInteger("fake_fd"), map->getString("name")->str()).second)
      return invalid("invalid/duplicate map metadata or nonnegative fake fd");
  }
  // Sort output names: json::Object iteration order is not a schema contract.
  std::vector<std::string> names;
  for (const auto &item : *outputs)
    names.push_back(item.first.str());
  llvm::sort(names);
  if (auto error = optimize(module, true))
    return std::move(error);
  DebugInfoFinder finder;
  finder.processModule(module);
  DIBuilder dib(module, true, *module.debug_compile_units().begin());
  json::Object schemas;
  std::optional<bool> transport;
  for (const std::string &name : names) {
    auto *request = outputs->getObject(name);
    if (!request || request->size() != 3 || !request->getString("program") ||
        !request->getString("record") || !request->getArray("fields"))
      return invalid("each output requires program, record, and fields");
    Output output;
    output.name = name;
    output.program = request->getString("program")->str();
    output.record = request->getString("record")->str();
    output.entry = module.getFunction(output.program);
    if (!output.entry || output.entry->isDeclaration() ||
        !output.entry->getSection().starts_with(".bpf.fn.") ||
        !programs.insert(output.program).second)
      return invalid("output must name a unique original BCC entry point");
    output.mpmo = mapNames.count("mpmo_buff_" + name);
    output.map = output.mpmo ? "mpmo_buff_" + name : name;
    if (transport && *transport != output.mpmo)
      return invalid("mixed output transports are unsupported");
    transport = output.mpmo;
    json::Object *map = nullptr;
    for (auto &value : maps)
      if (value.getAsObject()->getString("name") == output.map)
        map = value.getAsObject();
    if (!map || (output.mpmo && mapNames.count(name)) ||
        map->getInteger("type") != (output.mpmo ? 2 : 4) ||
        (output.mpmo && (map->getInteger("max_entries") != 32 ||
                        map->getInteger("key_size") != 4)))
      return invalid("missing or incompatible original output map: " + output.map);
    output.fd = *map->getInteger("fake_fd");
    output.mapSize = *map->getInteger("value_size");
    output.original = StructType::getTypeByName(module.getContext(), "struct." + output.record);
    auto debug = debugRecord(finder, output.record);
    if (!debug)
      return debug.takeError();
    auto originalFields = members(output.original, *debug, module.getDataLayout());
    if (!originalFields)
      return originalFields.takeError();
    StringSet<> wanted;
    for (const auto &field : *request->getArray("fields")) {
      auto value = field.getAsString();
      if (!value || value->empty() || !wanted.insert(*value).second)
        return invalid("field names must be nonempty unique strings");
    }
    if (wanted.empty())
      return invalid("select at least one field");
    const DataLayout &dl = module.getDataLayout();
    for (unsigned i = 0; i < originalFields->size(); ++i) {
      auto *member = (*originalFields)[i];
      auto *integer = dyn_cast<IntegerType>(output.original->getElementType(i));
      auto *basic = dyn_cast_or_null<DIBasicType>(unqualified(member->getBaseType()));
      if (!integer || !basic || basic->getSizeInBits() != integer->getBitWidth() ||
          (integer->getBitWidth() != 8 && integer->getBitWidth() != 16 &&
           integer->getBitWidth() != 32 && integer->getBitWidth() != 64))
        return invalid("only 8/16/32/64-bit integer output fields are supported");
      unsigned encoding = basic->getEncoding();
      bool isSigned = encoding == dwarf::DW_ATE_signed || encoding == dwarf::DW_ATE_signed_char;
      if (!isSigned && encoding != dwarf::DW_ATE_unsigned && encoding != dwarf::DW_ATE_unsigned_char)
        return invalid("unsupported output field encoding");
      if (wanted.erase(member->getName()))
        output.fields.push_back({member, integer, i, isSigned});
    }
    if (!wanted.empty())
      return invalid("unknown selected field: " + wanted.begin()->getKey());
    std::stable_sort(output.fields.begin(), output.fields.end(), [&](const Field &a, const Field &b) {
      return dl.getABITypeAlign(a.type) > dl.getABITypeAlign(b.type);
    });
    SmallVector<Type *> types;
    SmallVector<DIDerivedType *> fields;
    SmallVector<DIType *> bases;
    for (const Field &field : output.fields) {
      types.push_back(field.type);
      fields.push_back(field.debug);
      bases.push_back(field.debug->getBaseType());
    }
    output.compact = StructType::create(module.getContext(), types,
        "struct." + output.record + ".selected." + name);
    output.compactDebug = cloneDebug(dib, *debug, output.compact, fields, bases, dl);
    output.originalSize = dl.getTypeAllocSize(output.original);
    output.size = dl.getTypeAllocSize(output.compact);
    if (output.mpmo) {
      if (auto error = mpmo(module, output, *map, dib, finder))
        return std::move(error);
    } else {
      if (auto error = capture(module, output))
        return std::move(error);
      auto *oldEvents = result.getObject("perf_events")->getArray(name);
      if (!oldEvents)
        return invalid("missing BCC capture event metadata");
      json::Array events;
      for (const Field &field : output.fields) {
        std::optional<StringRef> match;
        for (const auto &event : *oldEvents) {
          auto text = event.getAsString();
          if (text && text->split('#').first == field.debug->getName()) {
            if (match)
              return invalid("duplicate BCC event field");
            match = *text;
          }
        }
        if (!match)
          return invalid("selected field missing from BCC event metadata");
        events.push_back(match->str());
      }
      (*result.getObject("perf_events"))[name] = std::move(events);
    }
    json::Array schemaFields;
    for (unsigned i = 0; i < output.fields.size(); ++i) {
      const auto &field = output.fields[i];
      schemaFields.push_back(json::Object{
          {"name", field.debug->getName().str()},
          {"offset", int64_t(dl.getStructLayout(output.compact)->getElementOffset(i))},
          {"size", int64_t(dl.getTypeStoreSize(field.type))}, {"signed", field.isSigned}});
    }
    schemas[name] = json::Object{
        {"source_event", output.record}, {"program", output.program}, {"map", output.map},
        {"size", int64_t(output.size)}, {"original_size", int64_t(output.originalSize)},
        {"data_offset", int64_t(output.dataOffset)}, {"map_value_size", int64_t(output.mapSize)},
        {"slots", output.mpmo ? json::Value(32) : json::Value(nullptr)},
        {"byte_order", dl.isLittleEndian() ? "little" : "big"},
        {"fields", std::move(schemaFields)}};
  }
  dib.finalize();
  for (Function &function : module)
    for (Instruction &instruction : instructions(function))
      if (auto *call = dyn_cast<CallBase>(&instruction))
        if ((helperID(*call) == 4 || helperID(*call) == 113) && call->arg_size() == 3) {
          call->setMemoryEffects(MemoryEffects::argMemOnly(ModRefInfo::ModRef));
          call->addParamAttr(0, Attribute::NoCapture);
          call->addParamAttr(0, Attribute::WriteOnly);
          call->addParamAttr(2, Attribute::NoCapture);
          call->addParamAttr(2, Attribute::ReadOnly);
        }
  do {
    if (auto error = optimize(module))
      return std::move(error);
  } while (pruneHelpers(module));
  if (verifyModule(module, &errs()))
    return invalid("reduction produced invalid LLVM IR");
  result["version"] = 1;
  result["llvm_version"] = LLVM_VERSION_STRING;
  result["transport"] = *transport ? "mpmo" : "capture";
  result["schemas"] = std::move(schemas);
  return result;
}

llvm::Expected<llvm::json::Object> readOriginalJSON(llvm::StringRef path) {
  auto buffer = llvm::MemoryBuffer::getFile(path);
  if (!buffer)
    return invalid(path + ": " + buffer.getError().message());
  auto value = llvm::json::parse((*buffer)->getBuffer());
  if (!value)
    return value.takeError();
  auto *object = value->getAsObject();
  if (!object)
    return invalid("expected a JSON object: " + path);
  return std::move(*object);
}

llvm::Error writeOriginalEvidence(llvm::Module &module,
                                 const llvm::json::Object &metadata,
                                 llvm::StringRef prefix, llvm::StringRef stage) {
  using namespace llvm;
  for (StringRef suffix : {"bc", "ll", "json"}) {
    std::string path = (prefix + "." + stage + "." + suffix).str();
    int fd = -1;
    if (auto error = sys::fs::openFileForWrite(path, fd, sys::fs::CD_CreateNew))
      return invalid(path + ": " + error.message());
    raw_fd_ostream stream(fd, true);
    if (suffix == "bc")
      WriteBitcodeToFile(module, stream);
    else if (suffix == "ll")
      module.print(stream, nullptr);
    else
      stream << formatv("{0:2}\n", json::Value(json::Object(metadata)));
    stream.flush();
    if (stream.has_error())
      return invalid("cannot write " + path);
  }
  return Error::success();
}
} // namespace kernmlops

#ifdef KERNMLOPS_ORIGINAL_REDUCE_MAIN
#include <llvm/IRReader/IRReader.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/InitLLVM.h>
#include <llvm/Support/SourceMgr.h>

int main(int argc, char **argv) {
  using namespace llvm;
  InitLLVM init(argc, argv);
  cl::opt<std::string> input("input", cl::Required, cl::desc("Original BCC bitcode/IR"));
  cl::opt<std::string> metadata("metadata", cl::Required, cl::desc("Original BCC metadata JSON"));
  cl::opt<std::string> selection("selection", cl::Required, cl::desc("Version 1 field selection JSON"));
  cl::opt<std::string> output("output", cl::Required, cl::desc("Exclusive-create evidence prefix"));
  cl::ParseCommandLineOptions(argc, argv, "Reduce ORIGINAL BCC/KERMIT IR offline\n");
  auto fail = [](Error error) {
    errs() << "original-reduce: " << toString(std::move(error)) << '\n';
    return 1;
  };
  LLVMContext context;
  SMDiagnostic diagnostic;
  auto module = parseIRFile(input, diagnostic, context);
  if (!module) {
    diagnostic.print(argv[0], errs());
    return 1;
  }
  auto before = kernmlops::readOriginalJSON(metadata);
  if (!before)
    return fail(before.takeError());
  if (auto error = kernmlops::writeOriginalEvidence(*module, *before, output, "before"))
    return fail(std::move(error));
  auto wanted = kernmlops::readOriginalJSON(selection);
  if (!wanted)
    return fail(wanted.takeError());
  auto after = kernmlops::reduceOriginal(*module, *before, *wanted);
  if (!after)
    return fail(after.takeError());
  if (auto error = kernmlops::writeOriginalEvidence(*module, *after, output, "after"))
    return fail(std::move(error));
  return 0;
}
#endif
