// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/TargetParser/Triple.h"

#include <cstdint>
#include <string>
#include <vector>

using namespace llvm;

static cl::opt<std::string> Input("input", cl::Required, cl::desc("Input BPF bitcode with full debug types"));
static cl::opt<std::string> Output("output", cl::Required, cl::desc("Rewritten BPF bitcode"));
static cl::opt<std::string> Schema("schema", cl::Required, cl::desc("Compact event layout JSON"));
static cl::opt<std::string> Program("program", cl::Required, cl::desc("Program entry point"));
static cl::opt<std::string> Event("event", cl::Required, cl::desc("Source event struct name"));
static cl::opt<std::string> Fields("fields", cl::Required, cl::desc("Comma-separated field names"));
static cl::opt<bool> PreserveOtherPrograms(
    "preserve-other-programs", cl::init(false),
    cl::desc("Keep companion entry points and their output layouts unchanged"));

static int error(const Twine &Message) {
  errs() << "field-extract: " << Message << '\n';
  return 1;
}

static uint64_t helperID(const CallBase &Call) {
  auto *Callee = dyn_cast<ConstantExpr>(Call.getCalledOperand());
  if (!Callee || Callee->getOpcode() != Instruction::IntToPtr)
    return 0;
  auto *ID = dyn_cast<ConstantInt>(Callee->getOperand(0));
  return ID && ID->getValue().getActiveBits() <= 64 ? ID->getZExtValue() : 0;
}

static bool overlaps(int64_t Offset, uint64_t Size, uint64_t Start,
                     uint64_t Length) {
  if (Offset < 0 || Size > UINT64_MAX - uint64_t(Offset))
    return true;
  return uint64_t(Offset) < Start + Length && Start < uint64_t(Offset) + Size;
}

// Check a particular stack byte range, not the entire escaped aggregate.
// Unrecognized pointer uses are barriers, including PHIs and pointer-to-int casts.
static bool readOrEscape(Value *Pointer, int64_t Offset, uint64_t Start,
                         uint64_t Length, const DataLayout &DL) {
  for (User *U : Pointer->users()) {
    if (auto *GEP = dyn_cast<GetElementPtrInst>(U)) {
      APInt Delta(DL.getIndexTypeSizeInBits(GEP->getType()), 0);
      if (!GEP->accumulateConstantOffset(DL, Delta) ||
          !Delta.isSignedIntN(64))
        return true;
      int64_t Next;
      if (__builtin_add_overflow(Offset, Delta.getSExtValue(), &Next) ||
          readOrEscape(GEP, Next, Start, Length, DL))
        return true;
      continue;
    }
    if (auto *Cast = dyn_cast<BitCastInst>(U)) {
      if (readOrEscape(Cast, Offset, Start, Length, DL))
        return true;
      continue;
    }
    if (auto *Load = dyn_cast<LoadInst>(U)) {
      TypeSize Size = DL.getTypeStoreSize(Load->getType());
      if (Size.isScalable() || overlaps(Offset, Size.getFixedValue(), Start, Length))
        return true;
      continue;
    }
    if (auto *Store = dyn_cast<StoreInst>(U)) {
      if (Store->getValueOperand() == Pointer)
        return true;
      continue;
    }
    if (auto *Set = dyn_cast<MemSetInst>(U)) {
      if (Set->getRawDest() != Pointer || Set->isVolatile())
        return true;
      continue;
    }
    if (auto *Copy = dyn_cast<MemTransferInst>(U)) {
      auto *Size = dyn_cast<ConstantInt>(Copy->getLength());
      if (Copy->isVolatile() ||
          (Copy->getRawSource() == Pointer &&
           (!Size || Size->getValue().getActiveBits() > 64 ||
            overlaps(Offset, Size->getZExtValue(), Start, Length))))
        return true;
      continue;
    }
    if (auto *Intrinsic = dyn_cast<IntrinsicInst>(U)) {
      if (isa<DbgInfoIntrinsic>(Intrinsic) ||
          Intrinsic->getIntrinsicID() == Intrinsic::lifetime_start ||
          Intrinsic->getIntrinsicID() == Intrinsic::lifetime_end)
        continue;
    }
    if (auto *Call = dyn_cast<CallBase>(U)) {
      // probe_read_kernel writes its destination but reads its source. Its
      // status can remain live independently of the copied bytes.
      if (helperID(*Call) == 113 && Call->arg_size() == 3 &&
          Call->getArgOperand(0) == Pointer &&
          Call->getArgOperand(2) != Pointer)
        continue;
    }
    return true;
  }
  return false;
}

static bool removeUnusedHelpers(Module &M) {
  const DataLayout &DL = M.getDataLayout();
  SmallVector<CallInst *> Dead;
  for (Function &F : M) {
    for (Instruction &I : instructions(F)) {
      auto *Call = dyn_cast<CallInst>(&I);
      if (!Call || !Call->use_empty())
        continue;
      switch (helperID(*Call)) {
      case 5:  // ktime_get_ns
      case 8:  // get_smp_processor_id
      case 14: // get_current_pid_tgid
      case 15: // get_current_uid_gid
        Dead.push_back(Call);
        break;
      case 113: { // probe_read_kernel
        if (Call->arg_size() != 3)
          break;
        auto *Size = dyn_cast<ConstantInt>(Call->getArgOperand(1));
        int64_t Offset = 0;
        auto *Stack = dyn_cast<AllocaInst>(GetPointerBaseWithConstantOffset(
            Call->getArgOperand(0), Offset, DL));
        if (!Size || Size->getValue().getActiveBits() > 64 || !Stack || Offset < 0)
          break;
        auto Allocation = Stack->getAllocationSize(DL);
        if (!Allocation || Allocation->isScalable() ||
            uint64_t(Offset) > Allocation->getFixedValue() ||
            Size->getZExtValue() > Allocation->getFixedValue() - Offset)
          break;
        if (!readOrEscape(Stack, 0, Offset, Size->getZExtValue(), DL))
          Dead.push_back(Call);
        break;
      }
      default:
        break;
      }
    }
  }
  for (CallInst *Call : Dead)
    Call->eraseFromParent();
  return !Dead.empty();
}

static void describeReadHelpers(Module &M) {
  for (Function &F : M)
    for (Instruction &I : instructions(F))
      if (auto *Call = dyn_cast<CallBase>(&I))
        if (helperID(*Call) == 113 && Call->arg_size() == 3) {
          // The kernel read cannot retain stack pointers or read its destination.
          // Do not mark it memory(none): its source read and destination write
          // remain ordered against other memory operations.
          Call->setMemoryEffects(MemoryEffects::argMemOnly(ModRefInfo::ModRef));
          Call->addParamAttr(0, Attribute::NoCapture);
          Call->addParamAttr(0, Attribute::WriteOnly);
          Call->addParamAttr(2, Attribute::NoCapture);
          Call->addParamAttr(2, Attribute::ReadOnly);
        }
}

static void optimize(Module &M, bool Prepare = false) {
  LoopAnalysisManager LAM;
  FunctionAnalysisManager FAM;
  CGSCCAnalysisManager CGAM;
  ModuleAnalysisManager MAM;
  PassBuilder PB;
  PB.registerModuleAnalyses(MAM);
  PB.registerCGSCCAnalyses(CGAM);
  PB.registerFunctionAnalyses(FAM);
  PB.registerLoopAnalyses(LAM);
  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);
  ModulePassManager PM;
  cantFail(PB.parsePassPipeline(
      PM, Prepare ? "globalopt,always-inline,function(mem2reg,instcombine)"
                  : "always-inline,function(sroa,mem2reg,instcombine,simplifycfg,dse,adce,"
                    "loop(loop-deletion)),globaldce"));
  PM.run(M, MAM);
}

struct Field {
  DIDerivedType *Debug;
  unsigned SourceIndex;
  IntegerType *Type;
  bool Signed;
};

int main(int argc, char **argv) {
  InitLLVM Init(argc, argv);
  cl::ParseCommandLineOptions(argc, argv, "Compact a fixed-size BPF event and prune unused reads\n");
  if (Input == Output || Input == Schema || Output == Schema)
    return error("input, output, and schema paths must differ");

  LLVMContext Context;
  SMDiagnostic Diagnostic;
  std::unique_ptr<Module> M = parseIRFile(Input, Diagnostic, Context);
  if (!M) {
    Diagnostic.print(argv[0], errs());
    return 1;
  }
  Triple Target(M->getTargetTriple());
  if (!Target.isBPF())
    return error("input must be compiled for the BPF target");
  if (verifyModule(*M, &errs()))
    return error("invalid input LLVM IR");
  // Resolve libbpf's constant helper pointers and inline small wrappers, but
  // leave aggregate storage intact until output field identities are resolved.
  optimize(*M, true);
  Function *Entry = M->getFunction(Program);
  if (!Entry || Entry->isDeclaration() || !Entry->hasSection())
    return error("program must name a defined SEC() entry point");
  json::Array PreservedPrograms;
  for (const Function &F : *M)
    if (!F.isDeclaration() && &F != Entry && F.hasSection() &&
        !F.getSection().starts_with(".text")) {
      if (!PreserveOtherPrograms)
        return error("multiple entry points require --preserve-other-programs");
      PreservedPrograms.push_back(F.getName().str());
    }
  for (const Function &F : *M)
    if (&F != Entry && !(PreserveOtherPrograms && F.hasSection() &&
                        !F.getSection().starts_with(".text")))
      for (const Instruction &I : instructions(F))
        if (auto *Call = dyn_cast<CallBase>(&I))
          if (helperID(*Call) == 130)
            return error("output in a non-inlined subprogram is unsupported");

  auto *Original = StructType::getTypeByName(Context, "struct." + Event);
  if (!Original || Original->isOpaque() || Original->isPacked() || !Original->isSized())
    return error("event must name a non-packed, fixed-layout C struct");
  const DataLayout &DL = M->getDataLayout();
  const StructLayout *OldLayout = DL.getStructLayout(Original);

  DebugInfoFinder Debug;
  Debug.processModule(*M);
  DICompositeType *OriginalDebug = nullptr;
  for (DIType *Type : Debug.types()) {
    auto *Record = dyn_cast<DICompositeType>(Type);
    if (Record && Record->getTag() == dwarf::DW_TAG_structure_type &&
        Record->getName() == Event && !Record->isForwardDecl()) {
      if (OriginalDebug && OriginalDebug != Record)
        return error("ambiguous event debug type");
      OriginalDebug = Record;
    }
  }
  if (!OriginalDebug || OriginalDebug->getSizeInBits() != OldLayout->getSizeInBits())
    return error("matching full event debug types are required; compile with -g");
  if (!Entry->getSubprogram() || M->debug_compile_units().empty())
    return error("program debug information is required");

  SmallVector<StringRef> Requested;
  StringRef(Fields).split(Requested, ',', -1, true);
  StringSet<> Wanted;
  for (StringRef Name : Requested)
    if (Name.empty() || !Wanted.insert(Name).second)
      return error("field names must be nonempty and unique");

  std::vector<Field> Kept;
  SmallVector<Type *> Types;
  for (DINode *Node : OriginalDebug->getElements()) {
    auto *Member = dyn_cast<DIDerivedType>(Node);
    if (!Member || Member->getTag() != dwarf::DW_TAG_member ||
        Member->isBitField() || Member->getOffsetInBits() % 8)
      return error("only ordinary integer scalar event fields are supported");
    DIType *Base = Member->getBaseType();
    while (auto *Derived = dyn_cast_or_null<DIDerivedType>(Base)) {
      if (Derived->getTag() != dwarf::DW_TAG_typedef &&
          Derived->getTag() != dwarf::DW_TAG_const_type &&
          Derived->getTag() != dwarf::DW_TAG_volatile_type)
        break;
      Base = Derived->getBaseType();
    }
    auto *Basic = dyn_cast_or_null<DIBasicType>(Base);
    unsigned Encoding = Basic ? Basic->getEncoding() : 0;
    bool Signed = Encoding == dwarf::DW_ATE_signed || Encoding == dwarf::DW_ATE_signed_char;
    bool Unsigned = Encoding == dwarf::DW_ATE_unsigned || Encoding == dwarf::DW_ATE_unsigned_char;
    uint64_t Bits = Member->getSizeInBits();
    if ((!Signed && !Unsigned) ||
        (Bits != 8 && Bits != 16 && Bits != 32 && Bits != 64))
      return error("unsupported field '" + Member->getName() + "': expected an 8/16/32/64-bit integer");
    uint64_t Offset = Member->getOffsetInBits() / 8;
    if (Offset >= OldLayout->getSizeInBytes())
      return error("invalid source field offset");
    unsigned Index = OldLayout->getElementContainingOffset(Offset);
    auto *Integer = dyn_cast<IntegerType>(Original->getElementType(Index));
    if (OldLayout->getElementOffset(Index) != Offset || !Integer ||
        Integer->getBitWidth() != Bits)
      return error("event debug layout does not match its LLVM storage");
    if (Wanted.erase(Member->getName())) {
      Kept.push_back({Member, Index, Integer, Signed});
      Types.push_back(Integer);
    }
  }
  if (!Wanted.empty())
    return error("unknown event field '" + Wanted.begin()->getKey() + "'");
  if (Kept.empty())
    return error("select at least one field");

  CallInst *Submit = nullptr;
  AllocaInst *Storage = nullptr;
  for (Instruction &I : instructions(Entry)) {
    auto *Call = dyn_cast<CallInst>(&I);
    if (!Call || helperID(*Call) != 130)
      continue;
    if (Submit)
      return error("exactly one bpf_ringbuf_output call is supported");
    if (Call->arg_size() != 4)
      return error("unexpected bpf_ringbuf_output signature");
    int64_t Offset = 0;
    Storage = dyn_cast<AllocaInst>(GetPointerBaseWithConstantOffset(
        Call->getArgOperand(1), Offset, DL));
    auto *Size = dyn_cast<ConstantInt>(Call->getArgOperand(2));
    auto *Count = Storage ? dyn_cast<ConstantInt>(Storage->getArraySize()) : nullptr;
    if (!Storage || Offset != 0 || Storage->getAllocatedType() != Original ||
        !Count || !Count->isOne() || !Size || Size->getValue().getActiveBits() > 64 ||
        Size->getZExtValue() != OldLayout->getSizeInBytes())
      return error("output must submit exactly one stack-allocated event struct");
    Submit = Call;
  }
  if (!Submit)
    return error("no supported bpf_ringbuf_output call in the selected program");

  std::string Name = Event + "_selected";
  auto *Compact = StructType::create(Context, Types, "struct." + Name);
  const StructLayout *Layout = DL.getStructLayout(Compact);
  if (OldLayout->getSizeInBytes() > 512 ||
      Layout->getSizeInBytes() > 512 - OldLayout->getSizeInBytes())
    return error("source plus compact payload exceeds the conservative 512-byte stack budget");
  IRBuilder<> Allocate(&*Entry->getEntryBlock().getFirstInsertionPt());
  auto *Payload = Allocate.CreateAlloca(Compact, nullptr, "selected_event");
  Payload->setAlignment(DL.getABITypeAlign(Compact));
  IRBuilder<> Emit(Submit);
  // Padding is part of the helper's readable range and must be initialized.
  Emit.CreateMemSet(Payload, Emit.getInt8(0), Layout->getSizeInBytes(),
                    Payload->getAlign());
  for (unsigned I = 0; I < Kept.size(); ++I) {
    Value *From = Emit.CreateStructGEP(Original, Storage, Kept[I].SourceIndex);
    Value *FieldValue = Emit.CreateAlignedLoad(Kept[I].Type, From, Align(1));
    Emit.CreateStore(FieldValue, Emit.CreateStructGEP(Compact, Payload, I));
  }
  Submit->setArgOperand(1, Payload);
  Submit->setArgOperand(2, ConstantInt::get(Submit->getArgOperand(2)->getType(),
                                          Layout->getSizeInBytes()));

  // Generate a new debug type; never mutate kernel types used by CO-RE.
  DIBuilder DIB(*M, true, *M->debug_compile_units().begin());
  SmallVector<Metadata *> Members;
  json::Array SchemaFields;
  for (unsigned I = 0; I < Kept.size(); ++I) {
    auto &F = Kept[I];
    uint64_t Offset = Layout->getElementOffset(I);
    Members.push_back(DIB.createMemberType(
        OriginalDebug->getScope(), F.Debug->getName(), F.Debug->getFile(),
        F.Debug->getLine(), F.Type->getBitWidth(), DL.getABITypeAlign(F.Type).value() * 8,
        Offset * 8, DINode::FlagZero, F.Debug->getBaseType()));
    SchemaFields.push_back(json::Object{
        {"name", F.Debug->getName().str()}, {"offset", int64_t(Offset)},
        {"size", int64_t(F.Type->getBitWidth() / 8)}, {"signed", F.Signed}});
  }
  auto *NewDebug = DIB.createStructType(
      OriginalDebug->getScope(), Name, OriginalDebug->getFile(),
      OriginalDebug->getLine(), Layout->getSizeInBits(),
      Layout->getAlignment().value() * 8, DINode::FlagZero, nullptr,
      DIB.getOrCreateArray(Members));
  DIB.retainType(NewDebug);
  auto *Variable = DIB.createAutoVariable(Entry->getSubprogram(), "selected_event",
                                        OriginalDebug->getFile(), OriginalDebug->getLine(),
                                        NewDebug, true);
  DIB.insertDeclare(Payload, Variable, DIB.createExpression(),
                    DILocation::get(Context, OriginalDebug->getLine(), 0,
                                    Entry->getSubprogram()), Submit);
  DIB.finalize();

  describeReadHelpers(*M);
  do {
    optimize(*M);
  } while (removeUnusedHelpers(*M));
  if (verifyModule(*M, &errs()))
    return error("transformation produced invalid LLVM IR");

  json::Object Description{
      {"program", std::string(Program)}, {"source_event", std::string(Event)},
      {"event", Name}, {"original_size", int64_t(OldLayout->getSizeInBytes())},
      {"size", int64_t(Layout->getSizeInBytes())},
      {"alignment", int64_t(Layout->getAlignment().value())},
      {"byte_order", DL.isLittleEndian() ? "little" : "big"},
      {"preserved_programs", std::move(PreservedPrograms)},
      {"fields", std::move(SchemaFields)}};
  std::error_code EC;
  raw_fd_ostream Bitcode(Output, EC, sys::fs::OF_None);
  if (EC)
    return error("cannot open output: " + EC.message());
  WriteBitcodeToFile(*M, Bitcode);
  Bitcode.flush();
  if (Bitcode.has_error())
    return error("cannot write output bitcode");
  raw_fd_ostream LayoutFile(Schema, EC, sys::fs::OF_Text);
  if (EC)
    return error("cannot open schema: " + EC.message());
  LayoutFile << formatv("{0:2}\n", json::Value(std::move(Description)));
  LayoutFile.flush();
  if (LayoutFile.has_error())
    return error("cannot write output schema");
  outs() << "Selected " << Kept.size() << " fields: "
         << OldLayout->getSizeInBytes() << " -> " << Layout->getSizeInBytes()
         << " payload bytes\n";
  return 0;
}
