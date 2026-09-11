// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include <linux/bpf.h>
#include <cstdint>
#include <memory>
#include <string>

using namespace llvm;

static cl::opt<std::string> Input("input", cl::Required, cl::desc("Selected BPF bitcode"));
static cl::opt<std::string> Output("output", cl::Required, cl::desc("MPMO BPF bitcode"));
static cl::opt<std::string> Program("program", cl::Required, cl::desc("rss_emit or dtlb_sample"));
static cl::opt<std::string> Shim("shim", cl::Required, cl::desc("Size-specialized Clang MPMO bitcode (-g -O2)"));
static cl::opt<unsigned> Bytes("size", cl::Required, cl::desc("Selected payload bytes: 24 (RSS) or 64 (DTLB)"));

static int error(const Twine &Message) {
  errs() << "mpmo-lower: " << Message << '\n';
  return 1;
}

static uint64_t helperID(const CallBase &Call) {
  auto *Callee = dyn_cast<Operator>(Call.getCalledOperand()->stripPointerCasts());
  if (!Callee || Callee->getOpcode() != Instruction::IntToPtr)
    return 0;
  auto *ID = dyn_cast<ConstantInt>(Callee->getOperand(0));
  return ID && ID->getValue().getActiveBits() <= 64 ? ID->getZExtValue() : 0;
}

static bool ringHelper(uint64_t ID) {
  switch (ID) {
  case BPF_FUNC_ringbuf_output:
  case BPF_FUNC_ringbuf_reserve:
  case BPF_FUNC_ringbuf_submit:
  case BPF_FUNC_ringbuf_discard:
  case BPF_FUNC_ringbuf_query:
  case BPF_FUNC_ringbuf_reserve_dynptr:
  case BPF_FUNC_ringbuf_submit_dynptr:
  case BPF_FUNC_ringbuf_discard_dynptr:
  case BPF_FUNC_user_ringbuf_drain:
    return true;
  default:
    return false;
  }
}

static bool checkTransport(Module &M, const CallInst *Selected = nullptr) {
  // Check every defined function, not just the selected entry's immediate calls.
  for (Function &F : M)
    for (Instruction &I : instructions(F)) {
      auto *Call = dyn_cast<CallBase>(&I);
      if (!Call || Call == Selected)
        continue;
      uint64_t ID = helperID(*Call);
      if (ringHelper(ID)) {
        error("remaining ring transport in '" + F.getName() + "'");
        return false;
      }
      auto *Callee = dyn_cast<Function>(Call->getCalledOperand()->stripPointerCasts());
      if (!ID && (!Callee || (Callee->isDeclaration() && !Callee->isIntrinsic()))) {
        error("unresolved call in '" + F.getName() + "'; cannot exclude ring transport");
        return false;
      }
    }
  return true;
}

static bool onlyRetainedMapUses(GlobalVariable &Map) {
  SmallVector<User *> Work(Map.users());
  SmallPtrSet<User *, 16> Seen;
  while (!Work.empty()) {
    User *U = Work.pop_back_val();
    if (!Seen.insert(U).second)
      continue;
    if (auto *Global = dyn_cast<GlobalValue>(U)) {
      if (Global->getName() == "llvm.used" || Global->getName() == "llvm.compiler.used")
        continue;
      return false;
    }
    if (!isa<Constant>(U))
      return false;
    Work.append(U->user_begin(), U->user_end());
  }
  return true;
}

int main(int argc, char **argv) {
  InitLLVM Init(argc, argv);
  cl::ParseCommandLineOptions(argc, argv, "Lower one selected RSS/DTLB ring output to KERMIT MPMO\n");
  if (Program != "rss_emit" && Program != "dtlb_sample")
    return error("--program must be rss_emit or dtlb_sample");
  bool RSS = Program == "rss_emit";
  if (Bytes != (RSS ? 24u : 64u))
    return error("--size must be 24 for rss_emit or 64 for dtlb_sample");
  if (Input == Output || Shim == Output || sys::fs::equivalent(Input, Output) ||
      sys::fs::equivalent(Shim, Output))
    return error("output must not overwrite input or shim");

  LLVMContext Context;
  SMDiagnostic Diagnostic;
  std::unique_ptr<Module> M = parseIRFile(Input, Diagnostic, Context);
  if (!M) {
    Diagnostic.print(argv[0], errs());
    return 1;
  }
  if (!Triple(M->getTargetTriple()).isBPF() || verifyModule(*M, &errs()))
    return error("input must be valid BPF LLVM IR");
  Function *Entry = M->getFunction(Program);
  if (!Entry || Entry->isDeclaration() || !Entry->hasSection() ||
      Entry->getSection().starts_with(".text"))
    return error("program must name a defined SEC() entry point");

  CallInst *Submit = nullptr;
  for (Instruction &I : instructions(Entry)) {
    auto *Call = dyn_cast<CallBase>(&I);
    if (!Call || helperID(*Call) != BPF_FUNC_ringbuf_output)
      continue;
    if (Submit || !isa<CallInst>(Call))
      return error("exactly one ordinary bpf_ringbuf_output call is required");
    Submit = cast<CallInst>(Call);
  }
  if (!Submit)
    return error("no bpf_ringbuf_output call in the selected program");
  if (Submit->arg_size() != 4 || !Submit->getType()->isIntegerTy(64) ||
      !Submit->getArgOperand(0)->getType()->isPointerTy() ||
      Submit->getArgOperand(1)->getType() != PointerType::getUnqual(Context) ||
      Submit->getCallingConv() != CallingConv::C || Submit->isMustTailCall() ||
      Submit->hasOperandBundles())
    return error("unexpected bpf_ringbuf_output signature");
  const DataLayout &DL = M->getDataLayout();
  int64_t Offset = 0;
  auto *Ring = dyn_cast<GlobalVariable>(GetPointerBaseWithConstantOffset(
      Submit->getArgOperand(0), Offset, DL));
  StringRef RingName = RSS ? "rss_events" : "dtlb_events";
  if (!Ring || Offset || Ring->getName() != RingName || Ring->isDeclaration() ||
      Ring->getSection() != ".maps")
    return error("output must use the original '" + RingName + "' map at offset zero");
  auto *Size = dyn_cast<ConstantInt>(Submit->getArgOperand(2));
  auto *Flags = dyn_cast<ConstantInt>(Submit->getArgOperand(3));
  if (!Size || !Size->getType()->isIntegerTy(64) || !Size->equalsInt(Bytes))
    return error("ring output size does not match --size");
  if (!Flags || !Flags->getType()->isIntegerTy(64) || !Flags->isZero())
    return error("ring output flags must be constant zero");
  Offset = 0;
  auto *Payload = dyn_cast<AllocaInst>(GetPointerBaseWithConstantOffset(
      Submit->getArgOperand(1), Offset, DL));
  auto Allocation = Payload ? Payload->getAllocationSize(DL) : std::nullopt;
  if (!Allocation || Allocation->isScalable() || Offset ||
      Allocation->getFixedValue() != Bytes)
    return error("payload must be one whole selected stack allocation matching --size");
  if (!checkTransport(*M, Submit))
    return 1;

  std::unique_ptr<Module> S = parseIRFile(Shim, Diagnostic, Context);
  if (!S) {
    Diagnostic.print(argv[0], errs());
    return 1;
  }
  if (!Triple(S->getTargetTriple()).isBPF() ||
      S->getTargetTriple() != M->getTargetTriple() ||
      S->getDataLayout() != DL || verifyModule(*S, &errs()))
    return error("shim must be valid BPF IR with the same target and data layout");
  Function *Emit = S->getFunction("policy_mpmo_emit");
  if (!Emit || Emit->isDeclaration() || Emit->hasLocalLinkage() || Emit->hasSection() ||
      Emit->isVarArg() || !Emit->getReturnType()->isIntegerTy(64) ||
      Emit->arg_size() != 1 || Emit->getArg(0)->getType() != PointerType::getUnqual(Context) ||
      Emit->getCallingConv() != CallingConv::C)
    return error("shim must define long policy_mpmo_emit(const void *payload)");
  unsigned Copies = 0;
  for (Instruction &I : instructions(Emit))
    if (auto *Copy = dyn_cast<MemCpyInst>(&I)) {
      auto *Length = dyn_cast<ConstantInt>(Copy->getLength());
      if (!Length || !Length->equalsInt(Bytes) || Copy->isVolatile() ||
          Copy->getSource() != Emit->getArg(0))
        return error("shim memcpy must copy exactly --size payload bytes");
      ++Copies;
    }
  if (Copies != 1)
    return error("shim must contain one constant-size memcpy; compile with -g -O2");

  StringRef HeadName = RSS ? "mpmo_head_rss" : "mpmo_head_dtlb";
  StringRef BuffName = RSS ? "mpmo_buff_rss" : "mpmo_buff_dtlb";
  for (StringRef Name : {HeadName, BuffName}) {
    auto *Map = S->getGlobalVariable(Name);
    if (!Map || Map->isDeclaration() || Map->getSection() != ".maps")
      return error("shim must define map '" + Name + "'");
  }
  for (const GlobalValue &G : S->global_values()) {
    if (G.getName().starts_with("llvm."))
      continue;
    if (&G != Emit && G.getName() != HeadName && G.getName() != BuffName)
      return error("unexpected shim symbol '" + G.getName() + "'; compile with -g -O2");
    if (M->getNamedValue(G.getName()))
      return error("input already contains shim symbol '" + G.getName() + "'");
  }
  if (!checkTransport(*S))
    return 1;
  if (Linker::linkModules(*M, std::move(S)))
    return error("cannot link MPMO shim");
  Emit = M->getFunction("policy_mpmo_emit");

  IRBuilder<> Builder(Submit);
  CallInst *Replacement = Builder.CreateCall(Emit, {Submit->getArgOperand(1)}, "mpmo_status");
  Replacement->setDebugLoc(Submit->getDebugLoc());
  // Keep the original status users and control flow, including error accounting
  // and RSS owner deletion. Never replace a possibly failing output with zero.
  Submit->replaceAllUsesWith(Replacement);
  Submit->eraseFromParent();
  InlineFunctionInfo IFI;
  InlineResult Inlined = InlineFunction(*Replacement, IFI);
  if (!Inlined.isSuccess())
    return error(Twine("cannot inline MPMO shim: ") + Inlined.getFailureReason());
  if (!Emit->use_empty())
    return error("unexpected remaining policy_mpmo_emit use");
  Emit->eraseFromParent();

  // Keep the original .maps declaration/BTF for loader autocreate=false, but
  // forbid executable references, aliases, and escapes through other globals.
  if (!onlyRetainedMapUses(*Ring))
    return error("original ring map still has live uses");
  if (!checkTransport(*M))
    return 1;
  if (verifyModule(*M, &errs()))
    return error("lowering produced invalid LLVM IR");

  std::error_code EC;
  raw_fd_ostream Bitcode(Output, EC, sys::fs::OF_None);
  if (EC)
    return error("cannot open output: " + EC.message());
  WriteBitcodeToFile(*M, Bitcode);
  Bitcode.flush();
  if (Bitcode.has_error())
    return error("cannot write output bitcode");
  outs() << "Lowered " << Program << ": " << Bytes << " payload bytes, 32 MPMO slots\n";
  return 0;
}
