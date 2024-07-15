#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::triton;

#define GEN_PASS_CLASSES
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

namespace {

struct FuncOpConvertPass
    : public TritonGPUFuncOpConvertBase<FuncOpConvertPass> {
  FuncOpConvertPass() = default;
  void runOnOperation() override {}
};
} // namespace

std::unique_ptr<Pass> mlir::createTritonGPUFuncOpConvertPass() {
  return std::make_unique<FuncOpConvertPass>();
}
