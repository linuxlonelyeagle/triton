#include "Dialect/NVGPU/IR/Dialect.h"
#include "TritonNVIDIAGPUToLLVM/Passes.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/LLVMCommon/VectorPattern.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Analysis/Membar.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

#include "PatternTritonGPUOpToLLVM.h"
#include "Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/TypeConverter.h"

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_CONVERTTRITONGPUMAKERANGETOLLVM
#include "TritonNVIDIAGPUToLLVM/Passes.h.inc"
} // namespace triton
} // namespace mlir

using namespace mlir;
using namespace mlir::triton::NVIDIA;
namespace ttng = mlir::triton::nvidia_gpu;

namespace {

// pass ws related named attrs.
static void addAttrs(Operation *op, ArrayRef<mlir::NamedAttribute> attrs) {
  for (const NamedAttribute attr : attrs)
    op->setAttr(attr.getName(), attr.getValue());
}

class TritonLLVMFunctionConversionTarget : public ConversionTarget {
public:
  explicit TritonLLVMFunctionConversionTarget(MLIRContext &ctx)
      : ConversionTarget(ctx) {
    addLegalDialect<index::IndexDialect>();
    addLegalDialect<LLVM::LLVMDialect>();
    addLegalDialect<NVVM::NVVMDialect>();
    addLegalOp<mlir::UnrealizedConversionCastOp>();
  }
};

class TritonLLVMConversionTarget : public ConversionTarget {
public:
  explicit TritonLLVMConversionTarget(MLIRContext &ctx)
      : ConversionTarget(ctx) {
    addLegalDialect<LLVM::LLVMDialect>();
    addLegalDialect<NVVM::NVVMDialect>();
    addLegalDialect<mlir::triton::nvgpu::NVGPUDialect>();
    // addIllegalDialect<triton::TritonDialect>();
    addIllegalDialect<triton::gpu::TritonGPUDialect>();
    addIllegalDialect<triton::nvidia_gpu::TritonNvidiaGPUDialect>();
    addLegalDialect<mlir::gpu::GPUDialect>();
    addLegalDialect<mlir::arith::ArithDialect>();
    addLegalOp<mlir::UnrealizedConversionCastOp>();
  }
};

struct ConvertTritonGPULoadStoreToLLVM
    : public triton::impl::ConvertTritonGPUMakeRangeToLLVMBase<
          ConvertTritonGPULoadStoreToLLVM> {
  using ConvertTritonGPUMakeRangeToLLVMBase::
      ConvertTritonGPUMakeRangeToLLVMBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<triton::nvgpu::NVGPUDialect, LLVM::LLVMDialect,
                    NVVM::NVVMDialect>();
  }

  ConvertTritonGPULoadStoreToLLVM(int32_t computeCapability)
      : ConvertTritonGPUMakeRangeToLLVMBase({computeCapability}) {}

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    mlir::LowerToLLVMOptions option(context);
    option.overrideIndexBitwidth(32);
    TritonGPUToLLVMTypeConverter typeConverter(context, option);
    TritonLLVMConversionTarget convTarget(*context);
    RewritePatternSet patterns(context);
    TargetInfo targetInfo(computeCapability);
    int benefit = patternBenefitPrioritizeOverLLVMConversions;
    populateMakeRangeOpToLLVMPattern(typeConverter, targetInfo, patterns,
                                     benefit);
    if (failed(applyPartialConversion(mod, convTarget, std::move(patterns))))
      return signalPassFailure();
  }
  static Value promoteOperand(OpBuilder &builder, Location loc, Value operand,
                              Type promotedType) {
    Type tensorPromotedType = cast<RankedTensorType>(operand.getType())
                                  .cloneWith(std::nullopt, promotedType);
    return builder.create<triton::FpToFpOp>(loc, tensorPromotedType, operand);
  }
};

} // anonymous namespace

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>>
createConvertTritonGpuMakeRangeToLLVMPass() {
  return std::make_unique<ConvertTritonGPULoadStoreToLLVM>();
}
std::unique_ptr<OperationPass<ModuleOp>>
createConvertTritonGpuMakeRangeToLLVMPass(int32_t computeCapability) {
  return std::make_unique<ConvertTritonGPULoadStoreToLLVM>(computeCapability);
}

} // namespace triton
} // namespace mlir