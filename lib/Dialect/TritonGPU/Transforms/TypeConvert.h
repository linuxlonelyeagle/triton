#ifndef TRITON_LIB_DIALECT_TRITONGPU_TRANSFORMS_TYPECONVERT_H_
#define TRITON_LIB_DIALECT_TRITONGPU_TRANSFORMS_TYPECONVERT_H_

#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"

#include "DotOpHelpers.h"

using namespace mlir;
using namespace mlir::triton;

class TritonGPUToLLVMTypeConverter : public LLVMTypeConverter {
public:
  using TypeConverter::convertType;

  TritonGPUToLLVMTypeConverter(MLIRContext *ctx, LowerToLLVMOptions &option,
                               const DataLayoutAnalysis *analysis = nullptr);

  Type convertTritonPointerType(triton::PointerType type);

  llvm::Optional<Type> convertTritonTensorType(RankedTensorType type);
};

namespace mlir {
class TritonLLVMFunctionConversionTarget : public ConversionTarget {
public:
  explicit TritonLLVMFunctionConversionTarget(MLIRContext &ctx);
};
} // namespace mlir

#endif // TRITON_LIB_DIALECT_TRITONGPU_TRANSFORMS_TYPECONVERT_H_