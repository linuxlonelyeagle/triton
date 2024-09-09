#ifndef TRITONGPU_CONVERSION_TRITONNVIDIAGPUTOLLVM_PASSES_H
#define TRITONGPU_CONVERSION_TRITONNVIDIAGPUTOLLVM_PASSES_H

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include <memory>

namespace mlir {

class ModuleOp;
template <typename T> class OperationPass;

namespace triton {

#define GEN_PASS_DECL
#include "nvidia/include/TritonNVIDIAGPUToLLVM/Passes.h.inc"

namespace NVIDIA {
std::unique_ptr<OperationPass<ModuleOp>>
createDecomposeUnsupportedConversionsPass();

} // namespace NVIDIA

std::unique_ptr<OperationPass<ModuleOp>> createConvertTritonGPUToLLVMPass();
std::unique_ptr<OperationPass<ModuleOp>>
createConvertTritonGPUToLLVMPass(int32_t computeCapability);

std::unique_ptr<OperationPass<ModuleOp>> createConvertTritonGPUFuncToLLVMPass();
std::unique_ptr<OperationPass<ModuleOp>>
createConvertTritonGPUFuncToLLVMPass(int32_t computeCapability);

std::unique_ptr<OperationPass<ModuleOp>> createTritonGpuInitShareMemoryPass();

std::unique_ptr<OperationPass<ModuleOp>>
createConvertTritonGpuLoadStoreToLLVMPass();
std::unique_ptr<OperationPass<ModuleOp>>
createConvertTritonGpuLoadStoreToLLVMPass(int32_t computeCapability);

std::unique_ptr<OperationPass<ModuleOp>>
createConvertTritonGpuMakeRangeToLLVMPass();
std::unique_ptr<OperationPass<ModuleOp>>
createConvertTritonGpuMakeRangeToLLVMPass(int32_t computeCapability);

std::unique_ptr<OperationPass<ModuleOp>>
createConvertTritonGpuViewOpToLLVMPass();
std::unique_ptr<OperationPass<ModuleOp>>
createConvertTritonGpuViewOpToLLVMPass(int32_t computeCapability);


std::unique_ptr<OperationPass<ModuleOp>>
createConvertTritonGpuSPMDOpToLLVMPass();
std::unique_ptr<OperationPass<ModuleOp>>
createConvertTritonGpuSPMDOpToLLVMPass(int32_t computeCapability);

std::unique_ptr<OperationPass<ModuleOp>>
createConvertTritonGpuElementwiseOpToLLVMPass();
std::unique_ptr<OperationPass<ModuleOp>>
createConvertTritonGpuElementwiseOpToLLVMPass(int32_t computeCapability);

#define GEN_PASS_REGISTRATION
#include "nvidia/include/TritonNVIDIAGPUToLLVM/Passes.h.inc"

} // namespace triton

} // namespace mlir

#endif
