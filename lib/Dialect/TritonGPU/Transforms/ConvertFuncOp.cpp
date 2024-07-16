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

using ::mlir::LLVM::DotOpFMAConversionHelper;
using ::mlir::LLVM::DotOpMmaV1ConversionHelper;
using ::mlir::LLVM::MMA16816ConversionHelper;
using ::mlir::triton::gpu::BlockedEncodingAttr;
using ::mlir::triton::gpu::DotOperandEncodingAttr;
using ::mlir::triton::gpu::getElemsPerThread;
using ::mlir::triton::gpu::MmaEncodingAttr;
using ::mlir::triton::gpu::SharedEncodingAttr;
using ::mlir::triton::gpu::SliceEncodingAttr;

#define GEN_PASS_CLASSES
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

namespace {

class TritonGPUToLLVMTypeConverter : public LLVMTypeConverter {
public:
  using TypeConverter::convertType;

  TritonGPUToLLVMTypeConverter(MLIRContext *ctx, LowerToLLVMOptions &option,
                               const DataLayoutAnalysis *analysis = nullptr)
      : LLVMTypeConverter(ctx, option, analysis) {
    addConversion([&](triton::PointerType type) -> llvm::Optional<Type> {
      return convertTritonPointerType(type);
    });
    addConversion([&](RankedTensorType type) -> llvm::Optional<Type> {
      return convertTritonTensorType(type);
    });
    // Internally store float8 as int8
    addConversion([&](triton::Float8Type type) -> llvm::Optional<Type> {
      return IntegerType::get(type.getContext(), 8);
    });
    // Internally store bfloat16 as int16
    addConversion([&](BFloat16Type type) -> llvm::Optional<Type> {
      return IntegerType::get(type.getContext(), 16);
    });
  }

  Type convertTritonPointerType(triton::PointerType type) {
    // Recursively translate pointee type
    return LLVM::LLVMPointerType::get(convertType(type.getPointeeType()),
                                      type.getAddressSpace());
  }

  llvm::Optional<Type> convertTritonTensorType(RankedTensorType type) {
    auto ctx = type.getContext();
    Attribute layout = type.getEncoding();
    SmallVector<int64_t> shape(type.getShape().begin(), type.getShape().end());

    if (layout &&
        (layout.isa<BlockedEncodingAttr>() || layout.isa<SliceEncodingAttr>() ||
         layout.isa<MmaEncodingAttr>())) {
      unsigned numElementsPerThread = getElemsPerThread(type);
      SmallVector<Type, 4> types(numElementsPerThread,
                                 convertType(type.getElementType()));
      return LLVM::LLVMStructType::getLiteral(ctx, types);
    } else if (auto shared_layout =
                   layout.dyn_cast_or_null<SharedEncodingAttr>()) {
      SmallVector<Type, 4> types;
      // base ptr
      auto ptrType =
          LLVM::LLVMPointerType::get(convertType(type.getElementType()), 3);
      types.push_back(ptrType);
      // shape dims
      auto rank = type.getRank();
      // offsets + strides
      for (auto i = 0; i < rank * 2; i++) {
        types.push_back(IntegerType::get(ctx, 32));
      }
      return LLVM::LLVMStructType::getLiteral(ctx, types);
    } else if (auto dotOpLayout =
                   layout.dyn_cast_or_null<DotOperandEncodingAttr>()) {
      if (dotOpLayout.getParent()
              .isa<BlockedEncodingAttr>()) { // for parent is blocked layout
        int numElemsPerThread =
            DotOpFMAConversionHelper::getNumElemsPerThread(shape, dotOpLayout);

        return LLVM::LLVMStructType::getLiteral(
            ctx, SmallVector<Type>(numElemsPerThread, type::f32Ty(ctx)));
      } else { // for parent is MMA layout
        auto mmaLayout = dotOpLayout.getParent().cast<MmaEncodingAttr>();
        auto wpt = mmaLayout.getWarpsPerCTA();
        Type elemTy = convertType(type.getElementType());
        if (mmaLayout.isAmpere()) {
          const llvm::DenseMap<int, Type> targetTyMap = {
              {32, vec_ty(elemTy, 1)},
              {16, vec_ty(elemTy, 2)},
              {8, vec_ty(elemTy, 4)},
          };
          Type targetTy;
          if (targetTyMap.count(elemTy.getIntOrFloatBitWidth())) {
            targetTy = targetTyMap.lookup(elemTy.getIntOrFloatBitWidth());
            // <2xi16>/<4xi8> => i32
            // We are doing this because NVPTX inserts extra integer instrs to
            // pack & unpack vectors of sub-word integers
            // Note: this needs to be synced with
            //       DotOpMmaV2ConversionHelper::loadX4
            if (elemTy.isa<IntegerType>() &&
                (elemTy.getIntOrFloatBitWidth() == 8 ||
                 elemTy.getIntOrFloatBitWidth() == 16))
              targetTy = IntegerType::get(ctx, 32);
          } else {
            assert(false && "Unsupported element type");
          }
          if (dotOpLayout.getOpIdx() == 0) { // $a
            auto elems =
                MMA16816ConversionHelper::getANumElemsPerThread(type, wpt[0]);
            return struct_ty(SmallVector<Type>(elems, targetTy));
          }
          if (dotOpLayout.getOpIdx() == 1) { // $b
            auto elems =
                MMA16816ConversionHelper::getBNumElemsPerThread(type, wpt[1]);
            return struct_ty(SmallVector<Type>(elems, targetTy));
          }
        }

        if (mmaLayout.isVolta()) {
          auto [isARow, isBRow, isAVec4, isBVec4, mmaId] =
              mmaLayout.decodeVoltaLayoutStates();
          DotOpMmaV1ConversionHelper helper(mmaLayout);
          if (dotOpLayout.getOpIdx() == 0) { // $a
            DotOpMmaV1ConversionHelper::AParam param(isARow, isAVec4);
            int elems =
                helper.numElemsPerThreadA(shape, isARow, isAVec4, param.vec);
            Type x2Ty = vec_ty(elemTy, 2);
            return struct_ty(SmallVector<Type>(elems, x2Ty));
          }
          if (dotOpLayout.getOpIdx() == 1) { // $b
            DotOpMmaV1ConversionHelper::BParam param(isBRow, isBVec4);
            int elems =
                helper.numElemsPerThreadB(shape, isBRow, isBVec4, param.vec);
            Type x2Ty = vec_ty(elemTy, 2);
            return struct_ty(SmallVector<Type>(elems, x2Ty));
          }
        }
      }

      llvm::errs() << "Unexpected dot operand layout detected in "
                      "TritonToLLVMTypeConverter";
      return llvm::None;
    }

    return llvm::None;
  }
};

class TritonLLVMFunctionConversionTarget : public ConversionTarget {
public:
  explicit TritonLLVMFunctionConversionTarget(MLIRContext &ctx)
      : ConversionTarget(ctx) {
    addLegalDialect<LLVM::LLVMDialect>();
    addLegalDialect<NVVM::NVVMDialect>();
    addIllegalOp<mlir::FuncOp>();
    addLegalOp<mlir::UnrealizedConversionCastOp>();
  }
};

struct FuncOpConversionBase : public ConvertOpToLLVMPattern<FuncOp> {
private:
  /// Only retain those attributes that are not constructed by
  /// `LLVMFuncOp::build`. If `filterArgAttrs` is set, also filter out argument
  /// attributes.
  static void filterFuncAttributes(ArrayRef<NamedAttribute> attrs,
                                   bool filterArgAttrs,
                                   SmallVectorImpl<NamedAttribute> &result) {
    for (const auto &attr : attrs) {
      if (attr.getName() == SymbolTable::getSymbolAttrName() ||
          attr.getName() == FunctionOpInterface::getTypeAttrName() ||
          attr.getName() == "std.varargs" ||
          (filterArgAttrs &&
           attr.getName() == FunctionOpInterface::getArgDictAttrName()))
        continue;
      result.push_back(attr);
    }
  }

  /// Helper function for wrapping all attributes into a single DictionaryAttr
  static auto wrapAsStructAttrs(OpBuilder &b, ArrayAttr attrs) {
    return DictionaryAttr::get(b.getContext(),
                               b.getNamedAttr("llvm.struct_attrs", attrs));
  }

protected:
  using ConvertOpToLLVMPattern<FuncOp>::ConvertOpToLLVMPattern;

  // Convert input FuncOp to LLVMFuncOp by using the LLVMTypeConverter provided
  // to this legalization pattern.
  LLVM::LLVMFuncOp
  convertFuncOpToLLVMFuncOp(FuncOp funcOp,
                            ConversionPatternRewriter &rewriter) const {
    // Convert the original function arguments. They are converted using the
    // LLVMTypeConverter provided to this legalization pattern.
    auto varargsAttr = funcOp->getAttrOfType<BoolAttr>("func.varargs");
    TypeConverter::SignatureConversion result(funcOp.getNumArguments());
    auto llvmType = getTypeConverter()->convertFunctionSignature(
        funcOp.getType(), varargsAttr && varargsAttr.getValue(), result);
    if (!llvmType)
      return nullptr;

    // Propagate argument/result attributes to all converted arguments/result
    // obtained after converting a given original argument/result.
    SmallVector<NamedAttribute, 4> attributes;
    filterFuncAttributes(funcOp->getAttrs(), /*filterArgAttrs=*/true,
                         attributes);
    if (ArrayAttr resAttrDicts = funcOp.getAllResultAttrs()) {
      assert(!resAttrDicts.empty() && "expected array to be non-empty");
      auto newResAttrDicts =
          (funcOp.getNumResults() == 1)
              ? resAttrDicts
              : rewriter.getArrayAttr(
                    {wrapAsStructAttrs(rewriter, resAttrDicts)});
      attributes.push_back(rewriter.getNamedAttr(
          FunctionOpInterface::getResultDictAttrName(), newResAttrDicts));
    }
    if (ArrayAttr argAttrDicts = funcOp.getAllArgAttrs()) {
      SmallVector<Attribute, 4> newArgAttrs(
          llvmType.cast<LLVM::LLVMFunctionType>().getNumParams());
      for (unsigned i = 0, e = funcOp.getNumArguments(); i < e; ++i) {
        auto mapping = result.getInputMapping(i);
        assert(mapping && "unexpected deletion of function argument");
        for (size_t j = 0; j < mapping->size; ++j)
          newArgAttrs[mapping->inputNo + j] = argAttrDicts[i];
      }
      attributes.push_back(
          rewriter.getNamedAttr(FunctionOpInterface::getArgDictAttrName(),
                                rewriter.getArrayAttr(newArgAttrs)));
    }
    for (const auto &pair : llvm::enumerate(attributes)) {
      if (pair.value().getName() == "llvm.linkage") {
        attributes.erase(attributes.begin() + pair.index());
        break;
      }
    }

    // Create an LLVM function, use external linkage by default until MLIR
    // functions have linkage.
    LLVM::Linkage linkage = LLVM::Linkage::External;
    if (funcOp->hasAttr("llvm.linkage")) {
      auto attr =
          funcOp->getAttr("llvm.linkage").dyn_cast<mlir::LLVM::LinkageAttr>();
      if (!attr) {
        funcOp->emitError()
            << "Contains llvm.linkage attribute not of type LLVM::LinkageAttr";
        return nullptr;
      }
      linkage = attr.getLinkage();
    }
    auto newFuncOp = rewriter.create<LLVM::LLVMFuncOp>(
        funcOp.getLoc(), funcOp.getName(), llvmType, linkage,
        /*dsoLocal*/ false, attributes);
    rewriter.inlineRegionBefore(funcOp.getBody(), newFuncOp.getBody(),
                                newFuncOp.end());
    if (failed(rewriter.convertRegionTypes(&newFuncOp.getBody(), *typeConverter,
                                           &result)))
      return nullptr;

    return newFuncOp;
  }
};

struct ReturenOpConvertsion : public ConvertOpToLLVMPattern<ReturnOp> {
  using ConvertOpToLLVMPattern<ReturnOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(ReturnOp returnOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<LLVM::ReturnOp>(
        returnOp, returnOp.getOperandTypes(), returnOp.getOperands());
    return success();
  }
};

/// FuncOp legalization pattern that converts MemRef arguments to pointers to
/// MemRef descriptors (LLVM struct data types) containing all the MemRef type
/// information.
struct FuncOpConversion : public FuncOpConversionBase {
  FuncOpConversion(LLVMTypeConverter &converter, int numWarps,
                   PatternBenefit benefit)
      : FuncOpConversionBase(converter, benefit), numWarps(numWarps) {}

  LogicalResult
  matchAndRewrite(FuncOp funcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto newFuncOp = convertFuncOpToLLVMFuncOp(funcOp, rewriter);
    if (!newFuncOp)
      return failure();

    auto ctx = funcOp->getContext();

    // Set an attribute to indicate this function is a kernel entry.
    newFuncOp->setAttr("nvvm.kernel",
                       rewriter.getIntegerAttr(type::u1Ty(ctx), 1));

    // Set an attribute for maxntidx, it could be used in latter LLVM codegen
    // for `nvvm.annotation` metadata.
    newFuncOp->setAttr("nvvm.maxntid",
                       rewriter.getIntegerAttr(i32_ty, 32 * numWarps));

    rewriter.eraseOp(funcOp);
    return success();
  }

private:
  int numWarps{0};
};

struct FuncOpConvertPass
    : public TritonGPUFuncOpConvertBase<FuncOpConvertPass> {
  FuncOpConvertPass() = default;
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp moduleOp = getOperation();
    int warps = triton::gpu::TritonGPUDialect::getNumWarps(moduleOp);
    mlir::LowerToLLVMOptions option(context);
    option.overrideIndexBitwidth(32);
    TritonGPUToLLVMTypeConverter typeConverter(context, option);
    TritonLLVMFunctionConversionTarget funcTarget(*context);
    RewritePatternSet funcPatterns(context);
    funcPatterns.add<FuncOpConversion>(typeConverter, warps, /*benefit*/ 1);
    funcPatterns.add<ReturenOpConvertsion>(typeConverter);
    if (failed(applyPartialConversion(moduleOp, funcTarget,
                                      std::move(funcPatterns))))
      return signalPassFailure();
  }
};
} // namespace

std::unique_ptr<Pass> mlir::createTritonGPUFuncOpConvertPass() {
  return std::make_unique<FuncOpConvertPass>();
}
