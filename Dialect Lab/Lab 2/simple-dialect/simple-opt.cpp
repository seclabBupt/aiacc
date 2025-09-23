#include "SimpleDialect.h"
#include "SimplePass.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassRegistry.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<simple::SimpleDialect>();
  
  // 注册我们的 Pass
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return simple::createSimpleToArithPass();
  });
  
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Simple dialect test\n", registry));
}
