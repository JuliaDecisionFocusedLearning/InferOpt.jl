using InferOpt
using JET

if VERSION >= v"1.9"
    JET.test_package(InferOpt; target_defined_modules=true)
end
