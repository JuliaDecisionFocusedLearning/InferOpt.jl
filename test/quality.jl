using Aqua
using InferOpt

Aqua.test_all(
    InferOpt;
    deps_compat=false,
    project_extras=false,
    ambiguities=false
)
