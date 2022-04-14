Aqua.test_all(
    InferOpt;
    deps_compat=false,
    project_extras=false,
    ambiguities=false  # TODO: set this to true once the problem with FrankWolfe > MathOptInterface > MutableArithmetics is solved
)
