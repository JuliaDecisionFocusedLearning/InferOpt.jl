# Optimization

!!! info "Work in progress"
    Come back later!

```@example
using AbstractTrees, InferOpt, InteractiveUtils
AbstractTrees.children(x::Type) = subtypes(x)
print_tree(InferOpt.AbstractOptimizationLayer)
```