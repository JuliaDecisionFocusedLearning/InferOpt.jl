@testitem "Test interfaces are correctly implemented" default_imports = false begin
    using InferOpt, RequiredInterfaces, Test
    const RI = RequiredInterfaces

    @test RI.check_interface_implemented(AbstractRegularized, RegularizedFrankWolfe)
    @test RI.check_interface_implemented(AbstractRegularized, SoftArgmax)
    @test RI.check_interface_implemented(AbstractRegularized, SparseArgmax)
end
