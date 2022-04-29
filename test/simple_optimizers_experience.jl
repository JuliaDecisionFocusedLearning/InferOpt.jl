## Dimensions and parameters

N = 1000
dim_xy = 10
epochs = 200

## True model

true_model = Dense([i ==j ? -1. : 0. for i in 1:dim_xy, j in 1:dim_xy])
maximizers = Dict("argmax" => argmax_optimizer, "ranking" => ranking)
cost(y; instance) = instance' * y

## Learning pipelines

pipelines = Dict(
    "argmax" => (model=Dense(dim_xy, dim_xy), loss=PerturbedCost(maximizers["argmax"], cost; ε=0.1, M=10)),
    "ranking" => (model=Dense(dim_xy, dim_xy), loss=PerturbedCost(maximizers["ranking"], cost; ε=0.1, M=10))
)

## Test loop

for setting in ["argmax", "ranking"]
    @testset verbose = true "Setting: $setting - Target: none" begin
        @unpack X, thetas, Y = generate_dataset(
            true_model, maximizers[setting]; N=N, dim_x=dim_xy, dim_y=dim_xy, σ=0
        )
        X_train, X_test = train_test_split(X)
        Y_train, Y_test = train_test_split(Y)

        @unpack model, loss = pipelines[setting]

        V_train = [loss.cost(y; instance=x) for (x, y) in zip(X_train, Y_train)]
        V_test = [loss.cost(y; instance=x) for (x, y) in zip(X_test, Y_test)]

        opt = ADAM()

        training_losses = Float64[]
        test_losses = Float64[]
        train_objective_gap = Float64[]
        test_objective_gap = Float64[]

        flux_loss(x) = loss(model(x); instance=x)

        @showprogress for _ in 1:epochs
            l_train = sum(flux_loss(x) for x in X_train)
            l_test = sum(flux_loss(x) for x in X_test)

            Y_train_pred = generate_predictions(model, maximizers[setting], X_train)
            V_train_pred = [cost(y; instance=x) for (x, y) in zip(X_train, Y_train_pred)]
            Δv_train = mean(
                (v_pred - v) / abs(v) for
                (v_pred, v) in zip(V_train_pred, V_train)
            )

            Y_test_pred = generate_predictions(model, maximizers[setting], X_test)
            V_test_pred = [cost(y; instance=x) for (x, y) in zip(X_test, Y_test_pred)]
            Δv_test = mean(
                (v_pred - v) / abs(v) for
                (v_pred, v) in zip(V_test_pred, V_test)
            )

            push!(training_losses, l_train)
            push!(test_losses, l_test)
            push!(train_objective_gap, Δv_train)
            push!(test_objective_gap, Δv_test)

            Flux.train!(flux_loss, Flux.params(model), X_train, opt)
        end

        @test training_losses[end] < training_losses[1]
        @test test_losses[end] < test_losses[1]
        @test train_objective_gap[end] < train_objective_gap[1]
        @test test_objective_gap[end] < test_objective_gap[1]

        if SHOW_PLOTS
            println(lineplot(training_losses))
            println(lineplot(test_losses))
            println(lineplot(train_objective_gap))
            println(lineplot(test_objective_gap))
        end
    end
end
