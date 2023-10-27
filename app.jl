module App
using PlotlyBase
include("app/trainmodel.jl")
using GenieFramework
@genietools

@app begin
    @in layer_neurons = [8,5]
    @in add_layer = false
    @in remove_layer = false
    @in update = false
    @out traces = []
    @in N_train = 350
    @in train = false
    @in training = false
    @out loss = 0.0
    @in epochs = 2000
    @onbutton add_layer begin
        @show layer_neurons
        push!(layer_neurons,1)
        layer_neurons = copy(layer_neurons)
    end
    @onbutton remove_layer begin
        @show layer_neurons
        pop!(layer_neurons)
        layer_neurons = copy(layer_neurons)
    end
    @onchange isready, update begin
        layer_neurons_extended = vcat([13, layer_neurons, 1]...)
        @show layer_neurons_extended
        N_layers = length(layer_neurons_extended)
        max_n = maximum(layer_neurons_extended)
        nodes_x_coord = [ones(Int(layer_neurons_extended[i]))*i for i in 1:N_layers]
        nodes_y_coord = [collect(1:n) .+ (max_n - n)/2 for n in layer_neurons_extended]
        x_edges = []
        y_edges = []
        for l in 1:N_layers-1
            for i in 1:layer_neurons_extended[l]
                for j in 1:layer_neurons_extended[l+1]
                    push!(x_edges,[nodes_x_coord[l][i],nodes_x_coord[l+1][j],nothing])
                    push!(y_edges,[nodes_y_coord[l][i],nodes_y_coord[l+1][j],nothing])
                end
            end
        end
        nodes = scatter(x=vcat(nodes_x_coord...), y=vcat(nodes_y_coord...), mode="markers",marker=attr(size=16), name="neuron")
        edges = scatter(x=vcat(x_edges...), y=vcat(y_edges...), mode="lines", name="edge")
        traces = [edges,nodes]
    end
    @onchange train begin
        @show "Training"
        training = true
        layer_neurons_extended = vcat([13, layer_neurons, 1]...)
        model,data,loss=train_network(layer_neurons_extended, N_train, epochs)
        training = false
    end
end

@page("/","app.jl.html")
end

