using DataFrames, Flux, Random
using MLDatasets: BostonHousing
using Flux: @epochs

function train_network(layer_neurons::Vector{Int},N_train::Int, epochs::Int)
    features = BostonHousing.features()
    target = BostonHousing.targets()
    N_test = length(target) - N_train
    train_idx = randperm(length(target))[1:N_train]
    test_idx = setdiff(1:length(target), train_idx)
    x_train = features[:,train_idx]
    x_test = features[:,test_idx]
    y_train = target[:,train_idx]
    y_test = target[:,test_idx]

    model = Dense(layer_neurons[1], layer_neurons[2], relu)
    for i in 2:length(layer_neurons)-1
        model = Chain(model, Dense(layer_neurons[i],layer_neurons[i+1],relu))
    end

    opt = ADAM()

    loss(x, y) = Flux.Losses.mse(model(x), y)
    err = loss(x_train,y_train)
    data = [(x_train,y_train)]
    parameters = Flux.params(model)
    @epochs epochs Flux.train!(loss, parameters, data, opt)
    return model, data, err
end
