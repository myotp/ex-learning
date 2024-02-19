defmodule ExLearning.Axon.NaiveLinearRegression do
  import Nx.Defn

  def build_model() do
    Axon.input("input", shape: {nil, 1})
    |> Axon.dense(1,
      kernel_initializer: :ones,
      bias_initializer: :ones,
      use_bias: true
    )
  end

  def train_model(model, inputs, y_true, iterations, lr) do
    {init_fn, pred_fn} = Axon.build(model)
    init_params = init_fn.(Nx.template({1, 1}, :u8), %{})
    {optimizer_init_fn, optimizer_update_fn} = Polaris.Optimizers.sgd(learning_rate: lr)
    init_optimizer_state = optimizer_init_fn.(init_params)

    {trained_params, _} =
      Enum.reduce(
        1..iterations,
        {init_params, init_optimizer_state},
        fn i, {params, optimizer_state} ->
          {new_params, new_optimizer_state, loss} =
            update(params, optimizer_state, inputs, y_true, optimizer_update_fn, pred_fn)

          IO.puts("Iteration #{i} => loss: #{loss |> Nx.mean() |> Nx.to_number()}")
          {new_params, new_optimizer_state}
        end
      )

    trained_params
  end

  def loss_value_and_grad(pred_fn, params, inputs, y_true) do
    value_and_grad(params, fn params ->
      y_pred = pred_fn.(params, inputs)
      Axon.Losses.mean_squared_error(y_true, y_pred)
    end)
  end

  def my_loss_value_and_grad({w, b}, inputs, y_true) do
    inputs = Nx.squeeze(inputs)
    y_true = Nx.squeeze(y_true)

    value_and_grad({w, b}, fn {w, b} ->
      ExLearning.Core.LinearRegression.nx_loss(inputs, y_true, w, b)
    end)
  end

  # https://hexdocs.pm/nx/Nx.Defn.html#value_and_grad/3-examples
  defn objective(predict_fn, loss_fn, params, inputs, y_true) do
    y_pred = predict_fn.(params, inputs)
    loss = loss_fn.(y_pred, y_true)
    {y_pred, loss}
  end

  # https://hexdocs.pm/polaris/Polaris.Optimizers.html#module-example
  defn update(params, optimizer_state, inputs, targets, update_fn, pred_fn) do
    # 用我自己的value_and_grad替换
    {loss, gradient} =
      value_and_grad(params, fn params ->
        y_pred = pred_fn.(params, inputs)
        Axon.Losses.mean_squared_error(targets, y_pred)
      end)

    {scaled_updates, new_optimizer_state} = update_fn.(gradient, optimizer_state, params)
    {Polaris.Updates.apply_updates(params, scaled_updates), new_optimizer_state, loss}
  end
end
