defmodule ExLearning.Axon.SimpleCNN do
  def build_model_4x4() do
    build_conv({nil, 4, 4, 1})
  end

  def build_conv(input_shape) do
    Axon.input("input", shape: input_shape)
    |> Axon.conv(1, kernel_size: {2, 2}, padding: :valid, use_bias: true)
  end

  def simple_params() do
    %{"conv_0" => %{"bias" => Nx.tensor([1]), "kernel" => simple_filter_2x2()}}
  end

  def simple_filter_2x2() do
    Nx.tensor([
      [-1, 2],
      [4, -2]
    ])
    |> Nx.reshape({2, 2, 1, 1})
  end

  def sample_input_4x4() do
    Nx.tensor([
      [1, 0, 0, 1],
      [1, 0, 1, 0],
      [1, 1, 0, 0],
      [0, 1, 0, 1]
    ])
    |> Nx.reshape({1, 4, 4, 1})
  end

  def build_sample_pooling_4x4() do
    Axon.input("input", shape: {nil, 4, 4, 1})
    |> Axon.max_pool(kernel_size: {2, 2}, padding: :valid)
  end

  def sample_pooling_input() do
    Nx.tensor([
      [3, 8, 1, 4],
      [5, 2, 6, -1],
      [-3, 5, 9, 1],
      [4, 5, 7, 2]
    ])
    |> Nx.reshape({1, 4, 4, 1})
  end
end
