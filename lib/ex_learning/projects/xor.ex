defmodule ExLearning.Projects.XOR do
  alias ExLearning.Axon.XOR

  @batch_size 32

  def run() do
    model = XOR.build_model({nil, 1}, {nil, 1})
    data = Stream.repeatedly(&load_data/0)
    params = XOR.train_model(model, data, 10)
    test_model(model, params)
  end

  def test_model(model, params) do
    t1 = Nx.tensor([[0], [0], [1], [1]])
    t2 = Nx.tensor([[0], [1], [0], [1]])
    Axon.predict(model, params, %{"num1" => t1, "num2" => t2})
  end

  defp load_data() do
    x1 = Nx.tensor(for _ <- 1..@batch_size, do: [Enum.random(0..1)])
    x2 = Nx.tensor(for _ <- 1..@batch_size, do: [Enum.random(0..1)])
    y = Nx.logical_xor(x1, x2)
    {%{"num1" => x1, "num2" => x2}, y}
  end
end
