defmodule ExLearning.Axon.XOR do
  def build_model(input_shape1, input_shape2) do
    inp1 = Axon.input("num1", shape: input_shape1)
    inp2 = Axon.input("num2", shape: input_shape2)

    # 这里不像之前Mnist只有一个输入来源了
    Axon.concatenate(inp1, inp2)
    |> Axon.dense(8, activation: :tanh)
    |> Axon.dense(1, activation: :sigmoid)
  end

  def train_model(model, data, epochs) do
    model
    |> Axon.Loop.trainer(:binary_cross_entropy, :sgd)
    |> Axon.Loop.run(data, %{}, epochs: epochs, iterations: 1000, compiler: EXLA)
  end
end
