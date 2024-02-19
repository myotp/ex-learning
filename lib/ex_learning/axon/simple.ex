defmodule ExLearning.Axon.Simple do
  # 简单模型, 最后输出层带有softmax
  def build_simple() do
    Axon.input("input", shape: {nil, 6})
    |> Axon.dense(3, activation: :relu)
    |> Axon.dense(3, activation: :softmax)
  end

  # 最后输出层不带softmax
  def build_simple_no_softmax() do
    Axon.input("input", shape: {nil, 6})
    |> Axon.dense(3, activation: :relu)
    |> Axon.dense(3)
  end

  # 单独只对最后输出层做softmax
  def extend_softmax(model) do
    Axon.activation(model, :softmax)
  end

  def simple_input() do
    Axon.input("input", shape: {nil, 3})
  end

  # similar to y=m*x+b
  def simple_linear() do
    Axon.input("input", shape: {nil, 1})
    |> Axon.dense(1)
  end
end
