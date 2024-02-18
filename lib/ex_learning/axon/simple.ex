defmodule ExLearning.Axon.Simple do
  def build_simple() do
    Axon.input("input", shape: {nil, 6})
    |> Axon.dense(3, activation: :relu)
    |> Axon.dense(3, activation: :softmax)
  end

  def build_simple_no_softmax() do
    Axon.input("input", shape: {nil, 6})
    |> Axon.dense(3, activation: :relu)
    |> Axon.dense(3)
  end

  def build_simple_only_softmax(model) do
    # TODO
    softmax(model)
  end
end
