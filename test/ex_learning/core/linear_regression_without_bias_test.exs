defmodule ExLearning.Core.LinearRegressionWithoutBiasTest do
  use ExUnit.Case

  alias ExLearning.Core.LinearRegressionWithoutBias

  test "predict/2" do
    assert LinearRegressionWithoutBias.predict(Nx.tensor([1, 2]), 2) == Nx.tensor([2, 4])
  end

  describe "loss/3 (MSE)" do
    test "MSE: errors" do
      x = Nx.tensor([1, 2])
      y = Nx.tensor([2, 4])
      loss = LinearRegressionWithoutBias.loss(x, y, 2) |> Nx.to_number()
      assert_in_delta loss, 0, 0.000001
    end

    test "MSE: squared" do
      x = Nx.tensor([1, 2])
      y = Nx.tensor([4, 6])
      loss = LinearRegressionWithoutBias.loss(x, y, 2) |> Nx.to_number()
      assert_in_delta loss, 4, 0.000001
    end

    test "MSE: mean" do
      x = Nx.tensor([1, 2])
      y = Nx.tensor([4, 10])
      loss = LinearRegressionWithoutBias.loss(x, y, 2) |> Nx.to_number()
      assert_in_delta loss, 20, 0.000001
    end
  end
end
