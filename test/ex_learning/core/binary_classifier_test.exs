defmodule ExLearning.Core.BinaryClassifierTest do
  alias ExLearning.Core.BinaryClassifier

  use ExUnit.Case

  describe "sigmoid/1" do
    test "sigmoid(0) = 0.5" do
      assert_in_delta BinaryClassifier.sigmoid(0) |> Nx.to_number(), 0.5, 0.0001
    end

    test "sigmoid(5) == 0.9933" do
      assert Nx.to_number(BinaryClassifier.sigmoid(5)) > 0.99
    end
  end

  describe "forward/2" do
    test "should return a single number" do
      x = Nx.tensor([0.5, 0.8, 0.7])
      w = Nx.tensor([[0.9], [0.3], [0.2]])
      result = BinaryClassifier.forward(x, w)
      assert Nx.shape(result) == {1}
    end
  end
end
