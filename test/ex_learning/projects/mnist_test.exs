defmodule ExLearning.Projects.MnistTest do
  use ExUnit.Case

  alias ExLearning.Projects.Mnist

  describe "one_hot_encoded/1" do
    test "success" do
      assert [1, 0, 0, 0, 0, 0, 0, 0, 0, 0] == Mnist.one_hot_encoded(0)
      assert [0, 0, 0, 1, 0, 0, 0, 0, 0, 0] == Mnist.one_hot_encoded(3)
      assert [0, 0, 0, 0, 0, 0, 0, 0, 0, 1] == Mnist.one_hot_encoded(9)
    end
  end
end
