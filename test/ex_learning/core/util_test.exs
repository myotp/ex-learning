defmodule ExLearning.Core.UtilTest do
  alias ExLearning.Core.Util
  use ExUnit.Case

  describe "append_column/2" do
    test "测试使用API正确" do
      images_tensor = Nx.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])

      assert Nx.tensor([[255, 1, 2, 3, 4], [255, 5, 6, 7, 8]]) ==
               Util.append_column(images_tensor, 255)
    end
  end

  describe "zeros/2" do
    test "success" do
      assert Nx.tensor([[0, 0, 0, 0], [0, 0, 0, 0]]) == Util.zeros(2, 4)
    end
  end
end
