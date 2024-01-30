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
end
