defmodule ExLearning.Projects.Digit5Test do
  alias ExLearning.Projects.Digit5
  use ExUnit.Case

  describe "image_bin_to_input_tensor/2" do
    test "正确给图片数据添加1到每一行之前" do
      images_bin = <<65, 66, 67, 68, 75, 76, 77, 78>>
      t = Digit5.image_bin_to_input_tensor(images_bin, 4)

      expected =
        <<1, 65, 66, 67, 68, 1, 75, 76, 77, 78>>
        |> Nx.from_binary({:u, 8})
        |> Nx.reshape({2, 5})

      assert t == expected
    end
  end
end
