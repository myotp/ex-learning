defmodule ExLearning.Projects.Digit5Test do
  alias ExLearning.Projects.Digit5
  use ExUnit.Case

  describe "image_bin_to_input_tensor/2" do
    test "正确给图片数据添加255到每一行之前" do
      images_bin = <<1, 2, 3, 4, 5, 6, 7, 8>>
      t = Digit5.image_bin_to_input_tensor(images_bin, 4)

      expected =
        <<255, 1, 2, 3, 4, 255, 5, 6, 7, 8>>
        |> Nx.from_binary({:u, 8})
        |> Nx.reshape({2, 5})

      assert t == expected
    end
  end
end
