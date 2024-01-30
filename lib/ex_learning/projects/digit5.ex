defmodule ExLearning.Projects.Digit5 do
  @train_image_file "train-images-idx3-ubyte.gz"
  # @train_label_file "train-labels-idx1-ubyte.gz"
  # @test_image_file "t10k-images-idx3-ubyte.gz"
  # @test_label_file "t10k-labels-idx1-ubyte.gz"

  def load_data() do
  end

  def load_file() do
  end

  def load_train_data() do
    <<_::32, _n_images::32, n_rows::32, n_cols::32, images_bin::binary>> =
      read_and_unzip!(@train_image_file)

    images_bin
    |> image_bin_to_input_tensor(n_rows * n_cols)
    |> Nx.divide(255)
  end

  def image_bin_to_input_tensor(images_bin, image_size) do
    do_images_to_tensor(images_bin, image_size, 255, [])
  end

  defp do_images_to_tensor(<<>>, _, _, tensors) do
    tensors
    |> Enum.reverse()
    |> Nx.concatenate()
  end

  defp do_images_to_tensor(images, size, num, acc) do
    <<image::binary-size(^size), rest::binary>> = images

    t =
      Nx.from_binary(<<num, image::binary>>, {:u, 8})
      |> Nx.reshape({1, size + 1})

    do_images_to_tensor(rest, size, num, [t | acc])
  end

  defp read_and_unzip!(filename) do
    filename
    |> File.read!()
    |> :zlib.gunzip()
  end
end
