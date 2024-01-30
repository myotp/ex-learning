defmodule ExLearning.Projects.Digit5 do
  alias ExLearning.Core.BinaryClassifier

  @train_image_file "train-images-idx3-ubyte.gz"
  @train_label_file "train-labels-idx1-ubyte.gz"
  @test_image_file "t10k-images-idx3-ubyte.gz"
  @test_label_file "t10k-labels-idx1-ubyte.gz"

  def run() do
    train_images = load_train_images()
    train_labels = load_train_labels()
    IO.puts("Loading data done, start training...")
    weight = BinaryClassifier.train(train_images, train_labels, 100, 0.00001)
    weight
  end

  def run_test(weight) do
    test_images = load_test_images()
    test_labels = load_test_labels()
    result = BinaryClassifier.forward(test_images, weight)

    Nx.subtract(result, test_labels)
    |> Nx.abs()
    |> Nx.sum()
    |> Nx.to_number()
    |> floor()
  end

  def load_train_images() do
    load_images_file(@train_image_file)
  end

  def load_train_labels() do
    load_labels_file(@train_label_file)
  end

  def load_test_images() do
    load_images_file(@test_image_file)
  end

  def load_test_labels() do
    load_labels_file(@test_label_file)
  end

  defp load_labels_file(filename) do
    <<_::32, _n_labels::32, labels::binary>> = read_and_unzip!(filename)

    labels
    |> :erlang.binary_to_list()
    |> Stream.map(&normalize_num_5/1)
    |> Stream.map(fn n -> [n] end)
    |> Enum.to_list()
    |> Nx.tensor()
  end

  # MNIST先简单做二分判断是否为5的特例
  defp normalize_num_5(5), do: 1
  defp normalize_num_5(_), do: 0

  defp load_images_file(filename) do
    <<_::32, _n_images::32, n_rows::32, n_cols::32, images_bin::binary>> =
      read_and_unzip!(filename)

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
      prepend_bias_column(image, num)
      |> Nx.from_binary({:u, 8})
      |> Nx.reshape({1, size + 1})

    do_images_to_tensor(rest, size, num, [t | acc])
  end

  defp prepend_bias_column(image_bin, num) do
    <<num, image_bin::binary>>
  end

  defp read_and_unzip!(filename) do
    filename
    |> File.read!()
    |> :zlib.gunzip()
  end
end
