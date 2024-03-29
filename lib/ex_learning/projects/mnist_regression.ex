defmodule ExLearning.Projects.MnistRegression do
  alias ExLearning.Core.MnistClassifier
  alias ExLearning.Core.Util

  @train_image_file "train-images-idx3-ubyte.gz"
  @train_label_file "train-labels-idx1-ubyte.gz"
  @test_image_file "t10k-images-idx3-ubyte.gz"
  @test_label_file "t10k-labels-idx1-ubyte.gz"

  def run() do
    train_images = load_train_images()
    train_labels = load_one_hot_encoded_train_labels()
    IO.puts("Loading data done, start training...")
    weight = MnistClassifier.train(train_images, train_labels, 200, 0.00001)
    run_test(weight)
  end

  defp run_test(weight) do
    test_images = load_test_images()
    test_labels = load_test_labels()
    result = MnistClassifier.classify(test_images, weight)

    success =
      Nx.subtract(result, test_labels)
      |> Nx.to_list()
      |> Enum.count(fn x -> x == 0 end)

    IO.puts("Success: #{success}/10000")
  end

  defp load_train_images() do
    load_images_file(@train_image_file)
  end

  defp load_test_images() do
    load_images_file(@test_image_file)
  end

  defp load_images_file(filename) do
    <<_::32, n_images::32, n_rows::32, n_cols::32, images_bin::binary>> =
      read_and_unzip!(filename)

    # 在CUDA机器上面跑这种操作很快
    images_bin
    |> Nx.from_binary({:u, 8})
    |> Nx.reshape({n_images, n_rows * n_cols})
    |> Util.prepend_column(1)
  end

  defp load_one_hot_encoded_train_labels() do
    <<_::32, _n_labels::32, labels::binary>> = read_and_unzip!(@train_label_file)

    labels
    |> :erlang.binary_to_list()
    |> Enum.map(&one_hot_encoded/1)
    |> Nx.tensor()
  end

  @one_hot_encoded_mapping 0..9
                           |> Enum.map(fn n ->
                             {n, List.duplicate(0, n) ++ [1] ++ List.duplicate(0, 9 - n)}
                           end)
                           |> Map.new()

  def one_hot_encoded(n) do
    Map.fetch!(@one_hot_encoded_mapping, n)
  end

  defp load_test_labels() do
    <<_::32, _n_labels::32, labels::binary>> = read_and_unzip!(@test_label_file)

    labels
    |> Nx.from_binary({:u, 8})
  end

  defp read_and_unzip!(filename) do
    filename
    |> File.read!()
    |> :zlib.gunzip()
  end
end
