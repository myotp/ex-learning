defmodule ExLearning.Projects.MnistCnn do
  alias ExLearning.Axon.MnistCNN

  @train_image_file "train-images-idx3-ubyte.gz"
  @train_label_file "train-labels-idx1-ubyte.gz"
  @test_image_file "t10k-images-idx3-ubyte.gz"
  @test_label_file "t10k-labels-idx1-ubyte.gz"

  @label_values Enum.to_list(0..9)
  @batch_size 64

  def run() do
    model = build_model()
    model_state = train_model(model)
    test_model(model, model_state)
  end

  def build_model() do
    # input_shape = {N, 784} 没有bias的问题了
    MnistCNN.build_model({nil, 28, 28, 1})
  end

  def train_model(model) do
    train_images = load_train_images()
    train_labels = load_train_labels()
    MnistCNN.train_model(model, train_images, train_labels, 10)
  end

  def test_model(model, model_state) do
    test_images = load_test_images()
    test_labels = load_test_labels()
    MnistCNN.test_model(model, model_state, test_images, test_labels)
  end

  # 输入(60000, 28, 28, 1)
  def load_train_images() do
    load_images_file(@train_image_file)
  end

  def load_test_images() do
    load_images_file(@test_image_file)
  end

  # 最终结果(60000, 10)
  def load_train_labels() do
    load_lables_file(@train_label_file)
  end

  def load_test_labels() do
    load_lables_file(@test_label_file)
  end

  def load_images_file(filename) do
    <<_::32, n_images::32, n_rows::32, n_cols::32, images_bin::binary>> =
      read_and_unzip!(filename)

    # 在CUDA机器上面跑这种操作很快
    images_bin
    |> Nx.from_binary({:u, 8})
    |> Nx.reshape({n_images, n_rows, n_cols, 1})
    |> Nx.divide(255)
    |> Nx.to_batched(@batch_size)
  end

  def load_lables_file(filename) do
    <<_::32, n_labels::32, labels::binary>> = read_and_unzip!(filename)

    labels
    |> Nx.from_binary({:u, 8})
    |> Nx.reshape({n_labels, 1})
    # one-hot encoded
    |> Nx.equal(Nx.tensor(@label_values))
    |> Nx.to_batched(@batch_size)
  end

  def read_and_unzip!(filename) do
    filename
    |> File.read!()
    |> :zlib.gunzip()
  end
end
