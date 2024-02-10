defmodule ExLearning.Axon.MnistCNN do
  def build_model(input_shape) do
    # input layer -> hidden layer -> output layer
    # 中间层用sigmoid最后层用softmax
    # 不用自己再去考虑bias的问题
    Axon.input("input", shape: input_shape)
    |> Axon.conv(32, kernel_size: {3, 3}, padding: :same, activation: :relu)
    |> Axon.max_pool(kernel_size: {2, 2}, padding: :same)
    |> Axon.conv(64, kernel_size: {3, 3}, padding: :same, activation: :relu)
    |> Axon.max_pool(kernel_size: {2, 2}, padding: :same)
    |> Axon.conv(128, kernel_size: {3, 3}, padding: :same, activation: :relu)
    |> Axon.max_pool(kernel_size: {2, 2}, padding: :same)
    |> Axon.flatten()
    |> Axon.dense(10, activation: :softmax)
  end

  def train_model(model, train_images, train_lables, epochs) do
    model
    |> Axon.Loop.trainer(
      :categorical_cross_entropy,
      Polaris.Optimizers.adamw(learning_rate: 0.005)
    )
    |> Axon.Loop.metric(:accuracy, "Accuracy")
    |> Axon.Loop.run(Stream.zip(train_images, train_lables), %{}, epochs: epochs, compiler: EXLA)
  end

  def test_model(model, model_state, test_images, test_labels) do
    model
    |> Axon.Loop.evaluator()
    |> Axon.Loop.metric(:accuracy, "Accuracy")
    |> Axon.Loop.run(Stream.zip(test_images, test_labels), model_state, compiler: EXLA)
  end
end
