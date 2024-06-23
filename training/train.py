import typer


app = typer.Typer()


@app.command()
def train_model(
    model_path: str = "../models/", model_name: str = "net-0", epochs: int = 150
):
    import tensorflow as tf
    from tensorflow import keras
    from keras.callbacks import ReduceLROnPlateau
    from utils import train_cache, val_ds, dataset_object
    from model import make_model

    model = make_model(
        lr_shape=(None, None, 3),  # The shape can not have None values
        num_filters=16,
        num_of_residual_blocks_a=2,
    )

    optim_edsr = keras.optimizers.Adam(5e-04, beta_1=0.5)
    tb_callback = keras.callbacks.TensorBoard(
        "./logs", update_freq=1, write_images=True
    )

    learning_rate_reduction = ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=10, min_lr=1e-8, verbose=1
    )

    try:
        model = tf.keras.models.load_model(f"{model_path}{model_name}.keras")
    except ValueError:
        print("Model not found, starting from scratch...")

    # Compiling model with loss as mean absolute error(L1 Loss) and metric as psnr
    model.compile(optimizer=optim_edsr, loss="mae", metrics=["accuracy"])

    # Dataset
    train_ds_0 = dataset_object(train_cache, training=True, hr_crop_size=32)
    train_ds_1 = dataset_object(train_cache, training=True, hr_crop_size=64)
    train_ds_2 = dataset_object(train_cache, training=True, hr_crop_size=128)
    train_ds_3 = dataset_object(train_cache, training=True, hr_crop_size=256)

    train_ds = [train_ds_0, train_ds_1, train_ds_2, train_ds_3]
    train_epochs = [epochs // 3, epochs // 3, epochs // 6, epochs // 15]

    # Training for more epochs will improve results
    for dataset, te in zip(train_ds, train_epochs):
        print(
            "==================== Training ds changed for epochs ===================="
        )
        model.fit(
            dataset,
            epochs=te,
            steps_per_epoch=256,
            validation_data=val_ds,
            callbacks=[tb_callback, learning_rate_reduction],
        )

        tf.keras.Model.save(model, f"{model_path}{model_name}.keras")
        print(f"Model saved in {model_path}{model_name}.keras")

    model.save_weights(f"{model_path}{model_name}.weights.h5")
    print(f"Weights saved in {model_path}{model_name}.weights.h5")


@app.command()
def transform_checkpoint_to_tflite(
    model_full_path: str, model_output_path: str, img_shape: str = "256x256"
):
    # This is a workaround to use the legacy keras API
    import os

    os.environ["TF_USE_LEGACY_KERAS"] = "1"

    import tensorflow as tf
    from model import make_model

    lr_shape = (*tuple(map(int, img_shape.split("x"))), 3)

    # Change this for a real representative dataset
    def representative_data_gen():
        num_calibration_images = 10
        for _ in range(num_calibration_images):
            image = tf.random.normal([1] + list(lr_shape))
            yield [image]

    model = make_model(
        lr_shape=lr_shape,
        num_filters=16,
        num_of_residual_blocks_a=2,
        batch_size=1,
    )

    model.load_weights(model_full_path)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # This enables quantization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # This sets the representative dataset for quantization
    converter.representative_dataset = representative_data_gen
    # This ensures that if any ops can't be quantized, the converter throws an error
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # For full integer quantization, though supported types defaults to int8 only, we explicitly declare it for clarity.
    converter.target_spec.supported_types = [tf.int8]
    # These set the input and output tensors to uint8 (added in r2.3)
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    tflite_model = converter.convert()

    # Save the model.
    with open(model_output_path, "wb") as f:
        f.write(tflite_model)


if __name__ == "__main__":
    app()
