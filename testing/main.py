import typer
import tensorflow as tf
import process_from_camera, process_image, process_video

app = typer.Typer()


def load_model(path: str):
    return tf.keras.models.load_model(path)


@app.command()
def upsample_camera(model: str, video_capture: int = 0, size: str = "800x600"):
    size = tuple(map(int, size.split("x")))
    model = load_model(model)
    process_from_camera.process_from_camera(model, video_capture, size)


@app.command()
def upsample_image(model: str, image_input: str, image_output: str):
    model = load_model(model)
    process_image.process_image(model, image_input, image_output)


@app.command()
def upsample_video(
    model: str, input_path: str, output_path: str, size: str = "2560x1440"
):
    size = tuple(map(int, size.split("x")))
    model = load_model(model)
    process_video.process_video(model, input_path, output_path, size)


if __name__ == "__main__":
    app()
