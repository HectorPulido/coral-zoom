# CORAL ZOOM PROJECT

This project leverages the power of the Coral TPU and deep learning models to super sample video game frames. The goal is to enhance the visual quality of games and FPS by upscaling lower resolution frames to higher resolutions, resulting in sharper and more detailed visuals.

The goal is to render a game in 800x600 resolution and upscale it to 1920x1080 or higher resolutions in real-time, of course if you have a potato PC, you can't run big deep learning models, that's why we use the Coral TPU to accelerate the process.

The project is based on the EDSR model but little modifications were made improving the performance and the quality of the model.

The project is divided into Four main parts:
1. Training: Train your own models using the training_edsr.ipynb notebook or the training/train.py script. ✅
2. Testing: Test the Coral Zoom project on your game using the testing/main.py script. ✅
3. Coral Zoom: The main project that uses the Coral TPU to run inference on the models. [THIS IS STILL IN PROGRESS 🕓]
4. GUI: Final project that can be executed and used by the user to upscale the game. [THIS WILL BE DONE LATER ❌]


>[!WARNING]
> This is a highly experimental project in progress, and the results may vary depending on the game and the model used. The project is not intended for production use. AND IT IS NOT READY FOR USE.

## Key Features
Coral TPU Acceleration: Utilizes the Coral TPU for efficient inference.
Deep Learning Models: Employs the Enhanced Deep Super-Resolution (EDSR) architecture for high-quality upscaling.
Flexible Testing: Supports processing images, videos, and live camera feeds.
Customizable: Train your own models for optimal results on specific games.

## Getting Started
## 0. Install Dependencies:
```bash
python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

Getting started with the coral tpu information can be found here:
https://coral.ai/docs/accelerator/get-started/


## 1. (Optional) Train Models:
Use the training_edsr.ipynb notebook to train your own EDSR models.
Or if you prefer use the training/train.py script to train your own models.

Adapt the training process to your specific game or hardware.

You can see the available options by running:
```
python training/train.py --help
```

## 2. Run Tests projects
Use the testing/main.py script to test the Coral Zoom project on your game. There are several examples of the use of the models:
* Process from camera input
```
python testing/main.py main.py upsample-camera '<your model path here>'
```

* Process from video file
```
python testing/main.py main.py upsample-video '<your model path here>' '<your video path here>' '<output video path here>' --size "1920x1080"
```

* Process from image file
```
python testing/main.py main.py upsample-image '<your model path here>' '<your image path here>' '<output image path here>'
```

## THIS PROJECT IS A BIG TODO LIST, SO IF YOU WANT TO CONTRIBUTE, PLEASE DO IT, WE NEED YOUR HELP TO MAKE THIS PROJECT A REALITY.



## Contributing

Your contributions are greatly appreciated! Please follow these steps:

1. Fork the project
2. Create your feature branch `git checkout -b feature/MyFeature`
3. Commit your changes `git commit -m "my cool feature"`
4. Push to the branch `git push origin feature/MyFeature`
5. Open a Pull Request

## License

Every code made by me in this repo is under the MIT license, but the models, the data used, and the hardware in this project are under their respective licenses.

## Contact

<hr>
<div align="center">
<h3 align="center">Let's connect 😋</h3>
</div>
<p align="center">
<a href="https://www.linkedin.com/in/hector-pulido-17547369/" target="blank">
<img align="center" width="30px" alt="Hector's LinkedIn" src="https://www.vectorlogo.zone/logos/linkedin/linkedin-icon.svg"/></a> &nbsp; &nbsp;
<a href="https://twitter.com/Hector_Pulido_" target="blank">
<img align="center" width="30px" alt="Hector's Twitter" src="https://www.vectorlogo.zone/logos/twitter/twitter-official.svg"/></a> &nbsp; &nbsp;
<a href="https://www.twitch.tv/hector_pulido_" target="blank">
<img align="center" width="30px" alt="Hector's Twitch" src="https://www.vectorlogo.zone/logos/twitch/twitch-icon.svg"/></a> &nbsp; &nbsp;
<a href="https://www.youtube.com/channel/UCS_iMeH0P0nsIDPvBaJckOw" target="blank">
<img align="center" width="30px" alt="Hector's Youtube" src="https://www.vectorlogo.zone/logos/youtube/youtube-icon.svg"/></a> &nbsp; &nbsp;
<a href="https://pequesoft.net/" target="blank">
<img align="center" width="30px" alt="Pequesoft website" src="https://github.com/HectorPulido/HectorPulido/blob/master/img/pequesoft-favicon.png?raw=true"/></a> &nbsp; &nbsp;
