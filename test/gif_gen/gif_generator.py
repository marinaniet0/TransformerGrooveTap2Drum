from bokeh.io import output_file, show, save, export_png
from bokeh.models.ranges import Range1d
import os
import sys
import pickle as pk
from selenium import webdriver
import imageio
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
from webdriver_manager.chrome import ChromeDriverManager


# Add sys.path.insert for GrooveEvaluator, preprocessed_dataset and hvo_sequence repos!
from GrooveEvaluator.evaluator import Evaluator

# Manually unzip all your evaluators in ./evaluators

def generate_images():
    driver = webdriver.Chrome(ChromeDriverManager().install())
    train_epochs = []
    test_epochs = []
    train_heatmaps = []
    test_heatmaps = []

    # ############################### #
    # Export Velocity Heatmaps to PNG #
    # ############################### #
    for root, dirs, files in os.walk("evaluators", topdown=False):
        for filename in files:
            if filename.startswith('test'):
                file = open(os.path.join(root, filename), 'rb')
                test_set_evaluator = pk.load(file)
                file.close()
                _gt_heatmaps_dict, _pred_heatmaps_dict = test_set_evaluator.get_logging_dict(
                    velocity_heatmap_html=True, global_features_html=False,
                    piano_roll_html=False, audio_files=False
                )
                velocity_heatmap = _pred_heatmaps_dict["velocity_heatmaps"]
                test_heatmaps.append(velocity_heatmap)
                epoch = filename.split('.')[0].split('_')[-1]
                test_epochs.append(epoch)
            if filename.startswith('train'):
                file = open(os.path.join(root, filename), 'rb')
                train_set_evaluator = pk.load(file)
                file.close()
                _gt_heatmaps_dict, _pred_heatmaps_dict = train_set_evaluator.get_logging_dict(
                    velocity_heatmap_html=True, global_features_html=False,
                    piano_roll_html=False, audio_files=False
                )
                velocity_heatmap = _pred_heatmaps_dict["velocity_heatmaps"]
                train_heatmaps.append(velocity_heatmap)
                epoch = filename.split('.')[0].split('_')[-1]
                train_epochs.append(epoch)

    n_tabs = len(test_heatmaps[0].tabs[0].child.tabs)

    for idx, epoch in enumerate(test_epochs):
        for tab_ix in range(n_tabs):
            test_heatmaps[idx].tabs[0].child.tabs[tab_ix].child.y_range = Range1d(0, 1480)
            test_heatmaps[idx].tabs[0].child.tabs[tab_ix].child.x_range = Range1d(0, 32)
            if not os.path.exists("images"):
                os.mkdir("images")
            export_png(test_heatmaps[idx].tabs[0].child.tabs[tab_ix].child,
                       filename="images/test_vel_heatmap_voice_{}_epoch_{}.png".format(tab_ix, epoch),
                       webdriver=driver)

    for idx, epoch in enumerate(train_epochs):
        for tab_ix in range(n_tabs):
            train_heatmaps[idx].tabs[0].child.tabs[tab_ix].child.y_range = Range1d(0, 1480)
            train_heatmaps[idx].tabs[0].child.tabs[tab_ix].child.x_range = Range1d(0, 32)
            if not os.path.exists("images"):
                os.mkdir("images")
            export_png(train_heatmaps[idx].tabs[0].child.tabs[tab_ix].child,
                       filename="images/train_vel_heatmap_voice_{}_epoch_{}.png".format(tab_ix, epoch),
                       webdriver=driver)
    driver.quit()


def write_epoch_on_images():
    for root, dirs, files in os.walk("images", topdown=False):
        for filename in files:
            img = Image.open(os.path.join(root, filename))
            text = "epoch " + filename.split('.')[0].split("_")[-1]
            draw = ImageDraw.Draw(img)
            font = ImageFont.truetype("Roboto-Regular.ttf", 18)
            draw.text((20, 20), text, (0, 0, 0), font=font)
            new_name = filename.split('.')[0] + "_text.png"
            img.save(os.path.join(root, new_name))


def generate_gifs(delay=0.3):
    # test_vel_heatmap_voice_0_epoch_11_text.png
    test_imgs = {}
    train_imgs = {}
    for root, dirs, files in os.walk("images", topdown=False):
        for filename in files:
            if filename.endswith('_text.png'):
                voice_idx = int(filename.split('_')[-4])
                epoch = int(filename.split('_')[-2])
                if filename.startswith('test'):
                    if voice_idx not in test_imgs:
                        test_imgs[voice_idx] = {}
                    test_imgs[voice_idx][epoch] = os.path.join(root, filename)
                elif filename.startswith('train'):
                    if voice_idx not in train_imgs:
                        train_imgs[voice_idx] = {}
                    train_imgs[voice_idx][epoch] = os.path.join(root, filename)

    for voice in test_imgs:
        test_imgs[voice] = sorted(test_imgs[voice].items())
    for voice in train_imgs:
        train_imgs[voice] = sorted(train_imgs[voice].items())

    for voice in test_imgs:
        imgs = test_imgs[voice]
        with imageio.get_writer('test_set_voice_{}.gif'.format(str(voice)), mode='I', duration=delay) as writer:
            for img in imgs:
                filename = img[1]
                image = imageio.imread(filename)
                writer.append_data(image)

    for voice in train_imgs:
        imgs = train_imgs[voice]
        with imageio.get_writer('train_set_voice_{}.gif'.format(str(voice)), mode='I', duration=delay) as writer:
            for img in imgs:
                filename = img[1]
                image = imageio.imread(filename)
                writer.append_data(image)
        # FIXME: Black stripe at the bottom of the gif

if __name__ == "__main__":
    generate_images()
    write_epoch_on_images()
    generate_gifs()
