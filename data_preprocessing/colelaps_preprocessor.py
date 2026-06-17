import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


class ColelapsPreprocessor:

    def __init__(self, cholec80_path, dest_path, frames_index_path):
        self.cholec80_path = cholec80_path
        self.dest_path = dest_path
        self.frames_index_path = frames_index_path

    def video_to_frames(self, video_path, final_frame):
        """
        Convert the video to frames
        :param video_path: String with the path of the video
        :param labels_path: String with the path of the video labels
        :param frames_path: String with the path where must be located the frames images
        :return: list with all the frame paths
        """
        video_tag = os.path.basename(os.path.normpath(video_path)).split('.')[0]
        print("Converting " + video_tag + " ...")

        # Take the labeled frames
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {video_path}")

        p_bar = tqdm(total=final_frame + 1)
        i = 0
        while cap.isOpened() and i <= final_frame:
            ret, frame = cap.read()
            if not ret:
                break
            file_path = os.path.join(self.dest_path, video_tag)
            os.makedirs(file_path, exist_ok=True)
            file_name = file_path + "/" + str(i) + ".jpg"
            new_frame = self.cut_frame(frame)
            cv2.imwrite(file_name, new_frame)
            i += 1
            p_bar.update(1)
        p_bar.close()
        cap.release()

        return None

    def cut_frame(self, frame):
        """
        Cut the black mask
        :param frame: frame
        :return: cropped frame
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        threshold_level = 10
        black_pixels = np.column_stack(np.where(gray > threshold_level))
        if black_pixels.shape[0] > 160 and black_pixels.shape[1] > 160:
            y_min_cut = min(black_pixels[:, 0])
            y_max_cut = max(black_pixels[:, 0])
            x_min_cut = min(black_pixels[:, 1])
            x_max_cut = max(black_pixels[:, 1])

            new_frame = frame[y_min_cut:y_max_cut, x_min_cut:x_max_cut]

        else:
            new_frame = frame

        return new_frame

    def process_videos(self, video_list):
        frames_index = pd.read_csv(self.frames_index_path)
        for video_name in video_list:
            final_frame = frames_index[frames_index["video_name"] == video_name]["final_frame"].values[0]
            video_frames_path = os.path.join(self.dest_path, video_name)
            expected_frame_count = final_frame + 1
            final_frame_path = os.path.join(video_frames_path, f"{final_frame}.jpg")

            if os.path.exists(final_frame_path):
                saved_frame_count = len([name for name in os.listdir(video_frames_path) if name.endswith(".jpg")])
                if saved_frame_count >= expected_frame_count:
                    print(f"Skipping {video_name}: {saved_frame_count} frames already exist")
                    continue

            self.video_to_frames(os.path.join(self.cholec80_path, f"videos/{video_name}.mp4"), final_frame)
