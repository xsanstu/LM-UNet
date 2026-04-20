import h5py
import numpy as np
from tqdm import tqdm

from root import ROOT_DIR


def create_dataset(input_length: int, image_ahead: int, rain_amount_thresh: float):
    """Create a dataset that has target images containing at least `rain_amount_thresh` (percent) of rain."""

    # precipitation_folder = ROOT_DIR / "data" / "precipitation"
    precipitation_folder = "C:/Users/Hu/Desktop/所有文件/南信大/科研/0dataset/KNMI"
    with h5py.File(
        precipitation_folder + "/RAD_NL25_RAC_5min_train_test_2016-2019.h5",
        "r",
    ) as orig_f:
        train_images = orig_f["train"]["images"]            # 314958*288*288
        train_timestamps = orig_f["train"]["timestamps"]    # 314958*1
        test_images = orig_f["test"]["images"]              # 105021*288*288
        test_timestamps = orig_f["test"]["timestamps"]      # 105021*1
        print("Train shape", train_images.shape)
        print("Test shape", test_images.shape)
        imgSize = train_images.shape[1]                     # 288
        num_pixels = imgSize * imgSize                      # 288*288

        filename = (
            precipitation_folder+ "/" + f"train_test_2016-2019_input-length_{input_length}_img-"
            + f"ahead_{image_ahead}_rain-threshold_{int(rain_amount_thresh * 100)}.h5"
        )

        with h5py.File(filename, "w") as f:
            train_set = f.create_group("train")
            test_set = f.create_group("test")
            train_image_dataset = train_set.create_dataset(
                "images",
                shape=(1, input_length + image_ahead, imgSize, imgSize),
                maxshape=(None, input_length + image_ahead, imgSize, imgSize),
                dtype="float32",
                compression="gzip",
                compression_opts=9,
            )
            train_timestamp_dataset = train_set.create_dataset(
                "timestamps",
                shape=(1, input_length + image_ahead, 1),
                maxshape=(None, input_length + image_ahead, 1),
                dtype=h5py.special_dtype(vlen=str),
                compression="gzip",
                compression_opts=9,
            )
            test_image_dataset = test_set.create_dataset(
                "images",
                shape=(1, input_length + image_ahead, imgSize, imgSize),
                maxshape=(None, input_length + image_ahead, imgSize, imgSize),
                dtype="float32",
                compression="gzip",
                compression_opts=9,
            )
            test_timestamp_dataset = test_set.create_dataset(
                "timestamps",
                shape=(1, input_length + image_ahead, 1),
                maxshape=(None, input_length + image_ahead, 1),
                dtype=h5py.special_dtype(vlen=str),
                compression="gzip",
                compression_opts=9,
            )

            origin = [[train_images, train_timestamps], [test_images, test_timestamps]]
            datasets = [
                [train_image_dataset, train_timestamp_dataset],
                [test_image_dataset, test_timestamp_dataset],
            ]
            # 将一个可遍历的数据对象（如列表、字符串）组合为一个索引序列，同时列出数据和数据下标。这样，在遍历数据时，可以同时获得索引和数据
            for origin_id, (images, timestamps) in enumerate(origin):
                image_dataset, timestamp_dataset = datasets[origin_id]
                first = True
                for i in tqdm(range(input_length + image_ahead, len(images))):
                    # If threshold of rain is bigger in the target image: add sequence to dataset
                    if np.sum(images[i] > 0) >= num_pixels * rain_amount_thresh:  # 当前的图中降雨大于阈值
                        imgs = images[i - (input_length + image_ahead) : i]       # 该图前18个连续的图
                        timestamps_img = timestamps[i - (input_length + image_ahead) : i]  # 该图前18图的时间戳
                        #                     print(imgs.shape)
                        #                     print(timestamps_img.shape)
                        # extend the dataset by 1 and add the entry
                        # 第一个已经在矩阵中分配了空间，其余没有所以如果不是第一个需要调正shape，即【0】维度+1
                        if first:
                            first = False
                        else:
                            image_dataset.resize(image_dataset.shape[0] + 1, axis=0)
                            timestamp_dataset.resize(timestamp_dataset.shape[0] + 1, axis=0)
                        image_dataset[-1] = imgs                # 在image_dataset最后赋值18*288*288
                        timestamp_dataset[-1] = timestamps_img  # timestamp_dataset最后赋值18*1
                        # image_dataset中每一个id存放的是一个样本即18个图像序列

if __name__ == "__main__":
    print("Creating dataset with at least 20% of rain pixel in target image")
    create_dataset(input_length=12, image_ahead=6, rain_amount_thresh=0.2)
    print("Creating dataset with at least 50% of rain pixel in target image")
    create_dataset(input_length=12, image_ahead=6, rain_amount_thresh=0.5)
