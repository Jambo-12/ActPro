


# Distributed Preprocess Video Frames for Act-LLaVA

## Dataset Setup

**PKU-MMD**: Download the raw videos from the official PKU-MMD repository and place them in the specified directory.

**ASTime**: Download the dataset from the official Hugging Face repository ([ASTime](https://huggingface.co/datasets/Jambo1988/ASTime)) and place it in the specified directory. 
You can download it via Git LFS using the following commands:
```bash
# Make sure you have git-lfs installed
git lfs install

# Clone the repository into the dataset directory
git clone https://huggingface.co/datasets/Jambo1988/ASTime dataset/ASTime
```

After downloading, please ensure your dataset directory structure is organized as follows:
```
└── dataset/
    ├── PKUMMD/
    │   └── annotations
    │   └── videos/           
    │       ├── 0001-L.mp4
    │       ├── 0001-R.mp4
    │       └── ...
    └── ASTime/            
        ├── annotations
        └── videos
```

## Sample video frames to 2 FPS

```
python -m data.preprocess.ffmpeg --frame_fps 2 --frame_resolution 384 --crop_center True --num_tasks 8 --num_nodes 1 --video_dir datasets/PKUMMD/videos
```

- Please run the script in ```Act-LLaVA/``` root folder.

- The results will be saved in a new folder with '{fps}fps' suffix. For example, ```datasets/PKUMMD/videos -> datasets/PKUMMD/videos_2fps```.

- Increase ```--num_tasks``` according to the CPU cores. 1/10 number of CPU cores is recommended.

## Encode sampled video frames

```
python -m data.preprocess.extract_feature --video_dir dataset/PKUMMD/videos_2fps
```

- Please run the script in ```Act-LLaVA/``` root folder.

- The results will be saved in a new folder ```features```. For example, ```dataset/PKUMMD/videos_2fps -> dataset/PKUMMD/features```.

- If you are on a cluster, you can set ```--num_nodes ... --slurm_partition ...``` to use them. The more nodes and GPUs, the faster preprocessing.