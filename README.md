# POSEIDON  
**POSe Estimation with Intelligent Deep Object Networks**

POSEIDON is a 6D object pose estimation system inspired by NVIDIA's DOPE architecture. It is designed to detect and estimate the full 6D pose (translation + rotation) of rigid objects from RGB images using deep belief and affinity map predictions. This system is optimized for both real-world robotics applications and synthetic data training pipelines.

## 🧠 Key Features
- **Deep CNN architecture** trained on synthetic and/or real data
- **Belief Maps + Affinity Fields** to detect object keypoints and centroids
- **PnP-based pose solving** from 2D-3D correspondences
- Modular dataset support with **NDDS-style synthetic data integration**
- ROS and real-time inference ready
- Configurable for multiple object types and camera intrinsics

---

## 📦 Installation

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/poseidon.git
cd poseidon
```

### 2. Create a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## 🗂 Project Structure

```
poseidon/
├── configs/                # Network and dataset configuration files
├── data/                   # Synthetic and real dataset loaders (NDDS-style supported)
├── models/                 # Pose estimation networks (belief/affinity map CNNs)
├── pose_estimation/        # Keypoint extraction + PnP pose solver
├── utils/                  # Visualization, metrics, and camera utilities
├── inference/              # Inference scripts and ROS integration
├── train/                  # Training entry point and training utilities
├── checkpoints/            # Pretrained model weights
└── README.md
```

---

## 🏁 Quick Start

### Inference on an RGB image:
```bash
python inference/run_poseidon.py --config configs/ycbv.yaml --image_path sample.jpg
```

### Training with NDDS dataset:
```bash
python train/train_poseidon.py --config configs/ycbv.yaml
```

---

## 🧪 Supported Datasets

- **NDDS (NVIDIA Deep Dataset Synthesizer)**-style datasets  
- **YCB-Video** (with conversion scripts available)
- Custom datasets (keypoints + object CAD required)

---

## 📸 Output Example

Given an input image, POSEIDON predicts:
- Belief maps for object keypoints
- Affinity fields connecting keypoints to the object center
- Reconstructed 3D pose using PnP (or deep PnP)

![example_output](docs/output_sample.png)

---

## 🤖 ROS Integration

To enable real-time inference with ROS:
```bash
roslaunch poseidon_ros poseidon.launch
```

Requirements: ROS Noetic + image_pipeline + cv_bridge

---

## 🛠 Configuration

Each object class is defined by:
- 3D keypoints on CAD model
- Camera intrinsics
- Model training config (YAML)

Edit YAML files under `configs/` for:
- Dataset paths
- Network architecture
- Training hyperparameters

---

## 🧠 Model Architecture

POSEIDON builds on a multi-stage hourglass-like architecture:
- **Feature extraction** via ResNet-18 or ResNet-34
- **Belief map heads** for visible keypoints
- **Affinity map heads** to link keypoints to centroids
- Non-maximum suppression and PnP for pose recovery

---

## 📈 Evaluation

POSEIDON supports standard 6D metrics:
- ADD(-S) for symmetric/asymmetric objects
- 2D projection error
- Visible surface discrepancy (VSD)

```bash
python evaluation/evaluate_pose.py --pred_dir outputs/ --gt_dir ground_truth/
```

---

## 📚 Citation

If you use POSEIDON in your work, please cite:

```bibtex
@misc{poseidon2025,
  title={POSEIDON: POSe Estimation with Intelligent Deep Object Networks},
  author={Your Name},
  year={2025},
  howpublished={\url{https://github.com/yourusername/poseidon}},
}
```

---

## 🧩 Related Work

- [DOPE: Deep Object Pose Estimation](https://github.com/NVlabs/Deep_Object_Pose)
- [PVN3D](https://github.com/ethnhe/PVN3D)
- [GDR-Net](https://github.com/THU-DA-YoungSubGroup/GDR-Net)

---

## 📬 Contact

For questions or collaborations, contact:  
**Your Name** – [your.email@example.com](mailto:your.email@example.com)

---

Would you like a minimal website or documentation hub (e.g., MkDocs or GitHub Pages) to go with this?