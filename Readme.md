# 🦾 Inside-Yolo

> **YOLOv8n Visualizer — Inside Object Detection (Advanced)**  
> Explore how YOLOv8n "sees" your images step-by-step, visualizing features from input to detection.

---

## 📖 Introduction

**Inside-Yolo** is an educational tool designed to visualize the inner workings of the [YOLOv8n](https://github.com/ultralytics/ultralytics) object detection model. By breaking down the detection pipeline, it reveals how early and middle feature activations evolve, offering insights into edges, textures, and parts recognized by the neural network.  
Ideal for researchers, students, and developers interested in deep learning interpretability.

---

## ✨ Features

- **Step-by-step Visualization**: See how YOLOv8n processes images from input to detection.
- **Feature Activation Maps**: View early (edges/textures) and middle (parts/shapes) activations.
- **Lightweight & Fast**: Uses YOLOv8n 'nano'—optimized for CPU use.
- **Easy-to-Use Script**: Minimal setup; run with a single command.
- **Educational Comments**: Code is documented for learning and extension.

---

## ⚡ Installation

1. **Clone the repository**
    ```bash
    git clone https://github.com/Pranesh-2005/Inside-Yolo.git
    cd Inside-Yolo
    ```

2. **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```
    > **Note:** Make sure you have Python 3.8+ installed.

---

## 🚀 Usage

1. **Prepare your input image**  
   Place your image in the project folder.

2. **Run the visualizer**
    ```bash
    python app.py --image your_image.jpg
    ```

3. **View results**  
   The script will display and/or save visualizations of feature activations and detection outputs.

> **Tip:** Use `--help` to see all available options:
> ```bash
> python app.py --help
> ```

---

## 🤝 Contributing

Contributions, suggestions, and feedback are welcome!

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/my-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/my-feature`)
5. Open a Pull Request

Please follow the [Contributor Covenant](https://www.contributor-covenant.org/) code of conduct.

---

## 📄 License

Distributed under the MIT License.  
See [LICENSE](LICENSE) for more information.

---

> **Made with ❤️ for open-source AI education**



## License
This project is licensed under the **MIT** License.

---
🔗 GitHub Repo: https://github.com/Pranesh-2005/Inside-Yolo