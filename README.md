# 🚘 Automatic Number Plate Detection

It is a web-based application that detects vehicle number plates from uploaded images. It uses computer vision techniques with OpenCV to identify potential number plate regions and display cropped plate images.

The application is built using Python and Streamlit, providing a simple and user-friendly interface for testing number plate detection.

## ✨ Features

- 📤 Upload vehicle images through a web interface.
- 🔍 Detect potential number plate regions using OpenCV.
- 🖼️ Display detected number plate regions as cropped images.
- 📥 Download detected plate crops.
- 🌐 Access the application through a deployed Streamlit web app.
- ⚡ Image processing using computer vision techniques.

## 🛠️ Technologies Used

- **Python** – Core programming language
- **OpenCV** – Image processing and number plate detection
- **NumPy** – Numerical operations and image array processing
- **Streamlit** – Web application interface

## ⚙️ How It Works

1. The user uploads an image of a vehicle.
2. The application preprocesses the image.
3. Edge detection and morphological operations help identify potential plate regions.
4. Contours and shape-based filtering are used to find likely number plates.
5. The application displays the detected regions and their cropped images.
6. The user can download the detected plate crops.

## 🚀 Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/ranjanrashi/ANPR-repo.git
```

### 2. Open the project folder

```bash
cd ANPR-repo
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run app_final.py
```

The application will open in your browser, usually at:

`http://localhost:8501`

## 📂 Project Structure

```text
ANPR-repo/
├── app_final.py
├── requirements.txt
└── README.md
```

## 🧪 Testing

The application can be tested using vehicle images with visible number plates.

For better results, use images with:
- Clear and readable plate regions
- Good lighting
- Minimal blur
- A visible, unobstructed number plate

Detection accuracy may vary depending on image quality, lighting, viewing angle, and background objects.

## 🔮 Future Improvements

- Integrate OCR to recognize number plate characters.
- Improve detection accuracy under different lighting conditions.
- Support more vehicle types and viewing angles.
- Evaluate detection performance on a larger collection of images.
- Improve the user interface and result visualization.

## ⚠️ Current Limitations

The web app currently focuses on **number plate detection and cropping**.
It does not yet perform optical character recognition (OCR) to convert the detected plate into text.

Its performance depends on the quality and visibility of the uploaded image.

## 🌐 Live Demo

**Try the Live Demo:** (https://anpr-repo-6l5twxyb7zers8nwsb4ezg.streamlit.app/)

## 👩‍💻 Author

**Rashi Ranjan**

GitHub: [@ranjanrashi](https://github.com/ranjanrashi)

---

If you find this project useful, feel free to explore the repository and suggest improvements!

