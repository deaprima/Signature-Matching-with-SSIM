import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

class SignatureVerifier:
    """Logika utama verifikasi tanda tangan"""
    def __init__(self):
        self.reference_img = None  
        self.test_img = None  
        self.threshold = 0.6  # ambang batas default SSIM (nilai kemiripan)

    def process_image(self, image_path):
        """Auto-crop dan preprocessing gambar tanda tangan"""
        try:
            img = cv2.imread(image_path)  
            if img is None:
                raise ValueError("File gambar tidak valid")

            # konversi ke grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # binarisasi (mengubah jadi biner hitam & putih) dengan metode Otsu Tresholding
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            # mencari kontur dari objek (garis luar tanda tangan)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                raise ValueError("Tidak ditemukan tanda tangan dalam gambar")

            # menggabungkan semua kontur dan cari bounding box (kotak luar)
            all_contours = np.vstack(contours)
            x, y, w, h = cv2.boundingRect(all_contours)

            # menambahkan padding agar tanda tangan tidak terlalu mepet
            padding = 15
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(img.shape[1] - x, w + 2 * padding)
            h = min(img.shape[0] - y, h + 2 * padding)
            cropped = img[y:y + h, x:x + w]  # crop sesuai bounding box

            # resize ke ukuran tetap 300x150
            resized = cv2.resize(cropped, (300, 150))

            # ubah ulang ke grayscale & binarize
            processed = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            _, processed = cv2.threshold(processed, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            return processed

        except Exception as e:
            raise ValueError(f"Image processing failed: {str(e)}")


    def verify_signature(self):
        """Bandingkan tanda tangan menggunakan SSIM"""
        if self.reference_img is None or self.test_img is None:
            raise ValueError("Kedua gambar harus dimuat dulu")

        score = ssim(self.reference_img, self.test_img)  # menghitung kemiripan SSIM
        similarity_percentage = round(score * 100, 2)  # mengubah ke persen
        return similarity_percentage, (score >= self.threshold)


class SignatureVerificationUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Signature Verification System")  
        self.root.geometry("1000x800")  

        self.verifier = SignatureVerifier()  
        self.setup_ui() 

    def setup_ui(self):
        self.setup_title()
        self.setup_threshold_control()
        self.setup_image_displays()
        self.setup_processed_images()
        self.setup_results_section()
        self.setup_verify_button()

    def setup_title(self):
        tk.Label(self.root, text="SIGNATURE VERIFICATION SYSTEM", font=("Arial", 18, "bold")).pack(pady=10)

    def setup_threshold_control(self):
        tk.Label(self.root, text="SSIM Threshold:").pack()
        self.threshold_slider = tk.Scale(
            self.root, from_=0.1, to=1.0, resolution=0.05,
            orient=tk.HORIZONTAL, command=self.update_threshold
        )
        self.threshold_slider.set(0.6)
        self.threshold_slider.pack()

    def setup_image_displays(self):
        frame_images = tk.Frame(self.root)
        frame_images.pack(pady=20)

        # Reference
        frame_ref = tk.Frame(frame_images, bd=2, relief=tk.GROOVE)
        frame_ref.grid(row=0, column=0, padx=20)
        tk.Label(frame_ref, text="Reference Signature", font=("Arial", 12)).pack()
        self.canvas_ref = tk.Canvas(frame_ref, width=400, height=200, bg='white')
        self.canvas_ref.pack()
        tk.Button(frame_ref, text="Load Reference", command=lambda: self.load_image("reference")).pack(pady=5)

        # Test
        frame_test = tk.Frame(frame_images, bd=2, relief=tk.GROOVE)
        frame_test.grid(row=0, column=1, padx=20)
        tk.Label(frame_test, text="Test Signature", font=("Arial", 12)).pack()
        self.canvas_test = tk.Canvas(frame_test, width=400, height=200, bg='white')
        self.canvas_test.pack()
        tk.Button(frame_test, text="Load Test", command=lambda: self.load_image("test")).pack(pady=5)

    def setup_processed_images(self):
        frame_processed = tk.Frame(self.root)
        frame_processed.pack(pady=10)

        tk.Label(frame_processed, text="Preprocessed Images", font=("Arial", 12)).pack()
        self.canvas_processed_ref = tk.Canvas(frame_processed, width=300, height=150, bg='white')
        self.canvas_processed_ref.pack(side=tk.LEFT, padx=10)
        self.canvas_processed_test = tk.Canvas(frame_processed, width=300, height=150, bg='white')
        self.canvas_processed_test.pack(side=tk.LEFT, padx=10)

    def setup_results_section(self):
        frame_result = tk.Frame(self.root, bd=2, relief=tk.GROOVE)
        frame_result.pack(pady=20, fill=tk.X, padx=50)

        self.label_ssim = tk.Label(frame_result, text="Similarity: -", font=("Arial", 12))
        self.label_ssim.pack()

        self.label_result = tk.Label(frame_result, text="Result: -", font=("Arial", 14, "bold"))
        self.label_result.pack(pady=10)

    def setup_verify_button(self):
        tk.Button(
            self.root, text="VERIFY SIGNATURE", command=self.verify_signatures,
            bg="#4CAF50", fg="white", font=("Arial", 12, "bold")
        ).pack(pady=20)

    def update_threshold(self, value):
        self.verifier.threshold = float(value)

    def load_image(self, img_type):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
        if not path:
            return

        try:
            processed_img = self.verifier.process_image(path)

            if img_type == "reference":
                self.verifier.reference_img = processed_img
                self.display_image(path, self.canvas_ref)
                self.show_processed_image(processed_img, self.canvas_processed_ref)
            else:
                self.verifier.test_img = processed_img
                self.display_image(path, self.canvas_test)
                self.show_processed_image(processed_img, self.canvas_processed_test)

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def display_image(self, path, canvas):
        img = Image.open(path)
        img = img.resize((400, 200), Image.Resampling.LANCZOS)
        img_tk = ImageTk.PhotoImage(img)
        canvas.image = img_tk
        canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)

    def show_processed_image(self, img, canvas):
        if img is not None:
            img_pil = Image.fromarray(img)
            img_pil = img_pil.resize((300, 150), Image.Resampling.LANCZOS)
            img_tk = ImageTk.PhotoImage(img_pil)
            canvas.image = img_tk
            canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)

    def verify_signatures(self):
        if self.verifier.reference_img is None or self.verifier.test_img is None:
            messagebox.showwarning("Warning", "Please load both images first!")
            return

        try:
            similarity, is_valid = self.verifier.verify_signature()
            # menampilkan persentase kemiripan dan nilai SSIM
            self.label_ssim.config(text=f"Similarity: {similarity:.2f}%")
            self.label_result.config(text=f"SSIM Value: {similarity / 100:.4f}")  # menampilkan nilai SSIM (antara 0 dan 1)

            # menambahkan pengecekan threshold untuk menampilkan validitas
            if similarity / 100 >= self.verifier.threshold:
                self.label_result.config(text=f"✅ SIGNATURE MATCHES", fg="green")
            else:
                self.label_result.config(text=f"❌ SIGNATURE DOES NOT MATCH", fg="red")

        except Exception as e:
            messagebox.showerror("Error", f"Verification failed: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = SignatureVerificationUI(root)
    root.mainloop()
