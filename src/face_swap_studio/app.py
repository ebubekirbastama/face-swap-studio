"""
Face Swap Studio - Modern CPU GUI (InsightFace)
================================================

Statik fotograflarda InsightFace / inswapper_128 kullanarak yuz degistirir.
Tamamen CPU ile calisacak sekilde hazirlanmistir.

Kurulum:
    pip install insightface onnxruntime opencv-python pillow numpy

Not:
- 1080p / 4K cikti secenegi kayit sirasinda yuksek kaliteli yeniden boyutlandirma yapar.
- Dusuk cozunurluklu bir fotografi 4K'ya buyutmek yeni gercek detay uretmez; sadece
  daha buyuk piksel boyutunda dosya olusturur.
"""

from __future__ import annotations

import logging
import os
import tempfile
import threading
import urllib.request
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageOps, ImageTk

from . import __version__

APP_NAME = "Face Swap Studio"
APP_USER_AGENT = f"FaceSwapStudio/{__version__}"
MIN_MODEL_BYTES = 1_000_000
LOG_DIR = Path.home() / ".face_swap_studio"
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    filename=LOG_DIR / "app.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
LOGGER = logging.getLogger(APP_NAME)


# -----------------------------------------------------------------------------
# Model kaynaklari
# -----------------------------------------------------------------------------
HF_MODEL_URL = (
    "https://huggingface.co/deepinsight/inswapper/resolve/main/inswapper_128.onnx"
)
GITHUB_RELEASE_URL = (
    "https://github.com/deepinsight/insightface/releases/download/v0.7/"
    "inswapper_128.onnx"
)

try:
    import insightface
    from insightface.app import FaceAnalysis
except ImportError:
    insightface = None
    FaceAnalysis = None


# -----------------------------------------------------------------------------
# Tema
# -----------------------------------------------------------------------------
COLORS = {
    "bg": "#0B1220",
    "surface": "#111827",
    "surface_2": "#172033",
    "surface_3": "#1F2937",
    "border": "#2B3952",
    "text": "#F8FAFC",
    "muted": "#94A3B8",
    "primary": "#2563EB",
    "primary_hover": "#1D4ED8",
    "success": "#16A34A",
    "success_hover": "#15803D",
    "warning": "#F59E0B",
    "danger": "#EF4444",
    "danger_hover": "#DC2626",
    "neutral": "#334155",
    "neutral_hover": "#475569",
}


OUTPUT_PRESETS = {
    "Orijinal boyut": None,
    "Full HD (1080p)": (1920, 1080),
    "4K UHD": (3840, 2160),
}


def ensure_inswapper_model(status_cb=None) -> str:
    """inswapper_128.onnx modelini bulur; yoksa geçici dosyaya güvenli biçimde indirir.

    Not: Model dosyası bu projenin MIT lisansına dahil değildir. Kullanıcı,
    InsightFace model lisans koşullarına uymaktan sorumludur.
    """
    model_dir = Path.home() / ".insightface" / "models" / "inswapper_128"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "inswapper_128.onnx"

    if model_path.exists() and model_path.stat().st_size > MIN_MODEL_BYTES:
        return str(model_path)

    last_err = None
    for label, url in (("GitHub Release", GITHUB_RELEASE_URL), ("Hugging Face", HF_MODEL_URL)):
        temp_path = None
        try:
            if status_cb:
                status_cb(f"Model indiriliyor: {label}...")
            request = urllib.request.Request(url, headers={"User-Agent": APP_USER_AGENT})
            with urllib.request.urlopen(request, timeout=60) as response:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".onnx", dir=model_dir) as tmp:
                    temp_path = Path(tmp.name)
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        tmp.write(chunk)
            if temp_path.stat().st_size <= MIN_MODEL_BYTES:
                raise RuntimeError("İndirilen model dosyası beklenenden küçük.")
            temp_path.replace(model_path)
            LOGGER.info("Model indirildi: %s", label)
            return str(model_path)
        except Exception as exc:
            last_err = exc
            LOGGER.warning("Model indirme başarısız (%s): %s", label, exc)
            if temp_path and temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass

    raise RuntimeError(f"Model indirilemedi: {last_err}")


class HoverButton(tk.Button):
    """Tkinter icin modern hover efektli buton."""

    def __init__(
        self,
        master,
        *,
        text,
        command=None,
        bg=COLORS["primary"],
        hover=COLORS["primary_hover"],
        fg="white",
        disabled_bg="#273449",
        disabled_fg="#64748B",
        **kwargs,
    ):
        self.normal_bg = bg
        self.hover_bg = hover
        self.disabled_bg = disabled_bg
        self.disabled_fg = disabled_fg
        self.normal_fg = fg

        super().__init__(
            master,
            text=text,
            command=command,
            bg=bg,
            fg=fg,
            activebackground=hover,
            activeforeground=fg,
            disabledforeground=disabled_fg,
            relief="flat",
            bd=0,
            cursor="hand2",
            font=("Segoe UI Semibold", 10),
            padx=16,
            pady=10,
            highlightthickness=0,
            **kwargs,
        )
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)

    def _on_enter(self, _event=None):
        if str(self["state"]) != "disabled":
            self.configure(bg=self.hover_bg)

    def _on_leave(self, _event=None):
        if str(self["state"]) != "disabled":
            self.configure(bg=self.normal_bg)

    def set_enabled(self, enabled: bool):
        if enabled:
            self.configure(
                state="normal",
                bg=self.normal_bg,
                fg=self.normal_fg,
                cursor="hand2",
            )
        else:
            self.configure(
                state="disabled",
                bg=self.disabled_bg,
                fg=self.disabled_fg,
                cursor="arrow",
            )


class ImageCard(tk.Frame):
    """Responsive onizleme karti."""

    def __init__(self, master, title: str, subtitle: str):
        super().__init__(
            master,
            bg=COLORS["surface"],
            highlightbackground=COLORS["border"],
            highlightcolor=COLORS["border"],
            highlightthickness=1,
            bd=0,
        )
        self._pil_image: Image.Image | None = None
        self._photo = None
        self._resize_after = None

        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        header = tk.Frame(self, bg=COLORS["surface"])
        header.grid(row=0, column=0, sticky="ew", padx=16, pady=(14, 8))

        tk.Label(
            header,
            text=title,
            bg=COLORS["surface"],
            fg=COLORS["text"],
            font=("Segoe UI Semibold", 12),
        ).pack(anchor="w")

        tk.Label(
            header,
            text=subtitle,
            bg=COLORS["surface"],
            fg=COLORS["muted"],
            font=("Segoe UI", 9),
        ).pack(anchor="w", pady=(2, 0))

        preview_wrap = tk.Frame(self, bg=COLORS["surface_2"])
        preview_wrap.grid(row=1, column=0, sticky="nsew", padx=14, pady=(0, 14))
        preview_wrap.grid_rowconfigure(0, weight=1)
        preview_wrap.grid_columnconfigure(0, weight=1)

        self.preview = tk.Label(
            preview_wrap,
            text="Fotoğraf seçilmedi",
            bg=COLORS["surface_2"],
            fg=COLORS["muted"],
            font=("Segoe UI", 10),
            compound="center",
        )
        self.preview.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        self.preview.bind("<Configure>", self._on_resize)

    def clear(self):
        self._pil_image = None
        self._photo = None
        self.preview.configure(image="", text="Fotoğraf seçilmedi")

    def set_image(self, image: Image.Image):
        self._pil_image = ImageOps.exif_transpose(image.convert("RGB"))
        self._render()

    def _on_resize(self, _event=None):
        if self._resize_after:
            try:
                self.after_cancel(self._resize_after)
            except tk.TclError:
                pass
        self._resize_after = self.after(80, self._render)

    def _render(self):
        if self._pil_image is None:
            return

        w = max(self.preview.winfo_width() - 20, 80)
        h = max(self.preview.winfo_height() - 20, 80)
        im = self._pil_image.copy()
        im.thumbnail((w, h), Image.Resampling.LANCZOS)
        self._photo = ImageTk.PhotoImage(im)
        self.preview.configure(image=self._photo, text="")


class FaceSwapApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(f"{APP_NAME} v{__version__} • InsightFace CPU")
        self.root.geometry("1240x790")
        self.root.minsize(820, 640)
        self.root.configure(bg=COLORS["bg"])

        self.source_path: str | None = None
        self.target_path: str | None = None
        self.result_img: np.ndarray | None = None

        self.face_app = None
        self.swapper = None
        self.model_ready = False
        self.busy = False
        self._layout_mode = None
        self._layout_after = None

        self.output_var = tk.StringVar(value="Full HD (1080p)")
        self.swap_all_faces_var = tk.BooleanVar(value=True)

        self._setup_ttk_style()
        self._build_ui()
        self.root.bind("<Configure>", self._schedule_reflow)
        self._load_models_async()

    # ------------------------------------------------------------------ UI ----
    def _setup_ttk_style(self):
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure(
            "Modern.TCombobox",
            fieldbackground=COLORS["surface_2"],
            background=COLORS["surface_2"],
            foreground=COLORS["text"],
            arrowcolor=COLORS["text"],
            bordercolor=COLORS["border"],
            lightcolor=COLORS["border"],
            darkcolor=COLORS["border"],
            padding=7,
        )
        style.map(
            "Modern.TCombobox",
            fieldbackground=[("readonly", COLORS["surface_2"])],
            foreground=[("readonly", COLORS["text"])],
            selectbackground=[("readonly", COLORS["surface_2"])],
            selectforeground=[("readonly", COLORS["text"])],
        )

        style.configure(
            "Modern.Horizontal.TProgressbar",
            troughcolor=COLORS["surface_3"],
            background=COLORS["primary"],
            bordercolor=COLORS["surface_3"],
            lightcolor=COLORS["primary"],
            darkcolor=COLORS["primary"],
            thickness=4,
        )

    def _build_ui(self):
        self.main = tk.Frame(self.root, bg=COLORS["bg"])
        self.main.pack(fill="both", expand=True, padx=20, pady=18)
        self.main.grid_columnconfigure(0, weight=1)
        self.main.grid_rowconfigure(3, weight=1)

        # Header
        header = tk.Frame(self.main, bg=COLORS["bg"])
        header.grid(row=0, column=0, sticky="ew")
        header.grid_columnconfigure(0, weight=1)

        title_block = tk.Frame(header, bg=COLORS["bg"])
        title_block.grid(row=0, column=0, sticky="w")

        tk.Label(
            title_block,
            text=APP_NAME,
            bg=COLORS["bg"],
            fg=COLORS["text"],
            font=("Segoe UI Semibold", 23),
        ).pack(anchor="w")
        tk.Label(
            title_block,
            text="InsightFace • CPU • Responsive arayüz • 1080p / 4K dışa aktarma",
            bg=COLORS["bg"],
            fg=COLORS["muted"],
            font=("Segoe UI", 10),
        ).pack(anchor="w", pady=(2, 0))

        self.model_badge = tk.Label(
            header,
            text="● MODEL YÜKLENİYOR",
            bg=COLORS["surface_2"],
            fg=COLORS["warning"],
            font=("Segoe UI Semibold", 9),
            padx=12,
            pady=7,
        )
        self.model_badge.grid(row=0, column=1, sticky="e")

        # Toolbar card
        self.toolbar = tk.Frame(
            self.main,
            bg=COLORS["surface"],
            highlightbackground=COLORS["border"],
            highlightthickness=1,
        )
        self.toolbar.grid(row=1, column=0, sticky="ew", pady=(16, 10))
        for col in range(8):
            self.toolbar.grid_columnconfigure(col, weight=0)
        self.toolbar.grid_columnconfigure(7, weight=1)

        self.source_btn = HoverButton(
            self.toolbar,
            text="1  Kaynak Yüz Seç",
            command=self.pick_source,
            bg=COLORS["neutral"],
            hover=COLORS["neutral_hover"],
        )
        self.source_btn.grid(row=0, column=0, padx=(14, 6), pady=14, sticky="ew")

        self.target_btn = HoverButton(
            self.toolbar,
            text="2  Hedef Foto Seç",
            command=self.pick_target,
            bg=COLORS["neutral"],
            hover=COLORS["neutral_hover"],
        )
        self.target_btn.grid(row=0, column=1, padx=6, pady=14, sticky="ew")

        self.swap_btn = HoverButton(
            self.toolbar,
            text="3  Face Swap Uygula",
            command=self.run_swap,
            bg=COLORS["primary"],
            hover=COLORS["primary_hover"],
        )
        self.swap_btn.grid(row=0, column=2, padx=6, pady=14, sticky="ew")
        self.swap_btn.set_enabled(False)

        self.save_btn = HoverButton(
            self.toolbar,
            text="4  Kaydet",
            command=self.save_result,
            bg=COLORS["success"],
            hover=COLORS["success_hover"],
        )
        self.save_btn.grid(row=0, column=3, padx=6, pady=14, sticky="ew")
        self.save_btn.set_enabled(False)

        self.output_label = tk.Label(
            self.toolbar,
            text="Çıktı",
            bg=COLORS["surface"],
            fg=COLORS["muted"],
            font=("Segoe UI", 9),
        )
        self.output_label.grid(row=0, column=4, padx=(18, 6), pady=14)

        self.output_combo = ttk.Combobox(
            self.toolbar,
            textvariable=self.output_var,
            values=list(OUTPUT_PRESETS.keys()),
            state="readonly",
            width=18,
            style="Modern.TCombobox",
        )
        self.output_combo.grid(row=0, column=5, padx=(0, 10), pady=14)

        self.faces_check = tk.Checkbutton(
            self.toolbar,
            text="Hedefteki tüm yüzlere uygula",
            variable=self.swap_all_faces_var,
            bg=COLORS["surface"],
            fg=COLORS["text"],
            activebackground=COLORS["surface"],
            activeforeground=COLORS["text"],
            selectcolor=COLORS["surface_2"],
            font=("Segoe UI", 9),
            cursor="hand2",
        )
        self.faces_check.grid(row=0, column=6, padx=(4, 12), pady=14)

        # Status
        status_wrap = tk.Frame(self.main, bg=COLORS["bg"])
        status_wrap.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        status_wrap.grid_columnconfigure(0, weight=1)

        self.status_label = tk.Label(
            status_wrap,
            text="Model hazırlanıyor, lütfen bekleyin...",
            bg=COLORS["bg"],
            fg=COLORS["muted"],
            font=("Segoe UI", 10),
            anchor="w",
        )
        self.status_label.grid(row=0, column=0, sticky="ew")

        self.progress = ttk.Progressbar(
            status_wrap,
            mode="indeterminate",
            style="Modern.Horizontal.TProgressbar",
            length=180,
        )
        self.progress.grid(row=0, column=1, sticky="e", padx=(10, 0))
        self.progress.start(12)

        # Image cards
        self.cards_wrap = tk.Frame(self.main, bg=COLORS["bg"])
        self.cards_wrap.grid(row=3, column=0, sticky="nsew")
        self.cards_wrap.grid_rowconfigure(0, weight=1)
        for col in range(3):
            self.cards_wrap.grid_columnconfigure(col, weight=1, uniform="cards")

        self.source_card = ImageCard(
            self.cards_wrap, "Kaynak Yüz", "Yüz kimliği buradan alınır"
        )
        self.target_card = ImageCard(
            self.cards_wrap, "Hedef Fotoğraf", "Poz, ışık ve sahne korunur"
        )
        self.result_card = ImageCard(
            self.cards_wrap, "Sonuç", "İşlem sonrası önizleme"
        )
        self.cards = [self.source_card, self.target_card, self.result_card]
        self._apply_layout("wide")

        # Footer
        footer = tk.Frame(self.main, bg=COLORS["bg"])
        footer.grid(row=4, column=0, sticky="ew", pady=(10, 0))
        footer.grid_columnconfigure(0, weight=1)

        tk.Label(
            footer,
            text="İpucu: En iyi sonuç için önden çekilmiş, net ve iyi ışıklandırılmış kaynak yüz kullanın.",
            bg=COLORS["bg"],
            fg=COLORS["muted"],
            font=("Segoe UI", 9),
        ).grid(row=0, column=0, sticky="w")

        tk.Label(
            footer,
            text="1080p / 4K büyütme = yüksek kaliteli yeniden örnekleme",
            bg=COLORS["bg"],
            fg=COLORS["muted"],
            font=("Segoe UI", 9),
        ).grid(row=0, column=1, sticky="e")

    def _schedule_reflow(self, event):
        if event.widget is not self.root:
            return
        if self._layout_after:
            try:
                self.root.after_cancel(self._layout_after)
            except tk.TclError:
                pass
        self._layout_after = self.root.after(120, self._reflow)

    def _reflow(self):
        width = self.root.winfo_width()
        if width < 980:
            mode = "narrow"
        else:
            mode = "wide"
        if mode != self._layout_mode:
            self._apply_layout(mode)

    def _apply_layout(self, mode: str):
        self._layout_mode = mode
        for card in self.cards:
            card.grid_forget()

        if mode == "wide":
            self.cards_wrap.grid_rowconfigure(0, weight=1)
            self.cards_wrap.grid_rowconfigure(1, weight=0)
            for col in range(3):
                self.cards_wrap.grid_columnconfigure(col, weight=1, uniform="cards")
            self.source_card.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
            self.target_card.grid(row=0, column=1, sticky="nsew", padx=6)
            self.result_card.grid(row=0, column=2, sticky="nsew", padx=(6, 0))

            # Toolbar tek satir
            for widget in (
                self.source_btn,
                self.target_btn,
                self.swap_btn,
                self.save_btn,
                self.output_label,
                self.output_combo,
                self.faces_check,
            ):
                widget.grid_forget()

            self.source_btn.grid(row=0, column=0, padx=(14, 6), pady=14, sticky="ew")
            self.target_btn.grid(row=0, column=1, padx=6, pady=14, sticky="ew")
            self.swap_btn.grid(row=0, column=2, padx=6, pady=14, sticky="ew")
            self.save_btn.grid(row=0, column=3, padx=6, pady=14, sticky="ew")
            self.output_label.grid(row=0, column=4, padx=(18, 6), pady=14)
            self.output_combo.grid(row=0, column=5, padx=(0, 10), pady=14)
            self.faces_check.grid(row=0, column=6, padx=(4, 12), pady=14)
        else:
            # Dar ekranda kartlari dikey yiginla
            for col in range(3):
                self.cards_wrap.grid_columnconfigure(col, weight=1, uniform="")
            for row in range(3):
                self.cards_wrap.grid_rowconfigure(row, weight=1)
            self.source_card.grid(row=0, column=0, sticky="nsew", pady=(0, 5))
            self.target_card.grid(row=1, column=0, sticky="nsew", pady=5)
            self.result_card.grid(row=2, column=0, sticky="nsew", pady=(5, 0))

            # Toolbar iki satir
            for widget in (
                self.source_btn,
                self.target_btn,
                self.swap_btn,
                self.save_btn,
                self.output_label,
                self.output_combo,
                self.faces_check,
            ):
                widget.grid_forget()

            for col in range(4):
                self.toolbar.grid_columnconfigure(col, weight=1)
            self.source_btn.grid(row=0, column=0, padx=(12, 5), pady=(12, 5), sticky="ew")
            self.target_btn.grid(row=0, column=1, padx=5, pady=(12, 5), sticky="ew")
            self.swap_btn.grid(row=0, column=2, padx=5, pady=(12, 5), sticky="ew")
            self.save_btn.grid(row=0, column=3, padx=(5, 12), pady=(12, 5), sticky="ew")
            self.output_label.grid(row=1, column=0, padx=(12, 4), pady=(5, 12), sticky="w")
            self.output_combo.grid(row=1, column=1, padx=(4, 6), pady=(5, 12), sticky="ew")
            self.faces_check.grid(row=1, column=2, columnspan=2, padx=(6, 12), pady=(5, 12), sticky="w")

    # ------------------------------------------------------------ Model -------
    def _load_models_async(self):
        threading.Thread(target=self._load_models, daemon=True).start()

    def _load_models(self):
        if insightface is None:
            self._set_status(
                "HATA: insightface kurulu değil. pip install insightface onnxruntime çalıştırın.",
                "danger",
            )
            self._set_model_badge("● MODEL HATASI", COLORS["danger"])
            self._stop_progress()
            return

        try:
            self._set_status("Yüz analiz modeli yükleniyor...", "info")
            self.face_app = FaceAnalysis(
                name="buffalo_l", providers=["CPUExecutionProvider"]
            )
            self.face_app.prepare(ctx_id=-1, det_size=(640, 640))

            onnx_path = ensure_inswapper_model(
                status_cb=lambda text: self._set_status(text, "info")
            )

            from insightface.model_zoo.inswapper import INSwapper

            self.swapper = INSwapper(model_file=onnx_path)
            self.model_ready = True
            self._set_status("Hazır. Kaynak yüz ve hedef fotoğrafı seçin.", "success")
            self._set_model_badge("● MODEL HAZIR", COLORS["success"])
            self._stop_progress()
            self._update_action_states()
        except Exception as exc:
            LOGGER.exception("Model yüklenemedi")
            self._set_status(f"Model yüklenemedi: {exc}", "danger")
            self._set_model_badge("● MODEL HATASI", COLORS["danger"])
            self._stop_progress()

    def _ui(self, callback):
        try:
            self.root.after(0, callback)
        except tk.TclError:
            pass

    def _set_status(self, text: str, kind: str = "info"):
        color_map = {
            "info": COLORS["muted"],
            "success": COLORS["success"],
            "warning": COLORS["warning"],
            "danger": COLORS["danger"],
        }
        color = color_map.get(kind, COLORS["muted"])
        self._ui(lambda: self.status_label.configure(text=text, fg=color))

    def _set_model_badge(self, text: str, color: str):
        self._ui(lambda: self.model_badge.configure(text=text, fg=color))

    def _start_progress(self):
        def _start():
            self.progress.grid()
            self.progress.start(10)

        self._ui(_start)

    def _stop_progress(self):
        def _stop():
            self.progress.stop()
            self.progress.grid_remove()

        self._ui(_stop)

    # --------------------------------------------------------- File picker ----
    def pick_source(self):
        path = filedialog.askopenfilename(
            title="Kaynak yüz fotoğrafını seç",
            filetypes=[
                ("Resim dosyaları", "*.jpg *.jpeg *.png *.webp *.bmp"),
                ("Tüm dosyalar", "*.*"),
            ],
        )
        if path:
            self.source_path = path
            self._load_preview(path, self.source_card)
            self._set_status(f"Kaynak seçildi: {Path(path).name}", "info")
            self._update_action_states()

    def pick_target(self):
        path = filedialog.askopenfilename(
            title="Hedef fotoğrafı seç",
            filetypes=[
                ("Resim dosyaları", "*.jpg *.jpeg *.png *.webp *.bmp"),
                ("Tüm dosyalar", "*.*"),
            ],
        )
        if path:
            self.target_path = path
            self._load_preview(path, self.target_card)
            self.result_img = None
            self.result_card.clear()
            self._set_status(f"Hedef seçildi: {Path(path).name}", "info")
            self._update_action_states()

    def _load_preview(self, path: str, card: ImageCard):
        try:
            with Image.open(path) as im:
                card.set_image(im.copy())
        except Exception as exc:
            messagebox.showerror("Görüntü hatası", f"Fotoğraf açılamadı:\n{exc}")

    def _show_result(self, result_bgr: np.ndarray):
        rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
        self.result_card.set_image(Image.fromarray(rgb))

    def _update_action_states(self):
        ready_for_swap = bool(
            self.model_ready and self.source_path and self.target_path and not self.busy
        )
        self._ui(lambda: self.source_btn.set_enabled(not self.busy))
        self._ui(lambda: self.target_btn.set_enabled(not self.busy))
        self._ui(lambda: self.swap_btn.set_enabled(ready_for_swap))
        self._ui(lambda: self.save_btn.set_enabled(self.result_img is not None and not self.busy))

    # ------------------------------------------------------------- Swap -------
    def run_swap(self):
        if not self.model_ready:
            messagebox.showwarning("Model hazır değil", "Model henüz yüklenmedi.")
            return
        if not self.source_path or not self.target_path:
            messagebox.showwarning(
                "Eksik seçim", "Önce kaynak yüz ve hedef fotoğrafı seçin."
            )
            return
        if self.busy:
            return

        self.busy = True
        self._swap_all_faces_current = bool(self.swap_all_faces_var.get())
        self.result_img = None
        self._update_action_states()
        self._set_status("Face swap uygulanıyor... CPU kullanımına göre sürebilir.", "info")
        self._start_progress()
        threading.Thread(target=self._do_swap, daemon=True).start()

    @staticmethod
    def _imread_unicode(path: str) -> np.ndarray | None:
        """Windows'ta Türkçe karakter içeren yollarda cv2.imread sorununu önler."""
        try:
            data = np.fromfile(path, dtype=np.uint8)
            return cv2.imdecode(data, cv2.IMREAD_COLOR)
        except Exception:
            return cv2.imread(path)

    def _do_swap(self):
        try:
            src_img = self._imread_unicode(self.source_path)
            tgt_img = self._imread_unicode(self.target_path)

            if src_img is None:
                raise RuntimeError("Kaynak fotoğraf okunamadı.")
            if tgt_img is None:
                raise RuntimeError("Hedef fotoğraf okunamadı.")

            src_faces = self.face_app.get(src_img)
            tgt_faces = self.face_app.get(tgt_img)

            if not src_faces:
                self._set_status("Kaynak fotoğrafta yüz bulunamadı.", "danger")
                return
            if not tgt_faces:
                self._set_status("Hedef fotoğrafta yüz bulunamadı.", "danger")
                return

            # Kaynakta birden fazla yüz varsa en büyük yüzü kullan.
            source_face = max(
                src_faces,
                key=lambda f: max(1, (f.bbox[2] - f.bbox[0]))
                * max(1, (f.bbox[3] - f.bbox[1])),
            )

            result = tgt_img.copy()
            faces_to_swap = tgt_faces if self._swap_all_faces_current else [tgt_faces[0]]

            for face in faces_to_swap:
                result = self.swapper.get(
                    result, face, source_face, paste_back=True
                )

            self.result_img = result
            self._ui(lambda: self._show_result(result))
            self._set_status(
                f"Tamamlandı. {len(faces_to_swap)} yüz işlendi. Kaydet ile dışa aktarabilirsiniz.",
                "success",
            )
        except Exception as exc:
            LOGGER.exception("Face swap işlemi başarısız")
            self._set_status(f"İşlem hatası: {exc}", "danger")
        finally:
            self.busy = False
            self._stop_progress()
            self._update_action_states()

    # -------------------------------------------------------------- Export ----
    @staticmethod
    def _target_size_for_orientation(
        image: np.ndarray, landscape_size: tuple[int, int]
    ) -> tuple[int, int]:
        """Preset'i hedef görselin yönüne göre yatay/dikey döndürür."""
        h, w = image.shape[:2]
        preset_w, preset_h = landscape_size
        if h > w:
            return preset_h, preset_w  # 1080x1920 / 2160x3840
        return preset_w, preset_h

    @staticmethod
    def _resize_cover(image: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
        """
        Görüntüyü bozmadan hedef çözünürlüğe taşır.
        Oran farklıysa kırpmak yerine siyah şerit ekleyerek tüm görüntüyü korur.
        """
        h, w = image.shape[:2]
        scale = min(target_w / w, target_h / h)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))

        interpolation = cv2.INTER_LANCZOS4 if scale > 1 else cv2.INTER_AREA
        resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)

        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        x = (target_w - new_w) // 2
        y = (target_h - new_h) // 2
        canvas[y : y + new_h, x : x + new_w] = resized
        return canvas

    def _prepare_export_image(self) -> np.ndarray:
        preset = OUTPUT_PRESETS.get(self.output_var.get())
        if preset is None:
            return self.result_img.copy()

        target_w, target_h = self._target_size_for_orientation(
            self.result_img, preset
        )
        return self._resize_cover(self.result_img, target_w, target_h)

    @staticmethod
    def _imwrite_unicode(path: str, image: np.ndarray) -> bool:
        ext = Path(path).suffix.lower()
        if ext not in {".png", ".jpg", ".jpeg", ".webp"}:
            ext = ".png"

        params = []
        if ext == ".png":
            params = [cv2.IMWRITE_PNG_COMPRESSION, 2]
        elif ext in {".jpg", ".jpeg"}:
            params = [cv2.IMWRITE_JPEG_QUALITY, 95]
        elif ext == ".webp":
            params = [cv2.IMWRITE_WEBP_QUALITY, 95]

        ok, encoded = cv2.imencode(ext, image, params)
        if not ok:
            return False
        encoded.tofile(path)
        return True

    def save_result(self):
        if self.result_img is None:
            messagebox.showinfo("Bilgi", "Önce Face Swap Uygula butonuna basın.")
            return

        default_name = "face_swap_sonuc_1080p.png"
        if self.output_var.get() == "4K UHD":
            default_name = "face_swap_sonuc_4k.png"
        elif self.output_var.get() == "Orijinal boyut":
            default_name = "face_swap_sonuc.png"

        path = filedialog.asksaveasfilename(
            title="Sonucu kaydet",
            initialfile=default_name,
            defaultextension=".png",
            filetypes=[
                ("PNG - Kayıpsız", "*.png"),
                ("JPEG - Yüksek kalite", "*.jpg"),
                ("WebP - Yüksek kalite", "*.webp"),
            ],
        )
        if not path:
            return

        try:
            export_img = self._prepare_export_image()
            ok = self._imwrite_unicode(path, export_img)
            if not ok:
                raise RuntimeError("Görüntü kodlanamadı.")

            h, w = export_img.shape[:2]
            self._set_status(
                f"Kaydedildi: {Path(path).name} • {w}×{h}px", "success"
            )
            messagebox.showinfo(
                "Kaydedildi",
                f"Sonuç başarıyla kaydedildi.\n\n"
                f"Dosya: {path}\n"
                f"Çözünürlük: {w} × {h} px\n"
                f"Preset: {self.output_var.get()}",
            )
        except Exception as exc:
            messagebox.showerror("Kayıt hatası", f"Dosya kaydedilemedi:\n{exc}")


def main():
    root = tk.Tk()
    app = FaceSwapApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
