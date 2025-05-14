import sys
import os
import torch
from transformers import LlavaForConditionalGeneration, TextIteratorStreamer, AutoProcessor
from PIL import Image
from threading import Thread # For model.generate in its own thread
from typing import Generator, List, Union, Optional, Dict # Typing not strictly needed for Generator here
from pathlib import Path
import base64 # For logo

# LIGER Kernel import - ensure liger_kernel is installed and in PYTHONPATH
try:
    from liger_kernel.transformers import apply_liger_kernel_to_llama
    LIGER_AVAILABLE = True
except ImportError:
    LIGER_AVAILABLE = False
    print("Warning: liger_kernel not found. LLM optimizations will be disabled.")
    def apply_liger_kernel_to_llama(model): # Stub
        print("LIGER Kernel not applied (stub).")
        pass


from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QFileDialog, QLineEdit,
    QTextEdit, QComboBox, QVBoxLayout, QHBoxLayout, QCheckBox, QMessageBox,
    QSizePolicy, QStatusBar, QProgressBar, QMainWindow, QSlider, QScrollArea,
    QGroupBox, QTextBrowser, QFrame, QGridLayout
)
from PyQt5.QtGui import QPixmap, QIcon, QTextCursor, QImage
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QObject, QSize

# --- Constants and Mappings ---
LOGO_SRC_BASE64 = "PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiIHN0YW5kYWxvbmU9Im5vIj8+CjwhRE9DVFlQRSBzdmcgUFVCTElDICItLy9XM0MvL0RURCBTVkcgMS4xLy9FTiIgImh0dHA6Ly93d3cudzMub3JnL0dyYXBoaWNzL1NWRy8xLjEvRFREL3N2ZzExLmR0ZCI+Cjxzdmcgd2lkdGg9IjEwMCUiIGhlaWdodD0iMTAwJSIgdmlld0JveD0iMCAwIDUzOCA1MzUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bWxuczp4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgeG1sOnNwYWNlPSJwcmVzZXJ2ZSIgeG1sbnM6c2VyaWY9Imh0dHA6Ly93d3cuc2VyaWYuY29tLyIgc3R5bGU9ImZpbGwtcnVsZTpldmVub2RkO2NsaXAtcnVsZTpldmVub2RkO3N0cm9rZS1saW5lam9pbjpyb3VuZDtzdHJva2UtbWl0ZXJsaW1pdDoyOyI+CiAgICA8ZyB0cmFuc2Zvcm09Im1hdHJpeCgxLDAsMCwxLC0xNDcuODcxLDAuMDAxOTA4NjMpIj4KICAgICAgICA8cGF0aCBkPSJNMTk1LjY3LDIyMS42N0MxOTYuNzMsMjA1LjM3IDIwMC4yOCwxODkuNzYgMjA3LjkxLDE3NS4zN0MyMjcuOTgsMTM3LjUxIDI1OS4zMywxMTQuODggMzAyLjAxLDExMS42M0MzMzQuMTUsMTA5LjE4IDM2Ni41OSwxMTAuNiAzOTguODksMTEwLjNDNDAwLjUzLDExMC4yOCA0MDIuMTYsMTEwLjMgNDA0LjQsMTEwLjNDNDA0LjQsMTAxLjk5IDQwNC41Niw5NC4wNSA0MDQuMjMsODYuMTJDNDA0LjE4LDg0Ljg0IDQwMi4xNSw4My4xMyA0MDAuNjYsODIuNDlDMzgzLjIzLDc1LjAyIDM3My4wNSw1OS43OSAzNzMuOTYsNDAuOTZDMzc1LjA5LDE3LjU0IDM5MS40NywyLjY2IDQxMC42NSwwLjM3QzQzNy44OSwtMi44OSA0NTUuNTYsMTUuODQgNDU5LjI2LDM0LjY5QzQ2Mi45Niw1My41NyA0NTIuMTgsNzYuOTMgNDMyLjgxLDgyLjY2QzQzMS42NCw4My4wMSA0MzAuMzMsODUuMjMgNDMwLjI4LDg2LjYyQzQzMC4wMyw5NC4yNiA0MzAuMTYsMTAxLjkyIDQzMC4xNiwxMTAuM0w0MzUuNjMsMTEwLjNDNDYzLjc5LDExMC4zIDQ5MS45NiwxMTAuMjggNTIwLjEyLDExMC4zQzU3NC44NCwxMTAuMzYgNjIzLjA0LDE0OC4zNSA2MzUuNjcsMjAxLjU1QzYzNy4yMywyMDguMTMgNjM3LjgzLDIxNC45MyA2MzguODksMjIxLjY3QzY2MC40MywyMjQuOTQgNjc1LjE5LDIzNi42MiA2ODIuMzYsMjU3LjRDNjgzLjU5LDI2MC45NyA2ODQuNjUsMjY0LjgyIDY4NC42NywyNjguNTRDNjg0Ljc3LDI4My4zNCA2ODUuNzYsMjk4LjMxIDY4My45NCwzMTIuOTFDNjgwLjg5LDMzNy4yOSA2NjIuODYsMzUzLjM2IDYzOC40NywzNTUuODJDNjM1LjE0LDM4NS4wOCA2MjEuOTEsNDA5LjQxIDYwMC40NSw0MjkuMjFDNTgxLjYsNDQ2LjYxIDU1OS4xNCw0NTcuNSA1MzMuNTcsNDU5LjE4QzUwOC4xOCw0NjAuODQgNDgyLjY0LDQ2MC4yIDQ1Ny4xNiw0NjAuMzhDNDM1LjE2LDQ2MC41MyA0MTMuMTcsNDYwLjM0IDM5MS4xNyw0NjAuNTNDMzg4Ljc2LDQ2MC41NSAzODUuOTUsNDYxLjU2IDM4NC4wMyw0NjMuMDRDMzcxLjU0LDQ3Mi42MiAzNTkuMTMsNDgyLjMxIDM0Ni45Miw0OTIuMjVDMzM4Ljk0LDQ5OC43NSAzMzEuMzksNTA1Ljc3IDMyMy41Niw1MTIuNDZDMzE3LjQ1LDUxNy42OCAzMTAuOTMsNTIyLjQ0IDMwNS4xMSw1MjcuOTVDMzAxLjE5LDUzMS42NiAyOTYuNTIsNTMzLjE3IDI5MS42OSw1MzQuMzZDMjg1LjY1LDUzNS44NSAyNzkuMjIsNTI5LjEzIDI3OS4wMSw1MjEuMTlDMjc4LDgsNTEyLjg2IDI3OC45NSw1MDQuNTMgMjc4Ljk0LDQ5Ni4xOUwyNzguOTQsNDU2LjY5QzIzMi44Miw0MzguMTYgMjAzLjU2LDQwNi4yMyAxOTUuMDcsMzU2LjA4QzE5My4yNiwzNTUuNzUgMTkwLjg0LDM1NSo0MSAxODguNDgsMzU0Ljg2QzE2Ny40NiwzNDkuOTEgMTU1LjA0LDMzNi4wMiAxNTAuNzIsMzE1LjYyQzE0Ni45OCwyOTcuOTkgMTQ2LjksMjc5LjY3IDE1MC42MSwyNjIuMDlDMTU1LjU1LDIzOC42OCAxNzEuNDIsMjI1LjU5IDE5NS42NiwyMjEuNjdMMTk1LjY3LDIyMS42N1pNMzA4LjA3LDQ4Ny44MkMzMTUuOTQsNDgxLjEzIDMyMi44NSw0NzUuMTMgMzI5LjksNDY5LjNDMzQ0LjM5LDQ1Ny4zMSAzNTguOSw0NDUuMzYgMzczLjU0LDQzMy41NkMzNzUuMTcsNDMyLjI1IDM3Ny42OCw0MzEuNCAzNzkuNzksNDMxLjM5QzQxNC43OCw0MzEuMjYgNDQ5Ljc4LDQzMS4zOCA0ODQuNzcsNDMxLjI0QzUwMC4zOSw0MzEuMTggNTE2LjEzLDQzMS43NiA1MzEuNjIsNDMwLjE2QzU3Ni45Miw0MjUuNDkgNjA5LjI0LDM4Ny43NyA2MDguOTUsMzQ0Ljg0QzYwOC42OCwzMDUuNTIgNjA4LjkzLDI2Ni4xOSA2MDguODcsMjI2Ljg2QzYwOC44NywyMjMuMjIgNjA4LjU4LDIxOS41NSA2MDcuOTksMjE1Ljk2QzYwMy4xMSwxODYuMjkgNTg4LjYxLDE2My4zMyA1NjEuMzIsMTQ5LjMyQzU0OS4wNCwxNDMuMDIgNTM2LjE1LDEzOS4yOSA1MjIuMjIsMTM5LjI5QzQ1My45LDEzOS4zMiAzODUuNTgsMTM5LjIgMzE3LjI2LDEzOS4zNUMzMDkuMiwxMzkuMzcgMzAwLjk2LDEzOS44OSAyOTMuMTEsMTQxLjZDMjU0LjE5LDE1MC4wNyAyMjUuMzMsMTg1LjY5IDIyNS4wMywyMjUuNDJDMjI0LjgsMjU2LjA4IDIyNC44NiwyODYuNzQgMjI0Ljk5LDMxNy40QzIyNS4wNSwzMzAuNTMgMjI0Ljc0LDM0My43NiAyMjYuMTgsMzU2Ljc3QzIyOC43NCwzODAuMDUgMjQwLjYsMzk4LjYyIDI1OC43OSw0MTIuOTNDMjczLjA0LDQyNC4xNCAyODkuNjMsNDMwLjAyIDMwNy42MSw0MzEuNTVDMzA3LjgyLDQzMi4wMyAzMDguMDYsNDMyLjMzIDMwOC4wNiw0MzIuNjNDMzA4LjA4LDQ1MC42IDMwOC4wOCw0NjguNTcgMzA4LjA4LDQ4Ny44MUwzMDguMDcsNDg3LjgyWk00MzUuNzksNDMuMzNDNDM1Ljk1LDMzLjQyIDQyNy42MSwyNC42NSA0MTcuOCwyNC40QzQwNi43NiwyNC4xMiAzOTguMjUsMzIuMDUgMzk4LjEzLDQyLjc0QzM5OC4wMSw1My4wNCA0MDYuNiw2Mi4xMiA0MTYuNDIsNjIuMDhDNDI3LjExLDYyLjA0IDQzNS42MSw1My44MSA0MzUuNzgsNDMuMzNMNDM1Ljc5LDQzLjMzWiIgc3R5bGU9ImZpbGw6cmdiKDczLDQ3LDExOCk7ZmlsbC1ydWxlOm5vbnplcm87Ii8+CiAgICAgICAgPHBhdGggZD0iTTQxOS4zLDM5MS42M0MzNzQuNDYsMzkwLjQgMzQxLjUxLDM3Mi42MyAzMTguMDEsMzM3LjcxQzMxNS42NywzMzQuMjMgMzEzLjc3LDMzMC4wNCAzMTMuMSwzMjUuOTVDMzExLjg0LDMxOC4yOCAzMTYuNTMsMzExLjcgMzIzLjcyLDMwOS40NkMzMzAuNjYsMzA3LjI5IDMzOC4zMiwzMTAuMSAzNDEuOTgsMzE3LjAzQzM0OS4xNSwzMzAuNjMgMzU5LjE2LDM0MS4zNSAzNzIuMywzNDkuMzFDNDAxLjMyLDM2Ni44OSA0NDQuNTYsMzYzLjcgNDcwLjYxLDM0Mi4zNUM0NzkuMSwzMzUuMzkgNDg2LjA4LDMyNy40MSA0OTEuNTUsMzE3Ljk3QzQ5NS4wNSwzMTEuOTMgNTAwLjIsMzA4LjE4IDUwNy40NywzMDguOTVDNTEzLjczLDMwOS42MSA1MTguODYsMzEyLjg4IDUyMC4xMiwzMTkuMjFDNTIwLjksMzIzLjEzIDUyMC43MywzMjguMjIgNTE4LjgzLDMzMS41NUM1MDAuNjMsMzYzLjMyIDQ3My41NSwzODIuOTUgNDM3LjI5LDM4OS4zN0M0MzAuNDQsMzkwLjU4IDQyMy40OCwzOTEuMTIgNDE5LjI5LDM5MS42M0w0MTkuMywzOTEuNjNaIiBzdHlsZT0iZmlsbDpyZ2IoMjUwLDEzOSwxKTtmaWxsLXJ1bGU6bm9uemVybzsiLz4KICAgICAgICA8cGF0aCBkPSJNNDYyLjcxLDI0MC4xOUM0NjIuOCwyMTYuOTEgNDgwLjI0LDE5OS43OSA1MDQuMDEsMTk5LjY3QzUyNi41NywxOTkuNTUgNTQ0Ljg5LDIxOC4wNyA1NDQuNTEsMjQxLjM0QzU0NC4xOCwyNjEuODUgNTMwLjA5LDI4MS45NiA1MDEuOTEsMjgxLjIzQzQ4MC42OCwyODAuNjggNDYyLjE1LDI2My44IDQ2Mi43MSwyNDAuMkw0NjIuNzEsMjQwLjE5WiIgc3R5bGU9ImZpbGw6cmdiKDI1MCwxMzksMSk7ZmlsbC1ydWxlOm5vbnplcm87Ii8+CiAgICAgICAgPHBhdGggZD0iTTM3MC45OSwyNDAuMDhDMzcxLDI2Mi43OSAzNTIuNTMsMjgxLjM1IDMyOS44OSwyODEuMzdDMzA3LjA1LDI4MS40IDI4OC45NiwyNjMuNDIgMjg4Ljk2LDI0MC42OEMyODguOTYsMjE4LjE0IDMwNi43MywyMDAgMzI5LjE2LDE5OS42MkMzNTIuMDIsMTk5LjI0IDM3MC45OCwyMTcuNTcgMzcwLjk5LDI0MC4wOFoiIHN0eWxlPSJmaWxsOnJnYigyNTAsMTM5LDEpO2ZpbGwtcnVsZTpub256ZXJvOyIvPgogICAgPC9nPgo8L3N2Zz4K"
MODEL_PATH = "fancyfeast/llama-joycaption-beta-one-hf-llava"
CAPTION_TYPE_MAP = {
	"Descriptive": [
		"Write a detailed description for this image.",
		"Write a detailed description for this image in {word_count} words or less.",
		"Write a {length} detailed description for this image.",
	],
	"Descriptive (Casual)": [
		"Write a descriptive caption for this image in a casual tone.",
		"Write a descriptive caption for this image in a casual tone within {word_count} words.",
		"Write a {length} descriptive caption for this image in a casual tone.",
	],
	"Straightforward": [
		"Write a straightforward caption for this image. Begin with the main subject and medium. Mention pivotal elements—people, objects, scenery—using confident, definite language. Focus on concrete details like color, shape, texture, and spatial relationships. Show how elements interact. Omit mood and speculative wording. If text is present, quote it exactly. Note any watermarks, signatures, or compression artifacts. Never mention what's absent, resolution, or unobservable details. Vary your sentence structure and keep the description concise, without starting with “This image is…” or similar phrasing.",
		"Write a straightforward caption for this image within {word_count} words. Begin with the main subject and medium. Mention pivotal elements—people, objects, scenery—using confident, definite language. Focus on concrete details like color, shape, texture, and spatial relationships. Show how elements interact. Omit mood and speculative wording. If text is present, quote it exactly. Note any watermarks, signatures, or compression artifacts. Never mention what's absent, resolution, or unobservable details. Vary your sentence structure and keep the description concise, without starting with “This image is…” or similar phrasing.",
		"Write a {length} straightforward caption for this image. Begin with the main subject and medium. Mention pivotal elements—people, objects, scenery—using confident, definite language. Focus on concrete details like color, shape, texture, and spatial relationships. Show how elements interact. Omit mood and speculative wording. If text is present, quote it exactly. Note any watermarks, signatures, or compression artifacts. Never mention what's absent, resolution, or unobservable details. Vary your sentence structure and keep the description concise, without starting with “This image is…” or similar phrasing.",
	],
	"Stable Diffusion Prompt": [
		"Output a stable diffusion prompt that is indistinguishable from a real stable diffusion prompt.",
		"Output a stable diffusion prompt that is indistinguishable from a real stable diffusion prompt. {word_count} words or less.",
		"Output a {length} stable diffusion prompt that is indistinguishable from a real stable diffusion prompt.",
	],
	"MidJourney": [
		"Write a MidJourney prompt for this image.",
		"Write a MidJourney prompt for this image within {word_count} words.",
		"Write a {length} MidJourney prompt for this image.",
	],
	"Danbooru tag list": [
		"Generate only comma-separated Danbooru tags (lowercase_underscores). Strict order: `artist:`, `copyright:`, `character:`, `meta:`, then general tags. Include counts (1girl), appearance, clothing, accessories, pose, expression, actions, background. Use precise Danbooru syntax. No extra text.",
		"Generate only comma-separated Danbooru tags (lowercase_underscores). Strict order: `artist:`, `copyright:`, `character:`, `meta:`, then general tags. Include counts (1girl), appearance, clothing, accessories, pose, expression, actions, background. Use precise Danbooru syntax. No extra text. {word_count} words or less.",
		"Generate only comma-separated Danbooru tags (lowercase_underscores). Strict order: `artist:`, `copyright:`, `character:`, `meta:`, then general tags. Include counts (1girl), appearance, clothing, accessories, pose, expression, actions, background. Use precise Danbooru syntax. No extra text. {length} length.",
	],
	"e621 tag list": [
		"Write a comma-separated list of e621 tags in alphabetical order for this image. Start with the artist, copyright, character, species, meta, and lore tags (if any), prefixed by 'artist:', 'copyright:', 'character:', 'species:', 'meta:', and 'lore:'. Then all the general tags.",
		"Write a comma-separated list of e621 tags in alphabetical order for this image. Start with the artist, copyright, character, species, meta, and lore tags (if any), prefixed by 'artist:', 'copyright:', 'character:', 'species:', 'meta:', and 'lore:'. Then all the general tags. Keep it under {word_count} words.",
		"Write a {length} comma-separated list of e621 tags in alphabetical order for this image. Start with the artist, copyright, character, species, meta, and lore tags (if any), prefixed by 'artist:', 'copyright:', 'character:', 'species:', 'meta:', and 'lore:'. Then all the general tags.",
	],
	"Rule34 tag list": [
		"Write a comma-separated list of rule34 tags in alphabetical order for this image. Start with the artist, copyright, character, and meta tags (if any), prefixed by 'artist:', 'copyright:', 'character:', and 'meta:'. Then all the general tags.",
		"Write a comma-separated list of rule34 tags in alphabetical order for this image. Start with the artist, copyright, character, and meta tags (if any), prefixed by 'artist:', 'copyright:', 'character:', and 'meta:'. Then all the general tags. Keep it under {word_count} words.",
		"Write a {length} comma-separated list of rule34 tags in alphabetical order for this image. Start with the artist, copyright, character, and meta tags (if any), prefixed by 'artist:', 'copyright:', 'character:', and 'meta:'. Then all the general tags.",
	],
	"Booru-like tag list": [
		"Write a list of Booru-like tags for this image.",
		"Write a list of Booru-like tags for this image within {word_count} words.",
		"Write a {length} list of Booru-like tags for this image.",
	],
	"Art Critic": [
		"Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc.",
		"Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc. Keep it within {word_count} words.",
		"Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc. Keep it {length}.",
	],
	"Product Listing": [
		"Write a caption for this image as though it were a product listing.",
		"Write a caption for this image as though it were a product listing. Keep it under {word_count} words.",
		"Write a {length} caption for this image as though it were a product listing.",
	],
	"Social Media Post": [
		"Write a caption for this image as if it were being used for a social media post.",
		"Write a caption for this image as if it were being used for a social media post. Limit the caption to {word_count} words.",
		"Write a {length} caption for this image as if it were being used for a social media post.",
	],
}
NAME_OPTION = "If there is a person/character in the image you must refer to them as {name}."
CAPTION_LENGTH_CHOICES = ["any", "very short", "short", "medium-length", "long", "very long"] + [str(i) for i in range(20, 261, 10)]
EXTRA_OPTIONS_LIST = [
    NAME_OPTION,
    "Do NOT include information about people/characters that cannot be changed (like ethnicity, gender, etc), but do still include changeable attributes (like hair style).",
    "Include information about lighting.",
    "Include information about camera angle.",
    "Include information about whether there is a watermark or not.",
    "Include information about whether there are JPEG artifacts or not.",
    "If it is a photo you MUST include information about what camera was likely used and details such as aperture, shutter speed, ISO, etc.",
    "Do NOT include anything sexual; keep it PG.",
    "Do NOT mention the image's resolution.",
    "You MUST include information about the subjective aesthetic quality of the image from low to very high.",
    "Include information on the image's composition style, such as leading lines, rule of thirds, or symmetry.",
    "Do NOT mention any text that is in the image.",
    "Specify the depth of field and whether the background is in focus or blurred.",
    "If applicable, mention the likely use of artificial or natural lighting sources.",
    "Do NOT use any ambiguous language.",
    "Include whether the image is sfw, suggestive, or nsfw.",
    "ONLY describe the most important elements of the image.",
    "If it is a work of art, do not include the artist's name or the title of the work.",
    "Identify the image orientation (portrait, landscape, or square) and aspect ratio if obvious.",
    """Use vulgar slang and profanity, such as (but not limited to) "fucking," "slut," "cock," etc.""",
    "Do NOT use polite euphemisms—lean into blunt, casual phrasing.",
    "Include information about the ages of any people/characters when applicable.",
    "Mention whether the image depicts an extreme close-up, close-up, medium close-up, medium shot, cowboy shot, medium wide shot, wide shot, or extreme wide shot.",
    "Do not mention the mood/feeling/etc of the image.",
    "Explicitly specify the vantage height (eye-level, low-angle worm’s-eye, bird’s-eye, drone, rooftop, etc.).",
    "If there is a watermark, you must mention it.",
    """Your response will be used by a text-to-image model, so avoid useless meta phrases like “This image shows…”, "You are looking at...", etc.""",
]

DARK_STYLESHEET = """
QWidget {
    background-color: #2b2b2b;
    color: #f0f0f0;
    border-color: #4f4f4f;
}
QPushButton {
    background-color: #3c3f41;
    color: #f0f0f0;
    border: 1px solid #4f4f4f;
    padding: 5px;
    min-height: 20px;
}
QPushButton:hover {
    background-color: #4f5254;
}
QPushButton:pressed {
    background-color: #2a2d2f;
}
QPushButton:disabled {
    background-color: #353535;
    color: #707070;
}
QLineEdit, QTextEdit, QComboBox {
    background-color: #3c3f41;
    color: #f0f0f0;
    border: 1px solid #4f4f4f;
    padding: 3px;
}
QComboBox QAbstractItemView {
    background-color: #3c3f41;
    color: #f0f0f0;
    selection-background-color: #5a5e60;
    border: 1px solid #4f4f4f;
}
QTextEdit {
    selection-background-color: #5a5e60;
}
QLabel {
    color: #f0f0f0;
    background-color: transparent;
}
QCheckBox {
    color: #f0f0f0;
}
QCheckBox::indicator {
    width: 13px;
    height: 13px;
    border: 1px solid #4f4f4f;
    background-color: #3c3f41;
}
QCheckBox::indicator:checked {
    background-color: #548af7;
}
QGroupBox {
    border: 1px solid #4f4f4f;
    margin-top: 10px;
    font-weight: bold;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 3px 0 3px;
    color: #f0f0f0;
}
QSlider::groove:horizontal {
    border: 1px solid #4f4f4f;
    height: 8px;
    background: #3c3f41;
    margin: 2px 0;
}
QSlider::handle:horizontal {
    background: #8a8a8a;
    border: 1px solid #4f4f4f;
    width: 18px;
    margin: -2px 0;
    border-radius: 3px;
}
QStatusBar {
    background-color: #2b2b2b;
    color: #f0f0f0;
}
QProgressBar {
    border: 1px solid #4f4f4f;
    text-align: center;
    color: #f0f0f0;
    background-color: #3c3f41;
}
QProgressBar::chunk {
    background-color: #548af7;
    width: 10px;
}
QScrollArea {
    border: 1px solid #4f4f4f;
    background-color: #2b2b2b;
}
QScrollBar:horizontal {
    border: 1px solid #4f4f4f;
    background: #3c3f41;
    height: 15px;
    margin: 0px 20px 0 20px;
}
QScrollBar::handle:horizontal {
    background: #8a8a8a;
    min-width: 20px;
    border-radius: 3px;
}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    border: 1px solid #4f4f4f;
    background: #5a5e60;
    width: 20px;
    subcontrol-position: right;
    subcontrol-origin: margin;
}
QScrollBar::sub-line:horizontal {
    subcontrol-position: left;
}
QScrollBar:vertical {
    border: 1px solid #4f4f4f;
    background: #3c3f41;
    width: 15px;
    margin: 20px 0 20px 0;
}
QScrollBar::handle:vertical {
    background: #8a8a8a;
    min-height: 20px;
    border-radius: 3px;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    border: 1px solid #4f4f4f;
    background: #5a5e60;
    height: 20px;
    subcontrol-position: bottom;
    subcontrol-origin: margin;
}
QScrollBar::sub-line:vertical {
    subcontrol-position: top;
}
#ImageDisplayLabel { /* Used for main image preview */
    border: 1px solid #4f4f4f;
    background-color: #1e1e1e;
}
/* Styling for ClickableLabels (thumbnails) */
ClickableLabel {
    border: 2px solid transparent; /* Default transparent border */
    background-color: #3c3f41; /* Darker background for thumbnails */
    padding: 2px;
}
ClickableLabel[selected="true"] {
    border: 2px solid #548af7; /* Blue border for selected thumbnail */
}
QMainWindow {
    background-color: #2b2b2b;
}
#GalleryScrollArea QWidget { /* Ensure background of gallery content widget */
    background-color: #2b2b2b;
}
"""

THUMBNAIL_HEIGHT = 100

# --- Clickable Label for Thumbnails ---
class ClickableLabel(QLabel):
    clicked = pyqtSignal(Path) # Signal to emit when clicked, carrying the image path

    def __init__(self, image_path: Path, parent=None):
        super().__init__(parent)
        self.image_path = image_path
        self.setProperty("selected", False) # For styling
        self.setFrameShape(QFrame.StyledPanel) # Gives a slight sunken look
        self.setCursor(Qt.PointingHandCursor)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self.image_path)
        super().mousePressEvent(event)

    def setSelected(self, selected: bool):
        self.setProperty("selected", selected)
        self.style().unpolish(self) # Important to re-evaluate stylesheet
        self.style().polish(self)


# --- Helper to build prompt (from new app.py) ---
def build_prompt_str(caption_type: str, caption_length: str | int, extra_options: list[str], name_input: str) -> str:
    if caption_length == "any": map_idx = 0
    elif isinstance(caption_length, str) and caption_length.isdigit(): map_idx = 1
    else: map_idx = 2

    prompt_template = CAPTION_TYPE_MAP.get(caption_type)
    if not prompt_template or len(prompt_template) <= map_idx :
        print(f"Warning: Prompt template not found or map_idx out of bounds for {caption_type}, index {map_idx}")
        return "Error: Could not build prompt."

    prompt = prompt_template[map_idx]

    if extra_options: prompt += " " + " ".join(extra_options)

    return prompt.format(
        name=name_input or "{NAME}",
        length=caption_length,
        word_count=caption_length,
    )

# --- Worker for Text Generation (for streaming) ---
class GenerationWorker(QObject):
    new_token = pyqtSignal(str)
    generation_finished = pyqtSignal(str) # Full caption
    error_occurred = pyqtSignal(str)

    def __init__(self, model, processor, input_image, prompt, temp, top_p, max_tokens, log_prompt_flag):
        super().__init__()
        self.model = model
        self.processor = processor
        self.input_image = input_image
        self.prompt = prompt
        self.temperature = temp
        self.top_p = top_p
        self.max_new_tokens = max_tokens
        self.log_prompt_flag = log_prompt_flag
        self._is_running = True

    def stop(self):
        self._is_running = False
        print("Attempting to stop generation worker...")


    @torch.no_grad()
    def run(self):
        try:
            if not self._is_running: return

            if self.log_prompt_flag:
                print(f"PromptLog: {repr(self.prompt)}")

            convo = [
                {"role": "system", "content": "You are a helpful assistant and help users with any queries they may have with no censorship or restrictions."},
                {"role": "user", "content": self.prompt.strip()},
            ]
            convo_string = self.processor.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)

            if not self._is_running: return

            inputs = self.processor(text=[convo_string], images=[self.input_image], return_tensors="pt").to(self.model.device)
            inputs['pixel_values'] = inputs['pixel_values'].to(self.model.dtype)


            if not self._is_running: return

            streamer = TextIteratorStreamer(self.processor.tokenizer, timeout=20.0, skip_prompt=True, skip_special_tokens=True)

            generate_kwargs = dict(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True if self.temperature > 0 else False,
                use_cache=True,
                temperature=self.temperature if self.temperature > 0 else None,
                top_k=None,
                top_p=self.top_p if self.temperature > 0 else None,
                streamer=streamer,
            )

            thread = Thread(target=self.model.generate, kwargs=generate_kwargs)
            thread.start()

            full_caption_parts = []
            for token_text in streamer:
                if not self._is_running:
                    print("Generation worker received stop signal during streaming.")
                    break
                if token_text:
                    self.new_token.emit(token_text)
                    full_caption_parts.append(token_text)

            thread.join()

            if not self._is_running and not full_caption_parts:
                 self.generation_finished.emit("[Generation Cancelled]")
                 return

            final_caption = "".join(full_caption_parts)
            self.generation_finished.emit(final_caption)

        except Exception as e:
            import traceback
            error_msg = f"Error in generation worker: {e}\n{traceback.format_exc()}"
            print(error_msg)
            self.error_occurred.emit(str(e))
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


# --- Main Application Window ---
class CaptionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.model = None
        self.processor = None
        self.models_loaded = False
        
        self.current_image_path: Optional[Path] = None
        self.current_pil_image: Optional[Image.Image] = None
        
        self.generation_thread: Optional[QThread] = None
        self.generation_worker: Optional[GenerationWorker] = None

        self.image_files: List[Path] = [] # List of paths for batch mode
        self.captions_cache: Dict[str, str] = {} # str(image_path): caption_text
        self.is_batch_mode: bool = False
        self.is_generating_batch: bool = False
        self.batch_generation_queue: List[Path] = []
        self.current_batch_item_path: Optional[Path] = None

        self.is_dark_mode_enabled = False
        self.thumbnail_widgets: List[ClickableLabel] = []


        self.logo_label = QLabel()
        try:
            logo_data = base64.b64decode(LOGO_SRC_BASE64)
            logo_pixmap = QPixmap()
            logo_pixmap.loadFromData(logo_data, "SVG")
            self.logo_label.setPixmap(logo_pixmap.scaled(56, 56, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        except Exception as e:
            print(f"Error loading logo: {e}")
            self.logo_label.setText("[Logo]")

        self.title_main_label = QLabel("JoyCaption <span style='font-weight:400'>Beta One</span>")
        self.title_main_label.setStyleSheet("font-size: 1.9rem; margin:0;")
        self.title_sub_label = QLabel("Image-captioning model  |  build mb3500zp")
        self.title_sub_label.setStyleSheet("font-size: 0.9rem; color:#666; margin:2px 0 0;")

        title_vbox = QVBoxLayout()
        title_vbox.addWidget(self.title_main_label)
        title_vbox.addWidget(self.title_sub_label)
        title_vbox.setSpacing(0)

        self.title_hbox = QHBoxLayout()
        self.title_hbox.addWidget(self.logo_label)
        self.title_hbox.addLayout(title_vbox)
        self.title_hbox.addStretch()
        self.title_hbox.setSpacing(16)

        self.setGeometry(50, 50, 1300, 800) 
        self.setMinimumSize(1100, 700)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        self.main_layout.addLayout(self.title_hbox)

        self.content_layout = QHBoxLayout()
        self.main_layout.addLayout(self.content_layout)

        self.initUI()
        self.update_button_states()
        self.update_prompt_display()

    def initUI(self):
        left_scroll_area = QScrollArea()
        left_scroll_area.setWidgetResizable(True)
        left_panel_widget = QWidget()
        left_panel_layout = QVBoxLayout(left_panel_widget)
        left_panel_layout.setSpacing(10)

        # --- Top Buttons ---
        top_buttons_layout = QHBoxLayout()
        self.load_models_button = QPushButton("Load Model")
        self.load_models_button.setToolTip(f"Loads '{MODEL_PATH}'.")
        self.load_models_button.clicked.connect(self.load_models_action)
        self.load_models_button.setMaximumWidth(180)
        top_buttons_layout.addWidget(self.load_models_button)

        self.select_image_button = QPushButton("Select Image")
        self.select_image_button.clicked.connect(self.select_image_action)
        self.select_image_button.setMaximumWidth(180)
        top_buttons_layout.addWidget(self.select_image_button)

        self.load_directory_button = QPushButton("Load Directory")
        self.load_directory_button.clicked.connect(self.load_directory_action)
        self.load_directory_button.setMaximumWidth(180)
        top_buttons_layout.addWidget(self.load_directory_button)
        top_buttons_layout.addStretch()
        left_panel_layout.addLayout(top_buttons_layout)
        
        self.image_path_label = QLabel("No image selected.")
        self.image_path_label.setWordWrap(True)
        left_panel_layout.addWidget(self.image_path_label)

        # --- Image Gallery (for batch mode) ---
        self.gallery_scroll_area = QScrollArea()
        self.gallery_scroll_area.setObjectName("GalleryScrollArea")
        self.gallery_scroll_area.setWidgetResizable(True)
        self.gallery_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.gallery_scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.gallery_widget = QWidget() 
        self.gallery_layout = QVBoxLayout(self.gallery_widget) 
        self.gallery_layout.setAlignment(Qt.AlignTop)
        self.gallery_scroll_area.setWidget(self.gallery_widget)
        self.gallery_scroll_area.setMinimumHeight(150) 
        self.gallery_scroll_area.setVisible(False) 
        left_panel_layout.addWidget(self.gallery_scroll_area)


        # --- Captioning Controls ---
        left_panel_layout.addWidget(QLabel("Caption Type:"))
        self.caption_type_combo = QComboBox()
        self.caption_type_combo.addItems(CAPTION_TYPE_MAP.keys())
        self.caption_type_combo.setCurrentText("Descriptive")
        self.caption_type_combo.currentTextChanged.connect(self.update_prompt_display_slot)
        left_panel_layout.addWidget(self.caption_type_combo)

        left_panel_layout.addWidget(QLabel("Caption Length:"))
        self.caption_length_combo = QComboBox()
        self.caption_length_combo.addItems(CAPTION_LENGTH_CHOICES)
        self.caption_length_combo.setCurrentText("long")
        self.caption_length_combo.currentTextChanged.connect(self.update_prompt_display_slot)
        left_panel_layout.addWidget(self.caption_length_combo)

        self.extra_options_group = QGroupBox("Extra Options")
        self.extra_options_group.setCheckable(True)
        self.extra_options_group.setChecked(False)
        self.extra_options_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding) # Allow to expand vertically
        
        extra_options_layout_container = QVBoxLayout() 
        
        extra_options_scroll = QScrollArea() 
        extra_options_scroll.setWidgetResizable(True)
        extra_options_scroll_widget = QWidget()
        extra_options_layout = QVBoxLayout(extra_options_scroll_widget)

        self.extra_checkboxes = []
        for option_text in EXTRA_OPTIONS_LIST:
            cb = QCheckBox(option_text)
            cb.stateChanged.connect(self.update_prompt_display_slot)
            if option_text == NAME_OPTION:
                cb.stateChanged.connect(self.toggle_name_input_visibility)
            extra_options_layout.addWidget(cb)
            self.extra_checkboxes.append(cb)
        
        extra_options_scroll.setWidget(extra_options_scroll_widget)
        extra_options_layout_container.addWidget(extra_options_scroll)
        self.extra_options_group.setLayout(extra_options_layout_container)
        left_panel_layout.addWidget(self.extra_options_group)


        self.name_input_label = QLabel("Person / Character Name:")
        self.name_input_line = QLineEdit()
        self.name_input_line.setPlaceholderText("e.g., 'the main character'")
        self.name_input_line.textChanged.connect(self.update_prompt_display_slot)
        left_panel_layout.addWidget(self.name_input_label)
        left_panel_layout.addWidget(self.name_input_line)
        self.toggle_name_input_visibility()

        gen_settings_group = QGroupBox("Generation Settings")
        gen_settings_group.setCheckable(True)
        gen_settings_group.setChecked(False)
        gen_settings_layout = QVBoxLayout()

        temp_layout = QHBoxLayout()
        temp_layout.addWidget(QLabel("Temperature (0.0-2.0):"))
        self.temp_slider = QSlider(Qt.Horizontal)
        self.temp_slider.setRange(0, 200)
        self.temp_slider.setValue(60)
        self.temp_slider.setTickInterval(10)
        self.temp_slider.setTickPosition(QSlider.TicksBelow)
        self.temp_value_label = QLabel(f"{self.temp_slider.value()/100:.2f}")
        self.temp_slider.valueChanged.connect(lambda v: self.temp_value_label.setText(f"{v/100:.2f}"))
        temp_layout.addWidget(self.temp_slider)
        temp_layout.addWidget(self.temp_value_label)
        gen_settings_layout.addLayout(temp_layout)

        topp_layout = QHBoxLayout()
        topp_layout.addWidget(QLabel("Top-p (0.0-1.0):"))
        self.topp_slider = QSlider(Qt.Horizontal)
        self.topp_slider.setRange(0, 100)
        self.topp_slider.setValue(90)
        self.topp_slider.setTickInterval(10)
        self.topp_slider.setTickPosition(QSlider.TicksBelow)
        self.topp_value_label = QLabel(f"{self.topp_slider.value()/100:.2f}")
        self.topp_slider.valueChanged.connect(lambda v: self.topp_value_label.setText(f"{v/100:.2f}"))
        topp_layout.addWidget(self.topp_slider)
        topp_layout.addWidget(self.topp_value_label)
        gen_settings_layout.addLayout(topp_layout)

        max_tok_layout = QHBoxLayout()
        max_tok_layout.addWidget(QLabel("Max New Tokens (1-2048):"))
        self.max_tokens_slider = QSlider(Qt.Horizontal)
        self.max_tokens_slider.setRange(1, 2048)
        self.max_tokens_slider.setValue(512)
        self.max_tokens_slider.setTickInterval(256)
        self.max_tokens_slider.setTickPosition(QSlider.TicksBelow)
        self.max_tokens_value_label = QLabel(str(self.max_tokens_slider.value()))
        self.max_tokens_slider.valueChanged.connect(lambda v: self.max_tokens_value_label.setText(str(v)))
        max_tok_layout.addWidget(self.max_tokens_slider)
        max_tok_layout.addWidget(self.max_tokens_value_label)
        gen_settings_layout.addLayout(max_tok_layout)

        gen_settings_group.setLayout(gen_settings_layout)
        left_panel_layout.addWidget(gen_settings_group)

        misc_options_layout = QHBoxLayout()
        self.log_prompt_checkbox = QCheckBox("Log Text Query")
        self.log_prompt_checkbox.setChecked(True)
        misc_options_layout.addWidget(self.log_prompt_checkbox)

        self.dark_mode_button = QPushButton("Enable Dark Mode")
        self.dark_mode_button.clicked.connect(self.toggle_dark_mode)
        misc_options_layout.addWidget(self.dark_mode_button)
        left_panel_layout.addLayout(misc_options_layout)


        left_panel_layout.addStretch(0) # Changed stretch to 0 to allow extra options to expand more
        left_scroll_area.setWidget(left_panel_widget)
        self.content_layout.addWidget(left_scroll_area, 2) 

        # --- Right Panel ---
        right_panel = QVBoxLayout()
        right_panel.setSpacing(10)

        self.image_display_label = QLabel("Image preview will appear here.")
        self.image_display_label.setObjectName("ImageDisplayLabel")
        self.image_display_label.setAlignment(Qt.AlignCenter)
        self.image_display_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.image_display_label.setMinimumSize(300, 300)
        self.image_display_label.setStyleSheet("border: 1px solid gray; background-color: #f0f0f0;")
        right_panel.addWidget(self.image_display_label, 3) 

        right_panel.addWidget(QLabel("Prompt (auto-generated, editable):"))
        self.prompt_display_text = QTextEdit()
        self.prompt_display_text.setPlaceholderText("Prompt will be built here based on your selections.")
        self.prompt_display_text.setAcceptRichText(False)
        self.prompt_display_text.setFixedHeight(100)
        right_panel.addWidget(self.prompt_display_text, 0)

        generation_buttons_layout = QHBoxLayout()
        self.generate_current_button = QPushButton("Generate Current Caption")
        self.generate_current_button.setFixedHeight(40)
        self.generate_current_button.clicked.connect(self.generate_caption_action)
        generation_buttons_layout.addWidget(self.generate_current_button)

        self.generate_batch_button = QPushButton("Generate Batch Captions")
        self.generate_batch_button.setFixedHeight(40)
        self.generate_batch_button.clicked.connect(self.generate_batch_captions_action)
        generation_buttons_layout.addWidget(self.generate_batch_button)
        right_panel.addLayout(generation_buttons_layout)


        right_panel.addWidget(QLabel("Generated Caption (editable):"))
        self.caption_output_text = QTextEdit()
        self.caption_output_text.setReadOnly(False)
        self.caption_output_text.setPlaceholderText("Caption will stream here...")
        right_panel.addWidget(self.caption_output_text, 2)

        save_buttons_layout = QHBoxLayout()
        self.save_caption_button = QPushButton("Save Current Caption")
        self.save_caption_button.clicked.connect(self.save_current_caption_action)
        save_buttons_layout.addWidget(self.save_caption_button)

        self.save_all_captions_button = QPushButton("Save All Batch Captions")
        self.save_all_captions_button.clicked.connect(self.save_all_captions_action)
        save_buttons_layout.addWidget(self.save_all_captions_button)
        right_panel.addLayout(save_buttons_layout)


        self.content_layout.addLayout(right_panel, 1) 

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.progress_bar = QProgressBar()
        self.status_bar.addPermanentWidget(self.progress_bar)
        self.progress_bar.hide()
        self.show_status("Ready. Please load the model first.", 5000)

    def show_status(self, message, timeout=0):
        self.status_bar.showMessage(message, timeout)
        QApplication.processEvents()

    def update_button_states(self):
        self.load_models_button.setEnabled(not self.models_loaded)
        
        is_generating_anything = self.generation_thread and self.generation_thread.isRunning()
        can_start_single_generation = self.models_loaded and self.current_pil_image is not None and not is_generating_anything
        can_start_batch_generation = self.models_loaded and bool(self.image_files) and not self.is_generating_batch and not is_generating_anything

        if is_generating_anything:
            current_op_text = "Generating Batch..." if self.is_generating_batch else "Generating Current..."
            self.generate_current_button.setText(current_op_text)
            self.generate_current_button.setEnabled(False)
            self.generate_batch_button.setEnabled(False)
        else:
            self.generate_current_button.setText("Generate Current Caption")
            self.generate_current_button.setEnabled(can_start_single_generation)
            self.generate_batch_button.setText("Generate Batch Captions")
            self.generate_batch_button.setEnabled(can_start_batch_generation)
        
        self.select_image_button.setEnabled(self.models_loaded and not is_generating_anything)
        self.load_directory_button.setEnabled(self.models_loaded and not is_generating_anything)
        
        for thumb_label in self.thumbnail_widgets:
            thumb_label.setEnabled(not is_generating_anything)

        self.save_caption_button.setEnabled(self.current_image_path is not None and not is_generating_anything)
        self.save_all_captions_button.setEnabled(
            bool(self.captions_cache) and self.is_batch_mode and not is_generating_anything
        )
        
        input_widgets_to_toggle = [
            self.caption_type_combo, self.caption_length_combo, self.extra_options_group,
            self.name_input_line, self.temp_slider, self.topp_slider, self.max_tokens_slider,
            self.prompt_display_text
        ]
        for widget in input_widgets_to_toggle:
            widget.setEnabled(not is_generating_anything)

    def toggle_name_input_visibility(self):
        name_option_checkbox = next((cb for cb in self.extra_checkboxes if cb.text() == NAME_OPTION), None)
        if name_option_checkbox:
            visible = name_option_checkbox.isChecked()
            self.name_input_label.setVisible(visible)
            self.name_input_line.setVisible(visible)

    def update_prompt_display_slot(self):
        QTimer.singleShot(50, self.update_prompt_display)

    def update_prompt_display(self):
        caption_type = self.caption_type_combo.currentText()
        caption_length = self.caption_length_combo.currentText()
        name_input = self.name_input_line.text()
        selected_extras = [cb.text() for cb in self.extra_checkboxes if cb.isChecked()]

        built_prompt = build_prompt_str(caption_type, caption_length, selected_extras, name_input)

        if self.prompt_display_text.toPlainText() != built_prompt:
            self.prompt_display_text.setPlainText(built_prompt)

    def _clear_gallery(self):
        for i in reversed(range(self.gallery_layout.count())):
            widget_to_remove = self.gallery_layout.itemAt(i).widget()
            if widget_to_remove:
                widget_to_remove.setParent(None)
                widget_to_remove.deleteLater()
        self.thumbnail_widgets.clear()
        self.gallery_scroll_area.setVisible(False)

    def _populate_gallery(self):
        self._clear_gallery()
        if not self.image_files or not self.is_batch_mode:
            self.gallery_scroll_area.setVisible(False)
            return

        self.gallery_scroll_area.setVisible(True)
        for img_path in self.image_files:
            try:
                pixmap = QPixmap(str(img_path))
                if pixmap.isNull():
                    thumb_label = ClickableLabel(img_path)
                    thumb_label.setText(f"Err: {img_path.name[:15]}...")
                    thumb_label.setToolTip(f"Error loading thumbnail for {img_path.name}")
                else:
                    scaled_pixmap = pixmap.scaledToHeight(THUMBNAIL_HEIGHT, Qt.SmoothTransformation)
                    thumb_label = ClickableLabel(img_path)
                    thumb_label.setPixmap(scaled_pixmap)
                    thumb_label.setToolTip(img_path.name)
                
                thumb_label.clicked.connect(self._on_thumbnail_clicked)
                self.gallery_layout.addWidget(thumb_label)
                self.thumbnail_widgets.append(thumb_label)
            except Exception as e:
                print(f"Error creating thumbnail for {img_path}: {e}")
        self.gallery_layout.addStretch() 

    def _update_gallery_selection_highlight(self, selected_path: Optional[Path]):
        for thumb_label in self.thumbnail_widgets:
            is_selected = thumb_label.image_path == selected_path
            thumb_label.setSelected(is_selected)


    def _on_thumbnail_clicked(self, image_path: Path):
        if self.generation_thread and self.generation_thread.isRunning():
            QMessageBox.warning(self, "Busy", "Cannot change image while generation is in progress.")
            return
        if image_path in self.image_files:
            idx = self.image_files.index(image_path)
            self._load_image_for_display(image_path, idx) 
        else:
            print(f"Clicked thumbnail path {image_path} not in current batch.")

    def _load_image_for_display(self, image_path: Path, index_in_batch: int = -1) -> bool:
        self.current_image_path = image_path # Set this early
        try:
            self.current_pil_image = Image.open(self.current_image_path).convert("RGB")
            self.display_image(self.current_image_path)
            
            if self.is_batch_mode:
                self._update_gallery_selection_highlight(image_path)
                if index_in_batch != -1:
                    self.image_path_label.setText(
                        f"Batch Image {index_in_batch + 1}/{len(self.image_files)}: {self.current_image_path.name}"
                    )
            else: 
                self.image_path_label.setText(f"Selected: {self.current_image_path.name}")
                self._update_gallery_selection_highlight(None) 

            # --- Caption Loading Logic ---
            caption_text_to_display = ""
            image_path_str = str(self.current_image_path) 

            if image_path_str in self.captions_cache:
                caption_text_to_display = self.captions_cache[image_path_str]
                # print(f"Loaded caption for {image_path_str} from cache.") # Debug
            else:
                caption_file_path = self.current_image_path.with_suffix(self.current_image_path.suffix + ".txt")
                if caption_file_path.exists():
                    try:
                        with open(caption_file_path, "r", encoding="utf-8") as f:
                            caption_text_to_display = f.read()
                        self.captions_cache[image_path_str] = caption_text_to_display 
                        # print(f"Loaded caption for {image_path_str} from file and cached.") # Debug
                    except Exception as e:
                        print(f"Error reading caption file {caption_file_path}: {e}")
                # else:
                    # print(f"No caption file found for {image_path_str}") # Debug
            
            self.caption_output_text.setPlainText(caption_text_to_display)
            if not caption_text_to_display:
                 self.caption_output_text.clear()

            self.show_status(f"Image '{self.current_image_path.name}' loaded.", 3000)
            self.update_button_states()
            return True
        except Exception as e:
            QMessageBox.critical(self, "Image Error", f"Could not load image {image_path.name}: {e}")
            self.current_pil_image = None
            self.current_image_path = None 
            self.image_display_label.setText(f"Error loading {image_path.name}.")
            self.show_status(f"Error loading image: {e}", 5000)
            if self.is_batch_mode:
                 self.image_path_label.setText(f"Error Image (load failed)")
                 self._update_gallery_selection_highlight(None) 
            self.update_button_states()
            return False


    def select_image_action(self):
        if self.generation_thread and self.generation_thread.isRunning():
            QMessageBox.warning(self, "Busy", "Cannot select image while generation is in progress.")
            return
            
        file_path_str, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp *.webp *.gif)")
        if file_path_str:
            file_path = Path(file_path_str)
            
            self.image_files = [] 
            self.is_batch_mode = False
            self._clear_gallery() 
            
            self._load_image_for_display(file_path)

    def load_directory_action(self):
        if self.generation_thread and self.generation_thread.isRunning():
            QMessageBox.warning(self, "Busy", "Cannot load directory while generation is in progress.")
            return

        dir_path_str = QFileDialog.getExistingDirectory(self, "Select Image Directory")
        if dir_path_str:
            dir_path = Path(dir_path_str)
            image_extensions = [".png", ".jpg", ".jpeg", ".bmp", ".webp", ".gif"]
            found_files = sorted([p for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() in image_extensions])

            if not found_files:
                QMessageBox.information(self, "No Images", f"No supported image files found in {dir_path.name}.")
                self.image_path_label.setText("No images found in selected directory.")
                self.is_batch_mode = False
                self.image_files = []
                self._clear_gallery()
            else:
                self.image_files = found_files
                self.is_batch_mode = True
                self.captions_cache.clear() # Clear old cache for the new batch directory
                self._populate_gallery()
                if self.image_files:
                    self._load_image_for_display(self.image_files[0], 0) # Load first image
                self.show_status(f"{len(self.image_files)} images loaded from directory.", 3000)
            
            self.update_button_states()

    def display_image(self, image_path: Path):
        pixmap = QPixmap(str(image_path))
        if pixmap.isNull():
            self.image_display_label.setText("Cannot display image.")
            return

        lbl_w = self.image_display_label.width()
        lbl_h = self.image_display_label.height()
        
        if lbl_w <=0 or lbl_h <= 0:
            self.image_display_label.setPixmap(pixmap) 
            return

        scaled_pixmap = pixmap.scaled(lbl_w, lbl_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_display_label.setPixmap(scaled_pixmap)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.current_image_path and self.image_display_label.pixmap() and not self.image_display_label.pixmap().isNull():
             self.display_image(self.current_image_path)

    def load_models_action(self):
        self.show_status(f"Loading Llava model ({MODEL_PATH})... This may take time.", 0)
        self.progress_bar.setRange(0,0)
        self.progress_bar.show()
        self.load_models_button.setEnabled(False)
        QApplication.processEvents()

        try:
            self.processor = AutoProcessor.from_pretrained(MODEL_PATH)
            self.show_status("Processor loaded. Loading model weights...", 0); QApplication.processEvents()

            device = "cuda" if torch.cuda.is_available() else "cpu"
            torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32
            if device == "cpu" and not hasattr(torch, 'bfloat16'):
                torch_dtype = torch.float32
                print("CPU mode: Using float32 for model.")
            
            print(f"Attempting to load model on device: {device} with dtype: {torch_dtype}")

            self.model = LlavaForConditionalGeneration.from_pretrained(
                MODEL_PATH,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
                device_map="auto"
            )
            self.model.eval()

            if LIGER_AVAILABLE and hasattr(self.model, 'language_model'):
                self.show_status("Applying LIGER kernel...", 0); QApplication.processEvents()
                apply_liger_kernel_to_llama(model=self.model.language_model)

            self.models_loaded = True
            self.show_status("Model loaded successfully!", 5000)
            QMessageBox.information(self, "Model Loaded", f"{MODEL_PATH} loaded.")
        except Exception as e:
            self.models_loaded = False
            error_detail = f"Failed to load model: {e}\nCheck console."
            self.show_status(error_detail, 0)
            QMessageBox.critical(self, "Model Load Error", error_detail)
            import traceback
            traceback.print_exc()
        finally:
            self.progress_bar.hide()
            self.progress_bar.setRange(0,100)
            self.update_button_states()

    def generate_caption_action(self):
        if not self.models_loaded or not self.current_pil_image:
            QMessageBox.warning(self, "Not Ready", "Load model and select an image.")
            return

        if self.generation_thread and self.generation_thread.isRunning():
            QMessageBox.information(self, "Busy", "Generation in progress.")
            return

        self.caption_output_text.clear()
        self.show_status("Generating caption...", 0)
        self.progress_bar.setRange(0,0)
        self.progress_bar.show()

        prompt = self.prompt_display_text.toPlainText()
        temp = self.temp_slider.value() / 100.0
        top_p_val = self.topp_slider.value() / 100.0
        max_tokens = self.max_tokens_slider.value()
        log_prompt = self.log_prompt_checkbox.isChecked()

        self.generation_thread = QThread(self)
        self.generation_worker = GenerationWorker(
            self.model, self.processor, self.current_pil_image, prompt,
            temp, top_p_val, max_tokens, log_prompt
        )
        self.generation_worker.moveToThread(self.generation_thread)

        self.generation_worker.new_token.connect(self.append_token_to_caption)
        self.generation_worker.generation_finished.connect(self.on_generation_finished)
        self.generation_worker.error_occurred.connect(self.on_generation_error)

        self.generation_thread.started.connect(self.generation_worker.run)
        self.generation_thread.finished.connect(self.generation_worker.deleteLater) 
        self.generation_thread.finished.connect(self.generation_thread.deleteLater)

        self.generation_thread.start()
        self.update_button_states()

    def generate_batch_captions_action(self):
        if not self.image_files or not self.is_batch_mode:
            QMessageBox.warning(self, "No Batch", "Load a directory for batch processing.")
            return
        if self.generation_thread and self.generation_thread.isRunning():
            QMessageBox.information(self, "Busy", "Generation process already running.")
            return

        reply = QMessageBox.question(self, "Confirm Batch Generation",
                                     f"Generate captions for all {len(self.image_files)} images?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.No:
            return

        self.is_generating_batch = True
        self.batch_generation_queue = list(self.image_files) 
        self.caption_output_text.clear() 
        self.update_button_states()
        self._start_next_batch_generation_item()

    def _start_next_batch_generation_item(self):
        if not self.batch_generation_queue:
            self.is_generating_batch = False
            self.current_batch_item_path = None
            self.show_status("Batch generation complete.", 5000)
            QMessageBox.information(self, "Batch Complete", f"All {len(self.image_files)} batch captions processed.")
            self.update_button_states()
            if self.image_files: 
                self._load_image_for_display(self.image_files[0], 0)
            return

        self.current_batch_item_path = self.batch_generation_queue.pop(0)
        current_idx_in_full_list = self.image_files.index(self.current_batch_item_path)
        
        self.image_path_label.setText( 
             f"Batch Processing {current_idx_in_full_list + 1}/{len(self.image_files)}: {self.current_batch_item_path.name}"
        )
        self.show_status(f"Batch: Loading {self.current_batch_item_path.name}...")
        
        if not self._load_image_for_display(self.current_batch_item_path, current_idx_in_full_list):
            error_caption = "[Error: Could not load this image for processing]"
            self.captions_cache[str(self.current_batch_item_path)] = error_caption
            self.caption_output_text.setPlainText(error_caption)
            self.show_status(f"Skipping {self.current_batch_item_path.name} due to load error.", 3000)
            QTimer.singleShot(100, self._start_next_batch_generation_item)
            return

        self.generate_caption_action()

    def append_token_to_caption(self, token):
        cursor = self.caption_output_text.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(token)
        self.caption_output_text.ensureCursorVisible()
        QApplication.processEvents()

    def on_generation_finished(self, full_caption):
        current_processed_path = self.current_batch_item_path if self.is_generating_batch else self.current_image_path
        
        if current_processed_path:
            self.captions_cache[str(current_processed_path)] = full_caption
            if self.is_generating_batch:
                 self.show_status(f"Caption for {current_processed_path.name} generated.", 3000)
            else:
                 self.show_status("Caption generation complete.", 5000)
        else: 
            self.show_status("Caption generation finished (unknown image context).", 5000)

        self.progress_bar.hide()
        
        if self.generation_thread and self.generation_thread.isRunning():
            self.generation_thread.quit()
            self.generation_thread.wait(500) 
        self.generation_thread = None 
        self.generation_worker = None
        
        if self.is_generating_batch:
            QTimer.singleShot(100, self._start_next_batch_generation_item)
        else:
            self.update_button_states()


    def on_generation_error(self, error_message):
        error_msg_display = f"Error during generation: {error_message}"
        self.show_status(error_msg_display, 0)
        self.progress_bar.hide()
        QMessageBox.critical(self, "Generation Error", f"An error occurred: {error_message}\nCheck console.")

        current_processed_path = self.current_batch_item_path if self.is_generating_batch else self.current_image_path
        if current_processed_path:
            self.captions_cache[str(current_processed_path)] = f"[Generation Error: {error_message}]"
        
        if self.generation_thread and self.generation_thread.isRunning():
            self.generation_thread.quit()
            self.generation_thread.wait(500)
        self.generation_thread = None
        self.generation_worker = None
        
        if self.is_generating_batch:
            QTimer.singleShot(100, self._start_next_batch_generation_item) 
        else:
            self.update_button_states()

    def save_current_caption_action(self):
        if not self.current_image_path:
            QMessageBox.warning(self, "No Image", "No current image to save caption for.")
            return

        caption_text = self.caption_output_text.toPlainText()
        image_path_str = str(self.current_image_path)
        self.captions_cache[image_path_str] = caption_text 

        caption_file_path = self.current_image_path.with_suffix(self.current_image_path.suffix + ".txt")
        try:
            with open(caption_file_path, "w", encoding="utf-8") as f:
                f.write(caption_text)
            self.show_status(f"Caption saved: {caption_file_path.name}", 3000)
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Could not save caption: {e}")
            self.show_status(f"Error saving caption file: {e}", 5000)

    def save_all_captions_action(self):
        if not self.captions_cache:
            QMessageBox.information(self, "No Captions", "No captions in cache to save.")
            return
        
        if not self.is_batch_mode and len(self.captions_cache) == 1 and self.current_image_path:
             self.save_current_caption_action()
             return

        reply = QMessageBox.question(self, "Confirm Save All",
                                     f"Save all {len(self.captions_cache)} captions in memory to .txt files?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.No:
            return

        saved_count = 0
        error_count = 0
        for image_path_str, caption_text in self.captions_cache.items():
            try:
                image_path = Path(image_path_str)
                caption_file_path = image_path.with_suffix(image_path.suffix + ".txt")
                with open(caption_file_path, "w", encoding="utf-8") as f:
                    f.write(caption_text)
                saved_count += 1
            except Exception as e:
                print(f"Error saving caption for {image_path_str}: {e}")
                error_count += 1
        
        msg = f"Saved {saved_count} captions."
        if error_count > 0:
            msg += f" Failed to save {error_count} (see console)."
        QMessageBox.information(self, "Batch Save Complete", msg)
        self.show_status(msg, 5000)

    def toggle_dark_mode(self):
        self.is_dark_mode_enabled = not self.is_dark_mode_enabled
        if self.is_dark_mode_enabled:
            self.setStyleSheet(DARK_STYLESHEET)
            self.dark_mode_button.setText("Disable Dark Mode")
            self.title_sub_label.setStyleSheet("font-size: 0.9rem; color:#aaa; margin:2px 0 0;")
        else:
            self.setStyleSheet("") 
            self.dark_mode_button.setText("Enable Dark Mode")
            # Reset specific styles that might not revert fully with empty stylesheet
            self.image_display_label.setStyleSheet("border: 1px solid gray; background-color: #f0f0f0;")
            self.title_sub_label.setStyleSheet("font-size: 0.9rem; color:#666; margin:2px 0 0;")
        
        # Re-polish relevant widgets to ensure stylesheet changes apply correctly
        self.image_display_label.style().unpolish(self.image_display_label)
        self.image_display_label.style().polish(self.image_display_label)
        for thumb in self.thumbnail_widgets:
            thumb.style().unpolish(thumb)
            thumb.style().polish(thumb)
        self.gallery_scroll_area.style().unpolish(self.gallery_scroll_area) # And its viewport
        self.gallery_scroll_area.style().polish(self.gallery_scroll_area)
        if self.gallery_scroll_area.widget():
            self.gallery_scroll_area.widget().style().unpolish(self.gallery_scroll_area.widget())
            self.gallery_scroll_area.widget().style().polish(self.gallery_scroll_area.widget())


    def closeEvent(self, event):
        if self.generation_thread and self.generation_thread.isRunning():
            self.show_status("Stopping generation before exit...", 0)
            if self.generation_worker:
                self.generation_worker.stop()
            self.generation_thread.quit()
            if not self.generation_thread.wait(2000):
                print("Generation thread did not stop gracefully, terminating.")
                self.generation_thread.terminate()
                self.generation_thread.wait()
        super().closeEvent(event)


if __name__ == "__main__":
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv)

    window = CaptionApp()
    window.show()
    sys.exit(app.exec_())