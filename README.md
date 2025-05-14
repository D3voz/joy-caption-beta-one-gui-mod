Vram usage- 17.4 GB For full load, 10 gb for 4-bit

## Installation

Follow these steps to set up the JoyCaption GUI Mod on your system:

1.  **Clone the Repository:**
    Open your terminal or command prompt and run:
    ```bash
    git clone https://github.com/D3voz/joy-caption-beta-one-gui-mod
    cd joycaption-beta-one-gui-mod
    ```
2.  **Create and Activate a Virtual Environment (Recommended):**

    # For Python 3.10 (adjust if your default Python is different or use 'python3.10 -m venv venv' or 'py -3.10 -m venv venv')
    python -m venv venv
    ```
    Activate the environment:
    *   **Windows (CMD/PowerShell):**
       
        venv\Scripts\activate
        
    *   **macOS/Linux (Bash/Zsh):**
        
        source venv/bin/activate
      
3.  **Install Triton (for Windows, Python 3.10/3.11):**
    Triton can offer performance benefits. The provided wheel is for specific Python versions on Windows.
    *   **For Python 3.10 on Windows:**
        ```bash
        pip install https://github.com/woct0rdho/triton-windows/releases/download/v3.1.0-windows.post8/triton-3.1.0-cp310-cp310-win_amd64.whl
    *   **For Linux/macOS or other Python versions:**
        pip install triton
4.  **Install Core Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

5.  **Upgrade Transformers and Tokenizers (Recommended):**
    ```bash
    pip install --upgrade transformers tokenizers
    ```

## Usage

1.  Ensure your virtual environment is activated
2.  python Run_GUI.py


##Side note
MAke sure to install  Visual Studio with C++ Build Tools  and Add Visual Studio Compiler Paths to System PATH if you have not done it already. 
   
