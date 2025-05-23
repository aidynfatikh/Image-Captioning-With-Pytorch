import subprocess

def launch_streamlit():
    subprocess.Popen([
        "streamlit", "run", "app.py",
        "--server.runOnSave", "false",
        "--logger.level", "error"
    ])

if __name__ == "__main__":
    launch_streamlit()
