from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC
from pathlib import Path
import time
import os

def wait_for_file(file_path: Path, timeout=20):
    """Wait until file exists (for downloads)."""
    end_time = time.time() + timeout
    while time.time() < end_time:
        if file_path.exists():
            return file_path
        time.sleep(0.5)
    raise TimeoutError(f"{file_path.name} not downloaded in {timeout}s")


def download_mof_nodes_linkers(cif_path, download_path="downloads", debug=False):
    download_dir = Path(download_path).absolute()
    download_dir.mkdir(parents=True, exist_ok=True)

    # Setup Chrome options
    options = Options()
    
    prefs = {
        "download.default_directory": str(download_dir),
        "download.prompt_for_download": False,
        "directory_upgrade": True,
        "safebrowsing.enabled": True
    }
    options.add_experimental_option("prefs", prefs)
    options.headless = not debug  # Headless if not debugging
    if not debug:
        options.add_argument("--headless=new")  # use the "new" headless mode
    driver = webdriver.Chrome(options=options)
    wait = WebDriverWait(driver, 10)  # 10s wait for elements

    try:
        driver.get("https://snurr-group.github.io/web-mofid/sbu.html")

        # Wait for file input and upload CIF
        file_input = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='file']")))
        file_input.send_keys(str(Path(cif_path).absolute()))

        # Click submit button if present
        submit_btn = wait.until(EC.element_to_be_clickable(
            (By.CSS_SELECTOR, "button, input[type='submit'], input[type='button']")
        ))
        submit_btn.click()

        # Select Visualization algorithm
        vis_algo_dropdown = wait.until(EC.presence_of_element_located((By.ID, "folderdownload")))
        Select(vis_algo_dropdown).select_by_visible_text("Metal-Oxo")

        # Select download dropdown
        download_dropdown = wait.until(EC.presence_of_element_located((By.ID, "cifdownload")))
        select_download = Select(download_dropdown)

        # Download nodes.cif
        select_download.select_by_visible_text("nodes.cif")
        download_button = wait.until(EC.element_to_be_clickable((By.ID, "rundownload")))
        download_button.click()
        nodes_file = wait_for_file(download_dir / "nodes.cif")

        # Download linkers.cif
        select_download.select_by_visible_text("linkers.cif")
        download_button.click()
        linkers_file = wait_for_file(download_dir / "linkers.cif")

    finally:
        driver.quit()

    return nodes_file, linkers_file


