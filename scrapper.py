from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from PIL import Image
import pytesseract
import io
import requests
import json
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Setup Selenium WebDriver with headless option
options = Options()
options.add_argument("--headless")
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# Open the main complaint website
url = "https://idyllic-cajeta-deb5e8.netlify.app/"
driver.get(url)

# Wait for the complaint list to be present
try:
    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "complaint-list")))
except Exception as e:
    logging.error("Complaint list not found: %s", e)
    driver.quit()
    exit()

# Extract complaint links
complaint_links = driver.find_elements(By.CSS_SELECTOR, "ul#complaint-list li a")
complaints_data = []

if not complaint_links:
    logging.info("No complaints found.")
else:
    for index, complaint in enumerate(complaint_links, start=1):
        complaint_text = complaint.text.strip()
        complaint_url = complaint.get_attribute("href")
        logging.info("Processing complaint %d: %s", index, complaint_text)

        # Open complaint page
        driver.get(complaint_url)

        # Extract details from the complaint page
        try:
            title = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "h1"))).text.strip()
            main_section = driver.find_element(By.TAG_NAME, "main")
            paragraphs = main_section.find_elements(By.TAG_NAME, "p")

            # Extract all paragraphs within the main section
            body_content = "\n".join([paragraph.text.strip() for paragraph in paragraphs])

            # Extract date from <strong> tag within the main section
            try:
                date_element = main_section.find_element(By.TAG_NAME, "strong")
                date_text = date_element.text.strip()
                # Format the date
                date_obj = datetime.strptime(date_text, "%B %d, %Y")
                formatted_date = date_obj.strftime("%B %d, %Y")
            except Exception as e:
                logging.warning("Date not found or invalid format: %s", e)
                formatted_date = "Date not found"

            # Filter out unwanted lines
            unwanted_lines = [
                "© 2025 Crime Awareness Forum. All rights reserved.",
                "No image found or failed to extract text from image.",
                "Body content copied to clipboard."
            ]
            body_content = "\n".join([line for line in body_content.split("\n") if line not in unwanted_lines])

            # Extract text from image if present
            try:
                image_element = main_section.find_element(By.TAG_NAME, "img")
                image_url = image_element.get_attribute("src")
                image_response = requests.get(image_url)
                image = Image.open(io.BytesIO(image_response.content))
                image_text = pytesseract.image_to_string(image)
                body_content += "\n" + image_text
            except Exception as e:
                logging.info("No image found or failed to extract text from image: %s", e)

            complaints_data.append({
                "title": title,
                "date": formatted_date,
                "link": complaint_url,
                "details": body_content
            })

        except Exception as e:
            logging.error("Failed to fetch details for complaint %d: %s", index, e)
            complaints_data.append({
                "title": complaint_text,
                "link": complaint_url,
                "details": "Failed to fetch details"
            })

        # Go back to the main page
        driver.back()

# Save data to JSON
with open("complaints_data.json", "w", encoding="utf-8") as json_file:
    json.dump(complaints_data, json_file, ensure_ascii=False, indent=4)

# Close the WebDriver
driver.quit()

logging.info("Scraping completed! Data saved to complaints_data.json")
