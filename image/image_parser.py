import time
import os
from selenium import webdriver
from selenium.webdriver.common.by import By
import requests

def download_images_from_google(query, num_images, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Настройка Selenium и драйвера
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")  # Без графического интерфейса
    driver = webdriver.Chrome(options=options)

    # Преобразуем запрос в URL
    url = f"https://www.google.com/search?hl=en&tbm=isch&q={query}"
    driver.get(url)

    # Прокрутим страницу вниз для загрузки изображений
    last_height = driver.execute_script("return document.body.scrollHeight")
    count = 0
    while count < num_images:
        # Прокрутка страницы вниз
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:  # Если достигнут конец страницы
            break
        last_height = new_height

    # Извлекаем ссылки на изображения
    img_elements = driver.find_elements(By.TAG_NAME, "img")
    image_urls = [img.get_attribute('src') for img in img_elements if img.get_attribute('src')]

    # Скачиваем изображения
    count = 0
    for img_url in image_urls:
        if count >= num_images:
            break
        try:
            img_data = requests.get(img_url).content
            image_name = os.path.join(save_dir, f"image_{count + 1}.jpg")
            with open(image_name, 'wb') as file:
                file.write(img_data)
                print(f"Downloaded {image_name}")
            count += 1
        except Exception as e:
            print(f"Error downloading image: {e}")
            continue

    driver.quit()
    print(f"Downloaded {count} images")

# Пример использования
download_images_from_google("real sun", 200, "real_sun")
