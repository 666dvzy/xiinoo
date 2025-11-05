import time
import requests
from typing import List
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import TimeoutException, StaleElementReferenceException

# --- CONFIGURACIÓN ---
URL_TO_SCRAPE = 'https://gamblingcounting.com/evolution-auto-roulette'
API_URL = 'https://roulette-ai-production.up.railway.app'  # ← TU URL AQUÍ
CSS_SELECTOR_ALL_NUMBERS = ".live-game-page__block__results.live-game-page__block__results--roulette.live-game-page__block__content div.roulette-number.roulette-number--small"
XPATH_SELECTOR_AGE_GATE_BUTTON = "//button[contains(., 'Sí, tengo 18 años')]"
XPATH_SELECTOR_COOKIE_BUTTON = "//button[contains(text(), 'Accept')]"

# --- AJUSTES DE TIEMPO Y CANTIDAD ---
TARGET_INITIAL_NUMBERS = 200 # Cantidad esperada de números en el historial inicial completo
INITIAL_LOAD_TIMEOUT = 60 # Timeout en segundos para la carga inicial
MONITORING_TIMEOUT = 120 # Timeout en segundos para el monitoreo de nuevos números
MONITORING_POLL_FREQUENCY = 0.1 # Frecuencia de sondeo para detectar nuevos números
INITIAL_SEND_DELAY = 0.05 # Pequeño retraso entre el envío de números iniciales (en segundos)
REQUEST_TIMEOUT = 60 # <-- AUMENTADO: Timeout para las peticiones al backend (en segundos)


# --- FUNCIONES AUXILIARES ---

def send_number_to_ia(number: int):
    """Envía un número a la API del backend para ser procesado."""
    process_url = f"{API_URL}/process_number"
    try:
        # Usar el timeout aumentado definido en las configuraciones
        requests.post(process_url, json={'number': number}, timeout=REQUEST_TIMEOUT) # <-- Usando REQUEST_TIMEOUT
        # print(f"-> Número {number} enviado al backend.")
    except requests.exceptions.RequestException as e:
        print(f"-> ERROR DE CONEXIÓN al enviar {number} a {process_url}: {e}")

def get_all_numbers_as_list(driver) -> List[int]:
    """
    Obtiene la lista actual de números de la página web.
    Maneja StaleElementReferenceException reintentando la búsqueda.
    Retorna la lista encontrada o [] si falla consistentemente.
    """
    attempts = 0
    max_attempts = 5
    while attempts < max_attempts:
        try:
            numbers = []
            elements = driver.find_elements(By.CSS_SELECTOR, CSS_SELECTOR_ALL_NUMBERS)

            if not elements:
                 time.sleep(0.5)
                 attempts += 1
                 continue

            processed_count = 0
            for el in elements:
                 try:
                    text = el.text.strip()
                    if text.isdigit():
                        numbers.append(int(text))
                    processed_count += 1
                 except StaleElementReferenceException:
                    print("  -> get_all_numbers_as_list: StaleElementReferenceException durante la iteración.")
                    break

            if processed_count == len(elements) or not elements:
                return numbers

            attempts += 1


        except StaleElementReferenceException:
            print(f"  -> get_all_numbers_as_list: StaleElementReferenceException durante find_elements, intento {attempts + 1}")
            time.sleep(0.2)
            attempts += 1
        except Exception as e:
            print(f"  -> get_all_numbers_as_list: Error inesperado: {e}, intento {attempts + 1}")
            time.sleep(0.2)
            attempts += 1

    print(f"  -> get_all_numbers_as_list: Falló al obtener números después de {max_attempts} intentos.")
    return []

# --- FUNCIÓN PRINCIPAL ---

def main():
    print("Iniciando el recolector con undetected_chromedriver...")
    options = uc.ChromeOptions()
    options.add_argument("--disable-background-timer-throttling")
    options.add_argument("--disable-backgrounding-occluded-windows")
    options.add_argument("--disable-renderer-backgrounding")
    # options.add_argument("--headless")
    # options.add_argument("--no-sandbox")
    # options.add_argument("--disable-dev-shm-usage")

    try:
        driver = uc.Chrome(options=options, use_subprocess=True, detected_executable_path=True)
        print("undetected_chromedriver inicializado.")
    except Exception as e:
        print(f"ERROR: No se pudo inicializar undetected_chromedriver ({e}). Intentando con Selenium estándar...")
        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.service import Service
            service = Service()
            options = Options()
            options.add_argument("--disable-background-timer-throttling")
            options.add_argument("--disable-backgrounding-occluded-windows")
            options.add_argument("--disable-renderer-backgrounding")
            driver = webdriver.Chrome(service=service, options=options)
            driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
                "source": """
                    Object.defineProperty(navigator, 'webdriver', {
                      get: () => undefined
                    })
                """
            })
            print("Selenium estándar inicializado.")
        except Exception as se:
            print(f"ERROR CRÍTICO: No se pudo inicializar ningún driver ({se}). Saliendo.")
            return


    driver.get(URL_TO_SCRAPE)

    # --- Manejo de banners ---
    try:
        WebDriverWait(driver, 15).until(lambda d: d.find_element(By.XPATH, XPATH_SELECTOR_AGE_GATE_BUTTON)).click()
        print("Banner de edad aceptado.")
        time.sleep(1)
    except TimeoutException:
        print("No se encontró banner de edad o ya fue manejado.")
    except Exception as e: print(f"Error al manejar banner de edad: {e}")

    try:
        WebDriverWait(driver, 15).until(lambda d: d.find_element(By.XPATH, XPATH_SELECTOR_COOKIE_BUTTON)).click()
        print("Banner de cookies aceptado.")
        time.sleep(1)
    except TimeoutException:
        print("No se encontró banner de cookies o ya fue manejado.")
    except Exception as e: print(f"Error al manejar banner de cookies: {e}")


    # --- Carga inicial de números ---
    print(f"Esperando la carga inicial de exactamente {TARGET_INITIAL_NUMBERS} números...")
    try:
        WebDriverWait(driver, INITIAL_LOAD_TIMEOUT).until(
            lambda d: len(get_all_numbers_as_list(d)) == TARGET_INITIAL_NUMBERS
        )
        print(f"Detectados {TARGET_INITIAL_NUMBERS} números en la carga inicial.")

    except TimeoutException:
        found_count_on_timeout = len(get_all_numbers_as_list(driver))
        print(f"ERROR CRÍTICO: Timeout ({INITIAL_LOAD_TIMEOUT}s) esperando {TARGET_INITIAL_NUMBERS} números. Se encontraron {found_count_on_timeout} al expirar el tiempo.")
        driver.quit()
        return

    initial_numbers = get_all_numbers_as_list(driver)
    if len(initial_numbers) != TARGET_INITIAL_NUMBERS:
         print(f"ADVERTENCIA: Después de la espera, la lista obtenida tiene {len(initial_numbers)} números en lugar de {TARGET_INITIAL_NUMBERS}. Esto puede ocurrir si hubo un Timeout o si la lista cambió justo después de la espera.")

    if not initial_numbers:
        print("ERROR CRÍTICO: No se pudo leer la lista inicial después de la espera. Lista vacía.")
        driver.quit()
        return

    print(f"Carga masiva de {len(initial_numbers)} números iniciales al backend con un retraso de {INITIAL_SEND_DELAY}s entre envíos...")
    # Envía los números iniciales (los más recientes primero)
    for number in reversed(initial_numbers):
        send_number_to_ia(number)
        time.sleep(INITIAL_SEND_DELAY)


    last_known_list = initial_numbers
    print(f"--- Carga inicial completada. Monitoreando en tiempo real para nuevos números... ---")


    # --- Bucle principal para monitorear nuevos números ---
    while True:
        try:
            print(f"Vigilando... (esperando cambio en el primer número, actualmente {last_known_list[0] if last_known_list else 'N/A'}...)")

            WebDriverWait(driver, MONITORING_TIMEOUT, poll_frequency=MONITORING_POLL_FREQUENCY).until(
                lambda d: last_known_list and d.find_elements(By.CSS_SELECTOR, CSS_SELECTOR_ALL_NUMBERS) and d.find_elements(By.CSS_SELECTOR, CSS_SELECTOR_ALL_NUMBERS)[0].text.strip() != str(last_known_list[0])
            )
            print("  -> Cambio detectado en el primer número. Obteniendo lista actualizada.")

            current_list = get_all_numbers_as_list(driver)

            if current_list and last_known_list and current_list[0] != last_known_list[0]:
                new_number = current_list[0]
                print(f"\n¡Nueva tirada detectada! El número es: {new_number}!")
                send_number_to_ia(new_number)
                last_known_list = current_list
            elif current_list:
                 last_known_list = current_list
            else:
                 print("  -> get_all_numbers_as_list retornó lista vacía después de detectar un posible cambio.")


        except TimeoutException:
            print(f"Timeout de vigilancia ({MONITORING_TIMEOUT}s). No se detectó nuevo número en el primer elemento. Refrescando página.")
            try:
                driver.refresh()
                time.sleep(5)
                last_known_list = get_all_numbers_as_list(driver)
                if not last_known_list:
                     print("  -> ERROR: No se pudo obtener la lista después de refrescar. Intentando obtenerla en el siguiente ciclo.")
                else:
                     print(f"  -> Página refrescada. Monitoreando de nuevo desde el primer número: {last_known_list[0]}.")

            except Exception as refresh_error:
                print(f"Error al refrescar la página: {refresh_error}. Saliendo.")
                break

        except StaleElementReferenceException:
            print("Stale element exception detectada fuera de get_all_numbers_as_list. Re-obteniendo lista para recuperar el estado.")
            last_known_list = get_all_numbers_as_list(driver)
            if not last_known_list:
                print("  -> ERROR: No se pudo obtener la lista después de StaleElementReferenceException. Reintentando en el siguiente ciclo.")
            else:
                 print(f"  -> Lista re-obtenida. Monitoreando de nuevo desde el primer número: {last_known_list[0]}.")


        except Exception as e:
            print(f"Error inesperado durante el monitoreo: {e}. Refrescando y reintentando...")
            try:
                driver.refresh()
                time.sleep(10)
                last_known_list = get_all_numbers_as_list(driver)
                if not last_known_list:
                     print("  -> ERROR: No se pudo obtener la lista después del error inesperado y refresco. Reintentando en el siguiente ciclo.")
                else:
                     print(f"  -> Recuperado de error inesperado. Monitoreando de nuevo desde el primer número: {last_known_list[0]}.")
            except Exception as recovery_error:
                print(f"No se pudo refrescar o recuperar después del error inesperado: {recovery_error}. Saliendo.")
                break

    driver.quit()

if __name__ == '__main__':
    main()

