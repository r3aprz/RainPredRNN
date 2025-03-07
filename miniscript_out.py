import os
import subprocess
import time

def clear_screen():
    if os.name == 'nt':  # Windows
        os.system('cls')
    else:  # Linux/macOS
        os.system('clear')

def show_file_content():
    try:
        subprocess.run(['cat', 'RainOut.out'], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Errore durante l'esecuzione del comando: {e}")
        exit(1)

if __name__ == "__main__":
    try:
        while True:  # Ciclo infinito
            clear_screen()
            show_file_content()
            time.sleep(2)  # Attendi 2 secondi prima di ripetere
    except KeyboardInterrupt:
        print("\nScript interrotto dall'utente.")