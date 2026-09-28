import threading, pyttsx3, time
def worker():
    try:
        e = pyttsx3.init()
        print('init success')
    except Exception as ex:
        print('init failed:', ex)
threading.Thread(target=worker).start()
time.sleep(1)
