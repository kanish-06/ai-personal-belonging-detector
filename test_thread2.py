import threading, pyttsx3, time
def worker():
    try:
        e = pyttsx3.init()
        print('init success')
        e.say('hello world')
        print('say success')
        e.runAndWait()
        print('runAndWait success')
    except Exception as ex:
        print('failed:', ex)
threading.Thread(target=worker).start()
time.sleep(5)
