import threading, pyttsx3, time, queue

q = queue.Queue()

def worker():
    try:
        e = pyttsx3.init()
        print('init success')
    except Exception as ex:
        print('init failed:', ex)
        return
        
    while True:
        text = q.get()
        if text == "STOP":
            break
        print('saying:', text)
        e.say(text)
        e.runAndWait()
        print('done saying:', text)

threading.Thread(target=worker, daemon=True).start()
q.put("hello")
time.sleep(2)
q.put("world")
time.sleep(2)
q.put("STOP")
