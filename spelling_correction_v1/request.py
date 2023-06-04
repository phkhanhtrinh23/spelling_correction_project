import requests

url = 'http://0.0.0.0:8016/api/spelling_correction/'
myobj = {'text': "I didn't receive even one leter from her."}

x = requests.post(url, json = myobj)

print(x.text)