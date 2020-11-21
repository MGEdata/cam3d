# GET user key
# in python
import http.client

conn = http.client.HTTPSConnection("api.snumat.com")
headers = {
    'authorization': "Basic { YOUR-KEY(ID:986798607@qq.com, PW:AAss125800) }",
}

conn.request("GET", "/v1/auth", headers=headers)

res = conn.getresponse()
data = res.read()

print(data.decode("utf-8"))

# Return
# {
#  key : "{ KEY }",
#  date : "{ YYYY.MM.DD, hh:mm:ss }"
# }