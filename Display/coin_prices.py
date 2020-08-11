import requests
from datetime import date

url = 'https://api.coindesk.com/v1/bpi/currentprice.json'
response = requests.get(url)
response_json = response.json()


def curr_price(coin):
    price = response_json[coin]['USD']['rate']
    return price


print('Price on ', date.today(), ' is ', curr_price('bpi'))
