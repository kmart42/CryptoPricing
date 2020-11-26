import requests
from datetime import date

url = 'https://api.coindesk.com/v1/bpi/currentprice.json'
TICKER_API_URL = 'https://api.coinmarketcap.com/v1/ticker/'
response = requests.get(url)
response_json = response.json()


def daily_price(coin):
    price = response_json[coin]['USD']['rate']
    return price


def curr_price(coin):
    response2 = requests.get(TICKER_API_URL + coin)
    response_json2 = response2.json()

    return float(response_json2[0]['price_usd'])


# print('Price on ', date.today(), ' is ', curr_price('bpi'))

# curr_price('bitcoin')

def main():
    last_price = -1
    while True:
        coin = 'bpi'
        price = daily_price(coin)
        if price != last_price:
            print(price)
            last_price = price


main()
