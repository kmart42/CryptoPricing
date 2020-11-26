from iexfinance.stocks import Stock
import os
API_TOKEN = 'pk_6d9d1f79677f4997a0461860456dde6a'
API = 'sk_c16116aba40c4aef808adba5de00f554'
os.environ['IEX_API_VERSION'] = 'iexcloud-sandbox'
os.environ['IEX_TOKEN'] = API_TOKEN

price = Stock("aapl", token=API_TOKEN)
ticker = 'MSFT'
company = Stock(ticker, output_format='pandas')
df_income = company.get_income_statement(period="quarter", last='4')
print(df_income)
