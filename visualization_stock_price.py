
import datetime
import matplotlib.pyplot as plt
from tools import fetch_stock_price

###########################################
# VISUALIZE DATASETS
##########################################
symbols = ["MSFT", "KO", "AAL", "MMM", "AXP", "GE", "GM", "JPM", "UPS"] # companies's code
plt.figure(figsize=(20,10))
ax = plt.subplot(1,1,1)
for sym in symbols:
	prices = fetch_stock_price(sym, datetime.date(2015, 1,1), datetime.date(2016, 12,31))
	ax.plot(range(len(prices)), prices, label=sym)

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels)
plt.xlabel("Trading days since 2015-1-1", fontsize=18)
plt.ylabel("Stock price [$]", fontsize=18)
plt.title("Prices of some American stocks in trading days of 2015 and 2016", fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.savefig("./graph/stock_prices_2015_2016.png")
plt.show()
