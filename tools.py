import os
import pickle
import quandl
import numpy as np

np.set_printoptions(precision=2)

##################################
# DOWNLOAD DATASETS
##################################
# Lets build a Python function to extract the adjusted price using the Python APIs. The function should be able to
# cache calls and specify an initial and final time stamp to get the historical data beyond the symbol
def date_obj_to_str(date_obj):
	return date_obj.strftime('%Y-%m-%d')


def save_pickle(something, path):
	if not os.path.exists(os.path.dirname(path)):
		os.makedirs(os.path.dirname(path))
	with open(path, 'wb') as fh:
		pickle.dump(something, fh, pickle.DEFAULT_PROTOCOL)


def load_pickle(path):
	with open(path, 'rb') as fh:
		pkl = pickle.load(fh)
	return pkl


# The return object of the function `fetch_stock_price` is a mono-dimensional array, containing 
# the stock price for the requested symbol, ordered from the `from_date` to `to_date`
# Caching is done within the function, that is, if there's a cache miss, the the quandl API is called
# the `date_to_str` object function is just a helper function, to convert datetime.date to the correct
# string format needed for the API
def fetch_stock_price(symbol,
					  from_date,
					  to_date,
					  cache_path="./tmp/prices/"):
	assert(from_date <= to_date)

	filename = "{}_{}_{}.pk".format(symbol, str(from_date), str(to_date))
	price_filepath = os.path.join(cache_path, filename)

	try: 
		prices = load_pickle(price_filepath)
		print("loaded from", price_filepath)

	except IOError:
		historic = quandl.get("WIKI/" + symbol,
							  start_date=date_obj_to_str(from_date),
							  end_date=date_obj_to_str(to_date))

		prices = historic["Adj. Close"].tolist()
		save_pickle(prices, price_filepath)
		print("saved into", price_filepath)

	return prices

# Testing fetch stock price function
# print(fetch_stock_price("GOOG", datetime.date(2017,1,1), datetime.date(2017,1,31)))

###########################################
# FORMAT DATASETS
##########################################

def format_dataset(values, temporal_features):
	feat_splits = [values[i: i+temporal_features] for i in range(len(values) - temporal_features)]
	feats = np.vstack(feat_splits)
	labels = np.array(values[temporal_features:])
	return feats, labels


# Function to reshape matrices to mono-dimensional (1D) array
def matrix_to_array(m):
	return np.asarray(m).reshape(-1)

