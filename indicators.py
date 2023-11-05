import numpy as np
def simple_moving_average(prices, window):
    """
    Calculate the Simple Moving Average.

    :param prices: List of price points
    :param window: Number of periods to consider for the moving average
    :return: List of simple moving average points
    """
    # Ensure the window size is valid
    if window > len(prices) or window < 1:
        raise ValueError("The window size must be between 1 and the size of the data set.")
    
    # Calculate the moving average
    sma = []
    for i in range(len(prices) - window + 1):
        average = sum(prices[i: i + window]) / window
        sma.append(average)

    return sma

def simple_moving_average_minus_price(prices, window):
    """
    Calculate the Simple Moving Average.

    :param prices: List of price points
    :param window: Number of periods to consider for the moving average
    :return: List of simple moving average points
    """
    # Ensure the window size is valid
    if window > len(prices) or window < 1:
        raise ValueError("The window size must be between 1 and the size of the data set.")
    # Calculate the moving average
    sma = []
    for i in range(len(prices) - window + 1):
        average = sum(prices[i: i + window]) / window
        sma.append(average)

    sma = sma-prices[window-1:]
    return sma

def exponential_moving_average(prices, window):
    """
    Calculate the Exponential Moving Average.
    
    :param prices: List of price points
    :param window: The window size
    :return: A list of EMA values
    """
    if len(prices) < window:
        raise ValueError("The number of price points must be at least as large as the window size.")

    ema_values = []
    sma = sum(prices[:window]) / window  # First value is SMA
    multiplier = 2 / (window + 1)
    ema_values.append(sma)

    for price in prices[window:]:
        ema = (price - ema_values[-1]) * multiplier + ema_values[-1]
        ema_values.append(ema)


    return ema_values

def exponential_moving_average_minus_price(prices, window):
    """
    Calculate the Exponential Moving Average.
    
    :param prices: List of price points
    :param window: The window size
    :return: A list of EMA values
    """
    if len(prices) < window:
        raise ValueError("The number of price points must be at least as large as the window size.")

    ema_values = []
    sma = sum(prices[:window]) / window  # First value is SMA
    multiplier = 2 / (window + 1)
    ema_values.append(sma)

    for price in prices[window:]:
        ema = (price - ema_values[-1]) * multiplier + ema_values[-1]
        ema_values.append(ema)

    ema_values = ema_values-prices[window-1:]
    return ema_values

def weighted_moving_average(prices, window):
    """
    Calculate the Weighted Moving Average.
    
    :param prices: List of price points
    :param window: The window size
    :return: A list of WMA values
    """
    if len(prices) < window:
        raise ValueError("The number of price points must be at least as large as the window size.")

    wma_values = []
    for i in range(len(prices) - window + 1):
        weights = list(range(1, window + 1))
        window_prices = prices[i:i + window]
        wma = sum(weight * price for weight, price in zip(weights, window_prices)) / sum(weights)
        wma_values.append(wma)


    return wma_values

def weighted_moving_average_minus_price(prices, window):
    """
    Calculate the Weighted Moving Average.
    
    :param prices: List of price points
    :param window: The window size
    :return: A list of WMA values
    """
    if len(prices) < window:
        raise ValueError("The number of price points must be at least as large as the window size.")

    wma_values = []
    for i in range(len(prices) - window + 1):
        weights = list(range(1, window + 1))
        window_prices = prices[i:i + window]
        wma = sum(weight * price for weight, price in zip(weights, window_prices)) / sum(weights)
        wma_values.append(wma)

    wma_values = wma_values-prices[window-1:]
    return wma_values

def moving_average_convergence_divergence(prices, fast=12, slow=26, signal=9):
    """
    Calculate MACD, Signal line, and MACD histogram.

    :param prices: List of price points
    :param fast: Fast period
    :param slow: Slow period
    :param signal: Signal period
    :return: Tuple of lists (MACD, Signal line, MACD histogram)
    """
    if len(prices) < slow:
        raise ValueError("Not enough price points.")

    ema_fast_values = exponential_moving_average(prices, fast)
    ema_slow_values = exponential_moving_average(prices, slow)

    macd = [fast - slow for fast, slow in zip(ema_fast_values, ema_slow_values[fast-slow:])]
    signal_line = exponential_moving_average(macd, signal)
    macd_histogram = [m - s for m, s in zip(macd[signal-1:], signal_line)]

    return macd_histogram


def parabolic_sar(high, low, start=0.02, increment=0.02, maximum=0.2):
    length = len(high)
    sar = [0] * length
    trend = [None] * length
    ep = [0] * length
    af = [0] * length
    
    sar[0] = low[0]
    trend[0] = "up"
    
    for i in range(1, length):
        if high[i] > high[i-1]:
            trend[i] = "up"
            ep[i] = high[i]
            if trend[i-1] == "up":
                af[i] = min(af[i-1] + increment, maximum)
            else:
                af[i] = start
        else:
            trend[i] = "down"
            ep[i] = low[i]
            if trend[i-1] == "down":
                af[i] = min(af[i-1] + increment, maximum)
            else:
                af[i] = start
        
        if trend[i] == "up":
            sar[i] = sar[i-1] + af[i] * (ep[i] - sar[i-1])
            if sar[i] > low[i]:
                sar[i] = low[i]
                trend[i] = "down"
                ep[i] = low[i]
                af[i] = start
        else:
            sar[i] = sar[i-1] + af[i] * (ep[i] - sar[i-1])
            if sar[i] < high[i]:
                sar[i] = high[i]
                trend[i] = "up"
                ep[i] = high[i]
                af[i] = start
                
    return sar


# use case = 1) Senkou Span A - B , 2) price - higher (a/b), 3) , tenkan_sen - kijun_sen, chikou_span- stock_prices

# need high, low , close for this thing
def ichimoku_cloud(high, low, close, tenkan_window=9, kijun_window=26, senkou_window=52, chikou_shift=26):
    tenkan_sen = [(max(high[i-tenkan_window:i]) + min(low[i-tenkan_window:i])) / 2 for i in range(tenkan_window, len(close)+1)]
    kijun_sen = [(max(high[i-kijun_window:i]) + min(low[i-kijun_window:i])) / 2 for i in range(kijun_window, len(close)+1)]
    
    senkou_span_a = [(tenkan + kijun) / 2 for tenkan, kijun in zip(tenkan_sen, kijun_sen)]
    senkou_span_b = [(max(high[i-senkou_window:i]) + min(low[i-senkou_window:i])) / 2 for i in range(senkou_window, len(close)+1)]
    
    chikou_span = close[:-chikou_shift]
    
    return tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, chikou_span

def aroon_oscillator(close, window=25):
    aroon_up = [((window - (i - close[i-window:i].index(max(close[i-window:i])))) / window) * 100 for i in range(window, len(close))]
    aroon_down = [((window - (i - close[i-window:i].index(min(close[i-window:i])))) / window) * 100 for i in range(window, len(close))]
    
    oscillator = [up - down for up, down in zip(aroon_up, aroon_down)]
    return oscillator

def relative_strength_index(price, period):
    deltas = np.diff(price)
    seed = deltas[:period+1]
    up = seed[seed >= 0].sum()/period
    down = -seed[seed < 0].sum()/period
    rs = up/down
    rsi = np.zeros_like(price)
    rsi[:period] = 100. - 100./(1. + rs)

    for i in range(period, len(price)):
        delta = deltas[i - 1]  # cause the diff is 1 shorter

        if delta > 0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta

        up = (up*(period - 1) + upval)/period
        down = (down*(period - 1) + downval)/period

        rs = up/down
        rsi[i] = 100. - 100./(1. + rs)

    return rsi




# Using the function with our hypothetical data and a 5-day window
stock_prices = [20.00, 20.15, 20.30, 20.45, 20.25, 20.10, 19.90, 19.75, 19.60, 19.80]
moving_averages = simple_moving_average(stock_prices, 5)

for i, ma in enumerate(moving_averages, 1):
    print(f"Day {i + 4}: SMA = ${ma:.2f}")  # the first SMA value corresponds to the 5th day (window size)