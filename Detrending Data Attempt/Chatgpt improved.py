from matplotlib import pyplot as plt
import pandas as pd
from datetime import datetime
import numpy as np
import random

def dates_days(date_strings, start_date):
    start = datetime.strptime(start_date, '%Y-%m-%d')
    return [(datetime.strptime(date, '%Y-%m-%d') - start).days + 1 for date in date_strings]

def average_lists(list1, list2):
    if len(list1) != len(list2):
        raise ValueError("Input lists must have the same length")
    
    return [(val1 + val2) / 2.0 for val1, val2 in zip(list1, list2)]

def cross_correlation(values1, days1, values2, days2, lag):
    overlapping_days = set(days1) & set(days2)
    filtered_values1 = [val1 for val1, day1 in zip(values1, days1) if day1 in overlapping_days]
    filtered_values2 = [val2 for val2, day2 in zip(values2, days2) if day2 in overlapping_days]
    
    mean1, mean2 = np.mean(filtered_values1), np.mean(filtered_values2)
    std1, std2 = np.std(filtered_values1), np.std(filtered_values2)

    diff1 = np.array(filtered_values1) - mean1
    diff2 = np.array(filtered_values2) - mean2

    min_len = len(overlapping_days)
    if lag > 0:
        diff1, diff2 = diff1[:-lag], diff2[lag:]
    elif lag < 0:
        diff1, diff2 = diff1[-lag:], diff2[:lag]

    product_sum = np.sum(diff1[:min_len] * diff2[:min_len])
    result = product_sum / (std1 * std2) / len(filtered_values1)
    
    return result

def process_data(values, times, batch_size=10):
    num_batches = len(values) // batch_size
    remaining_values = len(values) % batch_size

    averages = [np.mean(values[i:i+batch_size]) for i in range(0, len(values)-remaining_values, batch_size)]
    std_deviations = [np.std(values[i:i+batch_size], ddof=0) for i in range(0, len(values)-remaining_values, batch_size)]
    time_values = [times[i+batch_size//2] for i in range(0, len(times)-remaining_values, batch_size)]

    # Handle remaining values at the end
    if remaining_values > 0:
        remaining_values_data = values[-remaining_values:]
        remaining_values_times = times[-remaining_values:]
        averages.append(np.mean(remaining_values_data))
        std_deviations.append(np.std(remaining_values_data, ddof=0))
        time_values.append(remaining_values_times[remaining_values // 2])

    return averages, std_deviations, time_values

def detrend_data_non_overlapping(data, n):
    detrended_data = []

    for i in range(0, len(data), n):
        subset = data[i:i + n]
        subset_average = sum(subset) / len(subset) if len(subset) > 0 else 0

        for point in subset:
            detrended_point = point - subset_average
            detrended_data.append(detrended_point)

    return detrended_data

def plots(title,dates,data):
    plt.title(title)
    plt.plot(dates, data, ls = "none", marker = "o", ms = 2)
    plt.axhline(y=np.mean(data), color='r', linestyle='--', label='Horizontal Line')
    plt.xlabel("Dates [Days]")
    plt.ylabel("Stock Price [GBP]")
    plt.show()

def iccfs_to_tau(p1,d1,p2,d2):
    iccf_val = []
    for i in tau:
        iccf_val.append(cross_correlation(p1, d1, p2,d2,i))

    return iccf_val

def date_selection(start_date, end_date, date_list, value_list):
    start_datetime = datetime.strptime(start_date, "%Y-%m-%d")
    end_datetime = datetime.strptime(end_date, "%Y-%m-%d")

    filtered_dates = []
    filtered_values = []

    for date, value in zip(date_list, value_list):
        current_datetime = datetime.strptime(date, "%Y-%m-%d")

        if start_datetime <= current_datetime <= end_datetime:
            filtered_dates.append(date)
            filtered_values.append(value)

    return filtered_dates, filtered_values

def generate_subsample(list1, list2, subsample_percentage=0.8):
    original_length = len(list1)
    subsample_length = int(subsample_percentage * original_length)
    
    unique_indices = set(random.sample(range(original_length), subsample_length))

    subsample_1 = [list1[i] for i in unique_indices]
    subsample_2 = [list2[i] for i in unique_indices]

    return subsample_1, subsample_2

def centroid_function(y_values, time_lags):
    # Find the peak and its index
    peak_value = max(y_values)
    peak_index = np.argmax(y_values)
    peak_time_lag = time_lags[peak_index]

    # Find the threshold value (80% of the peak)
    threshold = 0.8 * peak_value

    # Find the indices where the ICCF intersects the threshold
    intersection_indices = np.where(y_values >= threshold)[0]

    # Find the two points closest to 80% of the peak
    closest_points = sorted(intersection_indices, key=lambda i: abs(i - peak_index))[:2]

    # Find the centroid (average of the time lags corresponding to the two points)
    centroid = np.mean([time_lags[point] for point in closest_points])

    return peak_time_lag, centroid
    
def centroid_stdv(delay,p1,d1,p2,d2):
    tc_values = []
    tp_values = []
    for i in range(40):
        sample_p1, sample_d1= generate_subsample(p1, d1)
        sample_p2, sample_d2 = generate_subsample(p2, d2)
        # generates a new sub sample
        iccfs = iccfs_to_tau(sample_p1, sample_d1, sample_p2, sample_d2)
        #Works out the iccfs per tau
        tc_values.append(centroid_function(iccfs,delay)[0])
        tp_values.append(centroid_function(iccfs,delay)[1])
        #records the tau centroid and tau peak
        """
        #Plotting the subsample data
        plt.plot(delay,iccfs)
        plt.xlabel("Time Lag [Days]");plt.ylabel("ICCF Value");plt.title("ICCF Values of "+ file1 + " against "+file2)
        plt.text(-500, 0.5, f'$\tau_p$ = {centroid_function(iccfs,delay)[0]}', fontsize=12, color='red')
        plt.text(-500, 0.4, f'$\tau_c$ = {centroid_function(iccfs,delay)[1]}', fontsize=12, color='red')
        plt.show()
        """
    tau_c_std = np.std(tc_values); tau_p_std = np.std(tp_values)
    return tau_c_std, tau_p_std



file1 = "RKT.csv"
file2 = "GSK.csv"
data1 = pd.read_csv(file1, skiprows=1).dropna()
data2 = pd.read_csv(file2, skiprows=1).dropna()
# cleans the data of N/A values
dates1 = data1.values[:,0];dates2 = data2.values[:,0] # Turning dates into an array

   
open_price_1 =  data1.values[:,1];open_price_2 =  data2.values[:,1]#Retrieves open price data
close_price_1 = data1.values[:,4];close_price_2 = data2.values[:,4]#Retrieves close price data
ave_price_1 = average_lists(open_price_1,close_price_1);ave_price_2 = average_lists(open_price_2,close_price_2)
start = '2021-11-27' ; end = '2023-11-27'
dates1,ave_price_1 = date_selection(start, end,dates1,ave_price_1)
dates2,ave_price_2 = date_selection(start, end,dates2,ave_price_2)
converted_dates1 = dates_days(dates1, dates1[0]);converted_dates2 = dates_days(dates2, dates1[0])
#Averages the data for the day

plots(file1, converted_dates1, ave_price_1); plots(file2, converted_dates2, ave_price_2)


tau = np.arange(-converted_dates1[-1],converted_dates1[-1],1) #Time lag to run through

plt.plot(tau,iccfs_to_tau(ave_price_1,converted_dates1,ave_price_2,converted_dates2))
plt.xlabel("Time Lag [Days]");plt.ylabel("ICCF Value");plt.title("ICCF Values of "+ file1 + " against "+file2)
plt.show()
#iccf values for the data for each tau

print(centroid_function(iccfs_to_tau(ave_price_1,converted_dates1,ave_price_2,converted_dates2),tau))
print(centroid_stdv(tau, ave_price_1, converted_dates1, ave_price_2, converted_dates2))