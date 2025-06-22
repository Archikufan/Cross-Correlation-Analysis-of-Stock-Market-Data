from matplotlib import pyplot as plt
import pandas as pd
from datetime import datetime
import numpy as np
import random


#changes the date format
def dates_days(date_strings, start_date):
    # Convert the start_date string to a datetime object
    start = datetime.strptime(start_date, '%Y-%m-%d')
    
    # Create a list to store the consecutive day numbers
    day_numbers = []
    
    for date_string in date_strings:
        # Convert the input date string to a datetime object
        date = datetime.strptime(date_string, '%Y-%m-%d')
        
        # Calculate the difference in days and add 1 to start from day 1
        day_number = (date - start).days + 1
        
        day_numbers.append(day_number)
    
    return day_numbers

#Function to find the mean of the open and close
def average_lists(list1, list2):
    # Check if the input lists have the same length
    if len(list1) != len(list2):
        raise ValueError("Input lists must have the same length")
    
    # Initialize an empty list to store the averages
    averages = []
    
    # Iterate through the lists and calculate the average at each position
    for value1, value2 in zip(list1, list2):
        avg = (value1 + value2) / 2.0
        averages.append(avg)
    
    return averages

#CCF function with time lag
def cross_correlation(values1, days1, values2, days2, lag):
    # Check for overlapping days
    overlapping_days = set(days1) & set(days2)
    
    # Filter the data to include only overlapping days
    filtered_values1 = [val1 for val1, day1 in zip(values1, days1) if day1 in overlapping_days]
    filtered_values2 = [val2 for val2, day2 in zip(values2, days2) if day2 in overlapping_days]
    
    # Calculate the mean of each list
    mean1 = np.mean(filtered_values1)
    mean2 = np.mean(filtered_values2)

    # Calculate the standard deviation of each list
    std1 = np.std(filtered_values1)
    std2 = np.std(filtered_values2)
    
    # Calculate the differences from the mean for each list
    diff1 = [x - mean1 for x in filtered_values1]
    diff2 = [y - mean2 for y in filtered_values2]

    # Determine the size of the overlapping portion of the lists
    min_len = len(overlapping_days)
    # Apply time lag by shifting one of the lists
    if lag > 0:
        diff1 = diff1[:-lag]
        diff2 = diff2[lag:]
    elif lag < 0:
        diff1 = diff1[-lag:]
        diff2 = diff2[:lag]

    # Calculate the cross-correlation for the overlapping portion of the lists
    product_sum = np.sum(np.array(diff1[:min_len]) * np.array(diff2[:min_len]))

    # Divide the sum of products by the product of standard deviations
    result = product_sum / (std1 * std2)/len(filtered_values1)
    
    return result

# Attempt at function to bin the data
def process_data(values, times):
    averages = []
    std_deviations = []
    time_values = []

    for i in range(0, len(values), 10):
        # Take 10 consecutive values starting from index i
        sub_values = values[i:i+10]
        sub_times = times[i:i+10]

        # Calculate the average and standard deviation of the 10 values
        average = np.mean(sub_values)
        std_deviation = np.std(sub_values, ddof=0)

        # Calculate the time value as the middle value of the 10 values
        time_value = sub_times[len(sub_times) // 2]

        # Append the results to their respective lists
        averages.append(average)
        std_deviations.append(std_deviation)
        time_values.append(time_value)

    # Handle remaining values at the end
    if len(values) % 10 != 0:
        remaining_values = values[len(values) - len(values) % 10:]
        remaining_times = times[len(times) - len(times) % 10:]
        average = np.mean(remaining_values)
        std_deviation = np.std(remaining_values, ddof=0)
        time_value = remaining_times[len(remaining_times) // 2]
        averages.append(average)
        std_deviations.append(std_deviation)
        time_values.append(time_value)

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

def generate_subsample(list1, list2):
    original_length = len(list1)
    subsample_length = int(0.8 * original_length)
    
    unique_indices = set()

    while len(unique_indices) < subsample_length:
        # Generate a random index
        random_index = random.randint(0, original_length - 1)

        # Check if the index is unique
        if random_index not in unique_indices:
            unique_indices.add(random_index)

    # Create the subsample by selecting elements from list1 based on the unique indices
    subsample_iccf = [list1[i] for i in unique_indices]
    
    # Create subsample_iccf by selecting corresponding elements from list2
    subsample_t = [list2[i] for i in unique_indices]

    return subsample_iccf, subsample_t

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
start = '2022-10-27' ; end = '2023-11-27'
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


"""
subset_size = 100
detrended_1 = detrend_data_non_overlapping(ave_price_1, subset_size);detrended_2 = detrend_data_non_overlapping(ave_price_2, subset_size);


plots(f"Detrended data for {file1}", converted_dates1, detrended_1)
plots(f"Detrended data for {file2}", converted_dates2, detrended_2)
"""