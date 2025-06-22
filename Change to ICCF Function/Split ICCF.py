from matplotlib import pyplot as plt
import pandas as pd
from datetime import datetime
import numpy as np
import random
import statsmodels.api as sm


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
    #print(f"{len(filtered_values1)}")
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
        diff2 = diff2[lag:]
        min_len -= lag
    elif lag < 0:
        diff1 = diff1[-lag:]
        min_len += lag
        
    # Ensure min_len is non-negative
    min_len = max(0, min_len)

    # Calculate the cross-correlation for the overlapping portion of the lists
    product_sum = np.sum(np.array(diff1[:min_len]) * np.array(diff2[:min_len]))

    # Divide the sum of products by the product of standard deviations
    result = product_sum / ((std1 * std2) * len(filtered_values1))
    
    return result

def auto_correlation(delay,price,date,file):
    plt.plot(delay,iccfs_to_tau(price,date,price,date), label = f"{file}")
    plt.xlabel("Time Lag [Days]");plt.ylabel("ICCF Value");plt.title("ACF of Residuals")
    plt.legend()
    #plt.xlim(-800, 800)
    #plt.show()

def detrend_acf(delay,data,dates,file, f):
    detrend1 = sm.nonparametric.lowess(data, dates, f)
    plt.plot(delay,iccfs_to_tau(detrend1[:, 0],detrend1[:, 1],detrend1[:, 0],detrend1[:, 1]), color = "red", label = "Loess Smoothed ACF")
    plt.xlabel("Time Lag [Days]");plt.ylabel("ICCF Value");plt.title("ACF of "+ file)
    plt.legend()
    plt.show()



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
    plt.plot(dates, data, ls = "none", marker = "o", ms = 2, label = "Raw Data")
    plt.axhline(y=np.mean(data), color='r', linestyle='--', label='Local Mean')
    plt.xlabel("Dates [Days]")
    plt.ylabel("Relative Prices [USD]")
    #print(f"{title} {np.mean(data)}")
    #plt.legend()
    #plt.show()

def detrend_plots(title,dates,data, f):
    detrend1 = sm.nonparametric.lowess(data, dates, frac=f)
    #plt.title(title)
    plt.plot(detrend1[:, 0], detrend1[:, 1], color='red', label='Lowess Smoothing')
    plt.xlabel("Dates [Days]")
    plt.ylabel("Stock Price [USD]")
    plt.legend()
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
    subsample_length = int(0.7 * original_length)
    
    unique_indices = set()

    while len(unique_indices) < subsample_length:
        # Generate a random index
        random_index = random.randint(0, original_length - 1)

        # Check if the index is unique
        if random_index not in unique_indices:
            unique_indices.add(random_index)

    # Create the subsample by selecting elements from list1 based on the unique indices
    subsample_p = [list1[i] for i in unique_indices]
    
    # Create subsample_iccf by selecting corresponding elements from list2
    subsample_t = [list2[i] for i in unique_indices]

    return subsample_p, subsample_t

def centroid_function(y_values, time_lags):
    # Find the peak and its index
    peak_value = max(y_values)
    peak_index = np.argmax(y_values)
    print(peak_index)
    peak_time_lag = time_lags[peak_index]

    # Find the threshold value (80% of the peak)
    threshold1 = 0.75 * peak_value
    threshold2 = 0.85* peak_value
    # Find the indices where the ICCF intersects the threshold
    condition = np.logical_and(y_values >= threshold1, y_values <= threshold2)
    intersection_indices = np.where(condition)[0]
    print(intersection_indices)
    # Find the two points closest to 80% of the peak
    left_indicie = 
    right_indicie = 
    # Find the indices of the two closest values
    closest_indices = np.array([left_indicie,right_indicie])
    
    # Get the corresponding values
    closest_values = [intersection_indices[i] for i in closest_indices]

    # Find the centroid (average of the time lags corresponding to the two points)
    centroid = np.mean([time_lags[point] for point in closest_values])

    return peak_time_lag, centroid
    
def centroid_stdv(delay,p1,d1,p2,d2):
    tc_values = []
    tp_values = []
    delay= delay/0.49
    for i in range(1000):
        sample_p1, sample_d1= generate_subsample(p1, d1)
        sample_p2, sample_d2 = generate_subsample(p2, d2)
        
        # generates a new sub sample
        iccfs = iccfs_to_tau(sample_p1, sample_d1, sample_p2, sample_d2)
        #Works out the iccfs per tau
        tp_values.append(centroid_function(iccfs,delay)[0])
        tc_values.append(centroid_function(iccfs,delay)[1])
        #records the tau centroid and tau peak
        # #Plotting the raw generated data7
        # plt.subplot(2 ,2,1)
        # plots(f"TSM Sub-sample {i}",sample_d1,sample_p1)
        # plt.subplot(2 ,2,2)
        # plots(f"INTC Sub-sample {i}",sample_d2,sample_p2)
        
        #Plotting the subsample data
        #plt.subplot(2 ,2,3)
        # plt.plot(delay,iccfs)
        # plt.xlabel("Time Lag [Days]");plt.ylabel("ICCF Value");#plt.title("ICCF Values of "+ file1 + " against "+file2)
        # plt.text(-450, 0.4, f'$\tau_p$ = {round(centroid_function(iccfs,delay)[0],2)}', fontsize=12, color='red')
        # plt.text(-450, 0.3, f'$\tau_c$ = {round(centroid_function(iccfs,delay)[1],2)}', fontsize=12, color='red')
        # plt.xlim(-1600,1600)
        # plt.show()
    print(f"$\tau_c$ = {np.mean(tc_values)}")
    print(f"$\tau_p$ = {np.mean(tp_values)}")
    tau_c_std = np.std(tc_values); tau_p_std = np.std(tp_values)
    plt.subplot(2 ,1,1)
    plt.hist(tc_values, bins=30, edgecolor='black')
    plt.xlabel("Centroid Values");plt.ylabel("Frequency");plt.title(r"Distribution of $\tau_c$ from 1000 generated sub-samples")
    plt.subplot(2 ,1,2)
    plt.hist(tp_values, bins=30, edgecolor='black')
    plt.xlabel("Peak Values");plt.ylabel("Frequency");plt.title(r"Distribution of $\tau_p$ from 1000 generated sub-samples")
    plt.subplots_adjust( hspace =0.7)
    plt.show()
    return tau_p_std, tau_c_std


def interpolate_iccf(iccf_values, tau_values, target_tau):
    """
    Interpolate the ICCF value at a given tau.

    Parameters:
    - iccf_values (numpy.ndarray): Array containing ICCF values.
    - tau_values (numpy.ndarray): Array containing corresponding tau values.
    - target_tau (float): Target tau value for interpolation.

    Returns:
    - float: Interpolated ICCF value at the target tau.
    """
    interpolated_iccf = np.interp(target_tau, tau_values, iccf_values)
    return interpolated_iccf

def anti_corr_centroid(y_values, time_lags):
    # Find the peak and its index
    peak_value = min(y_values)
    peak_index = np.argmin(y_values)
    peak_time_lag = time_lags[peak_index]

    # Find the threshold value (80% of the peak)
    threshold = 0.8 * peak_value

    # Find the indices where the ICCF intersects the threshold
    intersection_indices = np.where(y_values <= threshold)[0]

    # Find the two points closest to 80% of the peak on either side
    closest_points = [intersection_indices[0],intersection_indices[-1]]

    # Ensure that the two closest points are on either side of the peak


    # Find the centroid (average of the time lags corresponding to the two points)
    centroid = np.mean([time_lags[point] for point in closest_points])

    return peak_time_lag, centroid

    
data_frac = 0.35

file1 = "TSM.csv"
file2 = "INTC.csv"
data1 = pd.read_csv(file1, skiprows=1).dropna()
data2 = pd.read_csv(file2, skiprows=1).dropna()
# cleans the data of N/A values
dates1 = data1.values[:,0];dates2 = data2.values[:,0] # Turning dates into an array
file1 = "TSM"
file2 = "INTC"
   
open_price_1 =  data1.values[:,1];open_price_2 =  data2.values[:,1]#Retrieves open price data
close_price_1 = data1.values[:,4];close_price_2 = data2.values[:,4]#Retrieves close price data
ave_price_1 = average_lists(open_price_1,close_price_1);ave_price_2 = average_lists(open_price_2,close_price_2)
start = '2018-05-20' ; end = '2023-12-06'
dates1,ave_price_1 = date_selection(start, end,dates1,ave_price_1)
dates2,ave_price_2 = date_selection(start, end,dates2,ave_price_2)
converted_dates1 = dates_days(dates1, dates1[0]);converted_dates2 = dates_days(dates2, dates1[0])


#Averages the data for the day

tau = np.arange(-converted_dates1[-1],converted_dates1[-1],1) #Time lag to run through


# plots(f"Stock Market Data for {file1}", converted_dates1, ave_price_1); detrend_plots(f"Smoothed Data for {file1}", converted_dates1, ave_price_1,data_frac)
# plots(f"Stock Market Data for {file2}", converted_dates2, ave_price_2); detrend_plots(f"Smoothed Data for {file2}", converted_dates2, ave_price_2,data_frac)
# auto_correlation(tau, ave_price_1, converted_dates1, file1);#detrend_acf(tau, ave_price_1, converted_dates1, file1,data_frac)
# auto_correlation(tau, ave_price_2, converted_dates2, file2);#detrend_acf(tau, ave_price_2, converted_dates2, file2,data_frac)

smooth1 = sm.nonparametric.lowess(ave_price_1, converted_dates1, frac=data_frac)
sm_p1 = np.array(ave_price_1)-smooth1[:,1]
smooth2 = sm.nonparametric.lowess(ave_price_2, converted_dates2, frac=data_frac)
sm_p2 = np.array(ave_price_2)-smooth2[:,1]
 
# plots(f"Residuals After LOESS Detrending of {file1}", converted_dates1, sm_p1);plt.show()
# plots(f"Residuals After LOESS Detrending of {file2}", converted_dates2, sm_p2);plt.show()

#auto_correlation(tau, sm_p1, converted_dates1, f"Residuals of {file1}");auto_correlation(tau, sm_p2, converted_dates2, f"Residuals of {file2}");plt.show()

# auto_correlation(tau, sm_p1, converted_dates1, f"Auto Correlation of residuals of {file1}")
# auto_correlation(tau, sm_p2, converted_dates2, f"Auto Correlation of residuals of {file2}")

iccfs = iccfs_to_tau(ave_price_1,converted_dates1,ave_price_2,converted_dates2)
smoothed_iccfs = iccfs_to_tau(sm_p1,smooth1[:,0],sm_p2,smooth2[:,0])

smoothed_tau = centroid_function(smoothed_iccfs,tau)
#optimal_tau = centroid_function(iccfs,tau)

anit_corr_tau = anti_corr_centroid(smoothed_iccfs,tau)

plt.plot(tau,smoothed_iccfs, color = "red", label = "ICCF of Smoothed Data")
plt.plot(tau,iccfs, label = "ICCF of Raw Data")

plt.axhline(y=0.8*max(smoothed_iccfs), linestyle='--',color = "r", label='0.8*Peak of Smoothed Data')
# # plt.axhline(y=0.8*max(iccfs), linestyle='--', label='0.8*Peak')
# # plt.axhline(y=0.8*min(smoothed_iccfs), linestyle='--', label='0.8*Peak')
plt.axvline(x = smoothed_tau[1])
plt.axvline(x = smoothed_tau[0])
# # plt.axvline(x = anit_corr_tau[1])
plt.xlabel("Time Lag [Days]");plt.ylabel("ICCF Value");plt.title("ICCF of "+ file1 + " and "+file2);plt.show()
# #iccf values for the data for each tau



#for i in range(1):
    # print(f"Time period from {start} to {end}")
    # print(f"ICCF(Tau Peak) = {interpolate_iccf(iccfs,tau,optimal_tau[0])}")
    # print(f"Tau Peak = {optimal_tau[0]}  \nTau Centroid = {optimal_tau[1]}")
    # print(f"Anti Correlation Tau Peak = {anit_corr_tau[0]}  \n Anti Correlation Tau Centroid = {anit_corr_tau[1]}")
    # print(f"For the smoothed data\nTau Peak = {(smoothed_tau[0])} \nTau Centroid = {smoothed_tau[1]}")
    
    # opt_tau_stdv = centroid_stdv(tau, ave_price_1, converted_dates1, ave_price_2, converted_dates2)   
    # print(f"Anti CorrelationTau Peak = {anit_corr_tau[0]} ± {round(opt_tau_stdv[0],2)} \
    #    \nAnti CorrelationTau Centroid = {anit_corr_tau[1]} ± {round(opt_tau_stdv[1],2)}")
    
    # sm_tau_stdv = centroid_stdv(tau,sm_p1,smooth1[:,0],sm_p2,smooth2[:,0])
    # print(f"For the smoothed data\n Anti Correlation Tau Peak = {(anit_corr_tau[0])} ± {round(sm_tau_stdv[0],2)}\
    #           \n Anti Correlation Tau Centroid = {anit_corr_tau[1]} ± {round(sm_tau_stdv[1],2)}")
       
    # opt_tau_stdv = centroid_stdv(tau, ave_price_1, converted_dates1, ave_price_2, converted_dates2) 
    # print(f"Tau Peak = {optimal_tau[0]} ± {round(opt_tau_stdv[0],2)} \
    #       \nTau Centroid = {optimal_tau[1]} ± {round(opt_tau_stdv[1],2)}")
sm_tau_stdv = centroid_stdv(tau,sm_p1,smooth1[:,0],sm_p2,smooth2[:,0])
print(f"For the smoothed data\nTau Peak = {(smoothed_tau[0])} ± {round(sm_tau_stdv[0],2)} \
      \nTau Centroid = {smoothed_tau[1]} ± {round(sm_tau_stdv[1],2)}")

    # print("------------------------------------------------")

