import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve as sig_convolve
from scipy import signal
from scipy.signal import filtfilt
import csv


def SeizureAnalysis(PathOfFileToBeProcessed, FilteredDataFolderPath, FilterCoeffDesignedInMatlab):
    print("Program starting...")
    # RawRecordingData = '//Users//Winjoy//Desktop//SD20s.txt'
    inputFileName = PathOfFileToBeProcessed.split("\\")[-1].split(".")[-2]

    print("extracting raw data from " + inputFileName + "...")
    df = pd.read_csv(PathOfFileToBeProcessed, names=['sec', 'CH1', 'CH2', 'CH3', 'CH4', 'extra'])

    # extracting csv data and converting to array
    time = np.array(list(map(float, (df.sec.tolist())[:])))
    EEG_EEG2 = np.array(list(map(float, (df.CH1.tolist())[:])))
    EEG_EEG3 = np.array(list(map(float, (df.CH2.tolist())[:])))
    EMG = np.array(list(map(float, (df.CH3.tolist())[:])))
    EDF = np.array(list(map(float, (df.CH4.tolist())[:])))

    # Check that data is extracted properly
    # print(time)
    # print(EEG_EEG2)
    # print(EEG_EEG3)
    # print(EMG)
    # print(EDF)

    print("plotting raw data with frequency information...")
    # Plot the time varying data for each channel on the left side of the plot
    # https://matplotlib.org/stable/gallery/subplots_axes_and_figures/subplots_demo.html
    fig1, axs1 = plt.subplots(4, 2)
    fig1.suptitle('Raw Recording Data')
    axs1[0, 0].plot(time, EEG_EEG2)
    axs1[0, 0].set_ylabel('EEG2')
    axs1[1, 0].plot(time, EEG_EEG3)
    axs1[1, 0].set_ylabel('EEG3')
    axs1[2, 0].plot(time, EMG)
    axs1[2, 0].set_ylabel('EMG')
    axs1[3, 0].plot(time, EDF)
    axs1[3, 0].set_ylabel('EDF')
    axs1[3, 0].set_xlabel('Time (s)')

    # Plot FFT power spectrums of the various channels to the right of the plot
    # https://www.ritchievink.com/blog/2017/04/23/understanding-the-fourier-transform-by-example/
    # shared variables
    fs = 2000.0  # Hz
    T = 1.0 / fs
    SampleSize = EEG_EEG3.size  # all data share the same length
    xf = np.linspace(0, 1 / T, SampleSize)

    # take fft of channels
    fft_EEG_EEG3 = np.fft.fft(EEG_EEG3)
    fft_EEG_EEG2 = np.fft.fft(EEG_EEG2)
    fft_EMG = np.fft.fft(EMG)
    fft_EDF = np.fft.fft(EDF)

    # show the power spectrum of the fft for each channel to the right of the matching time varying plots above
    #Link for normalization: https://stackoverflow.com/questions/31153563/normalization-while-computing-power-spectral-density
    axs1[0, 1].plot(xf[:SampleSize // 2], np.abs(fft_EEG_EEG3)[:SampleSize // 2] * 1 / SampleSize)
    axs1[0, 1].set_xlim([0, 80])
    axs1[1, 1].plot(xf[:SampleSize // 2], np.abs(fft_EEG_EEG2)[:SampleSize // 2] * 1 / SampleSize)
    axs1[1, 1].set_xlim([0, 80])
    axs1[2, 1].plot(xf[:SampleSize // 2], np.abs(fft_EMG)[:SampleSize // 2] * 1 / SampleSize)
    axs1[3, 1].plot(xf[:SampleSize // 2], np.abs(fft_EDF)[:SampleSize // 2] * 1 / SampleSize)
    axs1[3, 1].set_xlabel('Frequency (Hz)')

    # Differentiating the different brain frequencies
    # Filters: https://scipy-cookbook.readthedocs.io/items/ApplyFIRFilter.html

    print("Acquiring filter coefficients...")
    # Read in filter coefficients design in mathlab
    # FilterCoeffDesignedInMatlab = '//Users//Winjoy//Desktop//A_B_D_G_T_H.txt'
    df = pd.read_csv(FilterCoeffDesignedInMatlab, names=['Alpha', 'Beta', 'Delta', 'Gamma', 'Theta', 'HPF', 'extra'])
    # File contains Coeff for FIR BPF with Hamming window Fs = 2000 Hz, Filter Order 1000, Brainwaves freq defined below
    FilterOrder = 1001
    ### Coeffs for filters
    # Notch Filters (NF): https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.iirnotch.html
    f0 = 60.0  # Frequency to be removed from signal (Hz)
    Q = 30.0  # Quality factor
    # Design notch filter
    Notch60Hz_b_Coeff, Notch60Hz_a_Coeff = signal.iirnotch(f0, Q, fs)
    # Band Pass Filters (BPF)
    # Alpha 8 - 13 Hz
    # BPFAlpha_Coefficients = np.array(list(map(float, (df.Alpha.tolist())[:])))
    BPFAlpha_Coefficients = signal.firwin(FilterOrder, [8, 13], pass_zero=False, window="hamming", fs=fs)
    # Beta 13 - 30 Hz
    # BPFBeta_Coefficients = np.array(list(map(float, (df.Beta.tolist())[:])))
    BPFBeta_Coefficients = signal.firwin(FilterOrder, [13, 30], pass_zero=False, window="hamming", fs=fs)
    # Delta 0 - 4Hz
    # BPFDelta_Coefficients = np.array(list(map(float, (df.Delta.tolist())[:])))
    BPFDelta_Coefficients = signal.firwin(FilterOrder, 4, window="hamming", fs=fs)
    # Gamma 30 - 80 Hz
    # BPFGamma_Coefficients = np.array(list(map(float, (df.Gamma.tolist())[:])))
    BPFGamma_Coefficients = signal.firwin(FilterOrder, [30, 80], pass_zero=False, window="hamming", fs=fs)
    # Theta 4 - 8 Hz
    # BPFTheta_Coefficients = np.array(list(map(float, (df.Theta.tolist())[:])))
    BPFTheta_Coefficients = signal.firwin(FilterOrder, [4, 8], pass_zero=False, window="hamming", fs=fs)
    # High pass Filter (HPF) > 80Hz
    # HPF80Hz_Coefficients = np.array(list(map(float, (df.HPF.tolist())[:])))
    HPF80Hz_Coefficients = signal.firwin(FilterOrder, 80, pass_zero=False, window="hamming", fs=fs)

    print("Filtering brain wave data...")
    # Implement FIR filters
    # Step 1: notch filter to get rid of 60 Hz signal
    EEG_EEG3_Notchfiltered = filtfilt(Notch60Hz_b_Coeff, Notch60Hz_a_Coeff, EEG_EEG3)
    EEG_EEG2_Notchfiltered = filtfilt(Notch60Hz_b_Coeff, Notch60Hz_a_Coeff, EEG_EEG2)

    # Step2: use notch filtered EEG data to extract brain waves and high freq osscillations
    # Alpha
    EEG_EEG3_BPFAlpha = sig_convolve(EEG_EEG3_Notchfiltered, BPFAlpha_Coefficients, mode='same')
    EEG_EEG2_BPFAlpha = sig_convolve(EEG_EEG2_Notchfiltered, BPFAlpha_Coefficients, mode='same')
    # Beta
    EEG_EEG3_BPFBeta = sig_convolve(EEG_EEG3_Notchfiltered, BPFBeta_Coefficients, mode='same')
    EEG_EEG2_BPFBeta = sig_convolve(EEG_EEG2_Notchfiltered, BPFBeta_Coefficients, mode='same')
    # Gamma
    EEG_EEG3_BPFGamma = sig_convolve(EEG_EEG3_Notchfiltered, BPFGamma_Coefficients, mode='same')
    EEG_EEG2_BPFGamma = sig_convolve(EEG_EEG2_Notchfiltered, BPFGamma_Coefficients, mode='same')
    # Delta
    EEG_EEG3_BPFDelta = sig_convolve(EEG_EEG3_Notchfiltered, BPFDelta_Coefficients, mode='same')
    EEG_EEG2_BPFDelta = sig_convolve(EEG_EEG2_Notchfiltered, BPFDelta_Coefficients, mode='same')
    # Theta
    EEG_EEG3_BPFTheta = sig_convolve(EEG_EEG3_Notchfiltered, BPFTheta_Coefficients, mode='same')
    EEG_EEG2_BPFTheta = sig_convolve(EEG_EEG2_Notchfiltered, BPFTheta_Coefficients, mode='same')
    # High Freq
    EEG_EEG3_HPF80Hz = sig_convolve(EEG_EEG3_Notchfiltered, HPF80Hz_Coefficients, mode='same')
    EEG_EEG2_HPF80Hz = sig_convolve(EEG_EEG2_Notchfiltered, HPF80Hz_Coefficients, mode='same')

    print("Plotting brain waves...")
    # Plot original EEG and Brain frequency
    fig2, axs2 = plt.subplots(7, 2)
    fig2.suptitle('Brain Waves')
    axs2[0, 0].plot(time, EEG_EEG2)
    axs2[0, 0].set_title('EEG_EEG2')
    axs2[0, 0].set_ylabel('EEG')
    # Alpha wave from EEG_EEG2
    axs2[1, 0].plot(time, EEG_EEG2_BPFAlpha)
    axs2[1, 0].set_ylabel('Alpha')
    # Beta wave from EEG_EEG2
    axs2[2, 0].plot(time, EEG_EEG2_BPFBeta)
    axs2[2, 0].set_ylabel('Beta')
    # Gamma wave from EEG_EEG2
    axs2[3, 0].plot(time, EEG_EEG2_BPFGamma)
    axs2[3, 0].set_ylabel('Gamma')
    # Delta wave from EEG_EEG2
    axs2[4, 0].plot(time, EEG_EEG2_BPFDelta)
    axs2[4, 0].set_ylabel('Delta')
    # Theta wave from EEG_EEG2
    axs2[5, 0].plot(time, EEG_EEG2_BPFTheta)
    axs2[5, 0].set_ylabel('Theta')
    # High Freq wave from EEG_EEG2
    axs2[6, 0].plot(time, EEG_EEG2_HPF80Hz)
    axs2[6, 0].set_ylabel('High Frequency')
    axs2[6, 0].set_xlabel('Time (s)')

    axs2[0, 1].plot(time, EEG_EEG3)
    axs2[0, 1].set_title('EEG_EEG3')
    # Alpha wave from EEG_EEG3
    axs2[1, 1].plot(time, EEG_EEG3_BPFAlpha)
    # Beta wave from EEG_EEG3
    axs2[2, 1].plot(time, EEG_EEG3_BPFBeta)
    # Gamma wave from EEG_EEG3
    axs2[3, 1].plot(time, EEG_EEG3_BPFGamma)
    # Delta wave from EEG_EEG3
    axs2[4, 1].plot(time, EEG_EEG3_BPFDelta)
    # Theta wave from EEG_EEG3
    axs2[5, 1].plot(time, EEG_EEG3_BPFTheta)
    # High Freq wave from EEG_EEG3
    axs2[6, 1].plot(time, EEG_EEG3_HPF80Hz)
    axs2[6, 1].set_xlabel('Time (s)')

    #Heatmap
    #plt.specgram(EEG_EEG2, Fs=fs)
    #plt.specgram(EEG_EEG3, Fs=fs)
    #plt.xlabel('Time')
    #plt.ylabel('Frequency')
    #plt.show()

    # Moving Filtered data into csv file
    # create row_ ist to write
    row_list = [EEG_EEG2, EEG_EEG2_BPFAlpha, EEG_EEG2_BPFBeta, EEG_EEG2_BPFGamma, EEG_EEG2_BPFDelta, EEG_EEG2_BPFTheta,
                EEG_EEG2_HPF80Hz, EEG_EEG3, EEG_EEG3_BPFAlpha, EEG_EEG3_BPFBeta, EEG_EEG3_BPFGamma, EEG_EEG3_BPFDelta,
                EEG_EEG3_BPFTheta, EEG_EEG3_HPF80Hz]

    # invert row list to get column list: https://stackoverflow.com/questions/10169919/python-matrix-transpose-and-zip
    col_list = np.transpose(row_list)

    OutputFileName = inputFileName + "_FilteredBrainWaves.txt"
    print("Writing file " + OutputFileName + "...")
    FullOutputPath = FilteredDataFolderPath + OutputFileName
    # write data to CSV
    with open(FullOutputPath, 'w',
              newline='') as file:  # https://www.programiz.com/python-programming/writing-csv-files
        writer = csv.writer(file)
        writer.writerow(['EEG2', 'EEG2_Alpha', 'EEG2_Beta', 'EEG2_Gamma', 'EEG2_Delta', 'EEG2_Theta', 'EEG2_HF',
                         'EEG3', 'EEG3_Alpha', 'EEG3_Beta', 'EEG3_Gamma', 'EEG3_Delta', 'EEG3_Theta', 'EEG3_HF'])
        writer.writerows(col_list)

    print("Finished.\n")

    # format and show final plot
    fig1.tight_layout()
    fig2.tight_layout(pad=0.5, w_pad=0.01, h_pad=0.01)
    plt.show()
