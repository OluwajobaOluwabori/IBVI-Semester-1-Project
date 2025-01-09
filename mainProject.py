import os, sys, time, pdb
import cv2  # pip install opencv-python
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi  # pip install scipy
import scipy.signal as sig
import pandas as pd, math
import wfdb
from IPython.core.pylabtools import figsize
from IPython.display import display
from scipy.signal import periodogram,welch
from scipy.signal.windows import hamming, hann, boxcar
from scipy.signal import find_peaks,butter, filtfilt
# from tensorflow.keras.models import load_model
from PIL import Image
from skimage.segmentation import flood
from skimage.morphology import erosion,dilation,remove_small_objects
from scipy.ndimage import binary_fill_holes
from skimage.measure import label, regionprops, regionprops_table
from skimage.segmentation import clear_border
from scipy.ndimage import binary_fill_holes
import csv
import json





import openpyxl

from absl.logging import exception


from PyQt5.QtGui import QImage, QPixmap,QFont
from PyQt5.QtPrintSupport import QPrintDialog, QPrinter


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import QColor, QIcon
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QMessageBox, QWidget, QTableWidget, QTableWidgetItem, QHeaderView,QVBoxLayout,QHBoxLayout, QFrame
from PyQt5.uic import loadUi

from matplotlib.backends.backend_qt5agg import (NavigationToolbar2QT as NavigationToolbar)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib
matplotlib.use('TKAgg')


import numpy as np
from numpy import *
from ProjectS1v1GUI import Ui_MainWindow
# import tensorflow as tf
# from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate
from skimage.io import imread, imshow
import skimage.morphology as morph
from scipy.stats import iqr
from PyQt5.QtWidgets import QApplication, QMainWindow, QDesktopWidget
import sqlite3  # For SQLite database integration
import mysql.connector

import csv      # For CSV data loading
b_Canvas = False

class MyMainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MyMainWindow, self).__init__(parent)
        QMainWindow.__init__(self, parent)
        self.processed_image = None
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.ui.icon_name_widget.setHidden(True)

        # Add logic for expanding and collapsing the menu
        self.ui.pbMenu.toggled.connect(self.toggle_menu)
        self.ui.pbMenuIconName.toggled.connect(self.toggle_menu)

        self.ui.pbDataManagement1.clicked.connect(self.switch_to_dataManagementPage)
        self.ui.pbDataManagement2.clicked.connect(self.switch_to_dataManagementPage)
        self.ui.pbDataAnalysis1.clicked.connect(self.switch_to_DataAnalysisPage)
        self.ui.pbDataAnalysis2.clicked.connect(self.switch_to_DataAnalysisPage)
        self.ui.pbSpectrumAnalysis1.clicked.connect(self.switch_to_SpectrumAnalysisPage)
        self.ui.pbSpectrumAnalysis2.clicked.connect(self.switch_to_SpectrumAnalysisPage)
        self.ui.pbImageProcessing1.clicked.connect(self.switch_to_ImageProcessingPage)
        self.ui.pbImageProcessing2.clicked.connect(self.switch_to_ImageProcessingPage)

        # Link the clicked action from the button to a function to read excel file
        self.ui.LoadBtnsignal.clicked.connect(self.loadDatasignal)
        self.ui.updateTimeButton.clicked.connect(self.plotSignal)
        self.ui.windowDurationSlider.valueChanged.connect(self.updateSliderLabel)

        self.ui.WindowcomboBox.currentTextChanged.connect(self.plotPeriodogram)
        self.ui.ScalingcomboBox.currentTextChanged.connect(self.plotPeriodogram)
        self.ui.NormcomboBox.currentTextChanged.connect(self.plotFFT)
        # self.ui.pbValidate.clicked.connect(self.updateMainWindow)

        self.ui.btnCalculateMetrics.clicked.connect(self.calculate_metrics)
        self.ui.exportButton.clicked.connect(self.export_metrics_to_file)
        self.ui.clearButton.clicked.connect(self.clearResultsTable)

        # Add an index tracker for plots
        self.current_plot_index = 0
        self.plots = ["Signal", "FFT", "Periodogram"]
        self.ui.prevBtn.clicked.connect(self.show_previous_plot)
        self.ui.nextBtn.clicked.connect(self.show_next_plot)
        self.ui.savePlotBtn.clicked.connect(self.save_current_plot)

        # Signals and Slots
        self.ui.btnLoadImage.clicked.connect(self.load_image)
        self.ui.btnGrayscale.clicked.connect(self.convert_to_grayscale)
        self.ui.btnDenoise.clicked.connect(self.apply_denoising)
        self.ui.btnBlur.clicked.connect(self.apply_blurring)
        self.ui.btnEdgeDetection.clicked.connect(self.apply_canny_edge)
        self.ui.btnThreshold.clicked.connect(self.apply_thresholding)
        self.ui.btnSharpen.clicked.connect(self.apply_sharpening)


        self.ui.btnClearBorders.clicked.connect(self.clear_borders)
        self.ui.btnFillHoles.clicked.connect(self.fill_holes)
        self.ui.btnRemoveSmallObjects.clicked.connect(self.remove_small_objects)
        self.ui.btnErosion.clicked.connect(self.apply_erosion)
        self.ui.btnDilation.clicked.connect(self.apply_dilation)
        self.ui.btnCalculateProps.clicked.connect(self.calculate_properties)
        self.ui.btnExportResults.clicked.connect(self.export_results)

        self.ui.btnClearDisplay.clicked.connect(self.clear_display)
        self.ui.btnUndo.clicked.connect(self.undo_action)
        self.ui.btnRedo.clicked.connect(self.redo_action)
        self.ui.btnSaveImage.clicked.connect(self.saveImage)
        self.ui.btnRedo.setEnabled(False)
        self.ui.btnUndo.setEnabled(False)
        self.ui.btnClearDisplay.setEnabled(False)
        self.ui.btnSaveImage.setEnabled(False)



        # Link the clicked action from the button to a function to read excel file
        self.ui.LoadBtn_excel.clicked.connect(self.loadExcelData)
        # Link the clicked action from the button to a function to plot data
        self.ui.pbPlot.clicked.connect(self.plotExcelData)
        # Link the combobox to a function when we select an item from the list
        self.ui.cbX.currentIndexChanged.connect(self.cbXvaluechanged)
        self.ui.cbY.currentIndexChanged.connect(self.cbYvaluechanged)

        # Button can't be clicked because no reason to if Excel not yet loaded
        self.ui.pbPlot.setEnabled(False)

        # Change the default name of the tab1 and tab2
        self.ui.tabWidget.setTabText(0, "Plot")
        self.ui.tabWidget.setTabText(1, "Data")
        # Force the focus on the first tab
        self.ui.tabWidget.setCurrentIndex(0)

        # Link the clicked action from the button to a function to plot data
        # self.ui.pbPlot.clicked.connect(self.plotData)
        # self.ui.pbFFT.clicked.connect(self.showFFTWindow)
        # self.ui.pbPeriodogram.clicked.connect(self.showPeriodogramWindow)


        # Button can't be clicked because no reason to if Excel not yet loaded
        # self.ui.pbPlot.setEnabled(False)
        # self.ui.pbFFT.setEnabled(False)
        # self.ui.pbPeriodogram.setEnabled(False)

        self.canvas = None
        self.toolbar = None
        self.metrics = {}

        self.history = []  # Stack for undo
        self.redo_stack = []  # Stack for redo

        # Signals and Slots for Data Management
        self.ui.loadCSVButton.clicked.connect(self.loadCSV)
        self.ui.connectDBButton.clicked.connect(self.connectDatabase)
        self.ui.insertDataButton.clicked.connect(self.insertData)
        self.ui.retrieveDataButton.clicked.connect(self.retrieveData)
        self.ui.updateDataButton.clicked.connect(self.updateData)
        self.ui.deleteDataButton.clicked.connect(self.deleteData)
        self.ui.insertRowButton.clicked.connect(self.insertRow)


        self.centerWindow()


    def centerWindow(self):
        """Center the main window on the screen, accounting for DPI scaling."""
        screen_geometry = QDesktopWidget().availableGeometry(self)  # Get available geometry for the current screen
        window_geometry = self.frameGeometry()  # Get the actual frame geometry of the window

        # Calculate the center point of the screen
        center_point = screen_geometry.center()

        # Move the window's geometry center to the screen's center
        window_geometry.moveCenter(center_point)

        # Apply the new position to the window
        self.move(window_geometry.topLeft())

    def switch_to_dataManagementPage(self):
        self.ui.stackedWidget.setCurrentIndex(0)
    def switch_to_DataAnalysisPage(self):
        self.ui.stackedWidget.setCurrentIndex(1)
    def switch_to_SpectrumAnalysisPage(self):
        self.ui.stackedWidget.setCurrentIndex(2)
        # self.ui.LoadBtnsignal.clicked.connect(self.loadDatasignal)
        # self.ui.pbPeriodogram.clicked.connect(self.Periodogram)
    def switch_to_ImageProcessingPage(self):
        self.ui.stackedWidget.setCurrentIndex(3)


    def loadDatasignal(self):
        try:
            myDlg = QFileDialog.getOpenFileName(None, 'OpenFile', "", "Record Files (*.dat)")

            self.myPath = myDlg[0][:-4] # Path + file + extension
            print(self.myPath)
            FileNameWithExtension = QFileInfo(myDlg[0]).fileName()  # Just the file + extension
            print(self.myPath)

            if myDlg[0] == myDlg[1] == '':
                # No file selected or cancel button clicked - so do nothing
                pass
            else:
                # Read and extract values from Excel
                self.data = wfdb.rdrecord(self.myPath)
                self.signal = self.data.p_signal[:, 0]
                self.signal = self.signal - mean(self.signal)
                self.data.fs = self.data.fs
                self.data.n_sig = self.data.n_sig
                duration = len(self.signal) / self.data.fs  # Duration in seconds
                self.time = np.linspace(0, duration, len(self.signal))


                mean_signal = np.mean(self.signal)
                """Applies a Butterworth bandpass filter."""
                nyquist = 0.5 * self.data.fs
                low = 0.5 / nyquist
                high = 40 / nyquist
                b, a = butter(4, [low, high], btype='band')
                self.signal = filtfilt(b, a, self.signal)

                # Detect R-peaks with a height threshold
                threshold = 0.5 * np.max(self.signal) #np.mean(self.signal) + 0.5 * np.std(self.signal)
                self.peaks, _ = find_peaks(self.signal, height=threshold)
                print(f"Potential heartbeat matches: {self.peaks}")

                print(f'The average signal value is : {mean_signal:.2f}')

                # Plot the loaded data and initial FFT and periodogram
                self.plotSignal()
                # Show success message box
                QMessageBox.information(self, "Success", "Data successfully loaded!")
        except Exception as e:
            print(e)
            QMessageBox.critical(self, "Error", f"Failed to load data: {e}")



    def show_previous_plot(self):
        """Navigate to the previous plot."""
        self.current_plot_index = (self.current_plot_index - 1) % len(self.plots)
        self.update_plot_display()

    def show_next_plot(self):
        """Navigate to the next plot."""
        self.current_plot_index = (self.current_plot_index + 1) % len(self.plots)
        self.update_plot_display()

    def update_plot_display(self):
        """Update the plot based on current index."""
        plot_name = self.plots[self.current_plot_index]
        print(f"Displaying: {plot_name}")
        if plot_name == "Signal":
            self.plotSignal()
        elif plot_name == "FFT":
            self.plotFFT()
        elif plot_name == "Periodogram":
            self.plotPeriodogram()

    def save_current_plot(self):
        """Save the currently displayed plot."""
        plot_name = self.plots[self.current_plot_index]
        file_name, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Plot", f"{plot_name}.png", "PNG Files (*.png);;All Files (*)"
        )
        if file_name:
            print(f"Saving {plot_name} plot to {file_name}")
            # Add logic to save the current plot using Matplotlib
            self.canvas.figure.savefig(file_name)
            QMessageBox.information(self, "Saved", f"Plot saved as {file_name}")

            # Example: self.figure.savefig(file_name)

    def plotSignal(self):
        """Plots the signal and peaks."""
        try:
            layout = self.ui.mplwindow.layout()
            self.clearCanvas(layout)

            # Parse start time and window duration
            start_time, window_duration = self.getTimeAndWindow()
            start_index = int(start_time * self.data.fs)
            end_index = start_index + int(window_duration * self.data.fs)

            # Boundary check
            if start_index < 0 or end_index > len(self.signal):
                QMessageBox.warning(self, "Invalid Range", "The specified range is out of bounds.")
                return

            # Extract signal segment and peaks
            segment = self.signal[start_index:end_index]
            segment_time = self.time[start_index:end_index]
            peaks_indices = [i for i in self.peaks if start_index <= i < end_index]
            peaks_indices = np.array(peaks_indices) - start_index

            # Plot Signal
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(segment_time, segment, label="Signal")
            ax.plot(segment_time[peaks_indices], segment[peaks_indices], "x", label='Peaks')
            ax.set_title(f"Signal Plot(Duration: {window_duration}s, Start: {start_time}s, Stop: {start_time + window_duration}s)")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude")
            ax.legend()

            self.displayCanvas(fig, layout)
        except Exception as e:
            print(e)

    def plotFFT(self):
        """Plots the FFT of the signal."""
        try:
            layout = self.ui.mplwindow.layout()
            self.clearCanvas(layout)

            # Parse time and window duration
            start_time, window_duration = self.getTimeAndWindow()
            start_index = int(start_time * self.data.fs)
            end_index = start_index + int(window_duration * self.data.fs)
            segment = self.signal[start_index:end_index]

            # Calculate FFT
            norm_value = self.ui.NormcomboBox.currentText()
            fft_values = np.fft.fft(segment, norm=norm_value)
            n = len(segment)
            freqs = np.fft.fftfreq(n, d=1 / self.data.fs)
            positive_freqs = freqs[:n // 2]
            fft_magnitude = np.abs(fft_values[:n // 2])

            # Plot FFT
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(positive_freqs, fft_magnitude)
            ax.set_title("FFT Plot")
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("Amplitude")

            self.displayCanvas(fig, layout)
        except Exception as e:
            print(e)

    def plotPeriodogram(self):
        """Plots the Periodogram of the signal."""
        try:
            layout = self.ui.mplwindow.layout()
            self.clearCanvas(layout)

            # Parse time and window duration
            start_time, window_duration = self.getTimeAndWindow()
            start_index = int(start_time * self.data.fs)
            end_index = start_index + int(window_duration * self.data.fs)
            segment = self.signal[start_index:end_index]

            # Periodogram Parameters
            scaling_value = self.ui.ScalingcomboBox.currentText()
            window_value = self.ui.WindowcomboBox.currentText()

            # Calculate Periodogram
            freqs, Pxx = periodogram(segment, fs=self.data.fs, scaling=scaling_value, window=window_value)
            Pxx_dB = 10 * np.log10(Pxx)

            # Plot Periodogram
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(freqs, Pxx_dB)
            ax.set_title("Periodogram Plot")
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("Power (dB/Hz)")

            self.displayCanvas(fig, layout)
        except Exception as e:
            print(e)

    def clearCanvas(self, layout):
        """Clears the current canvas and toolbar."""
        if self.canvas:
            layout.removeWidget(self.canvas)
            self.canvas.deleteLater()
            self.canvas = None
        if self.toolbar:
            layout.removeWidget(self.toolbar)
            self.toolbar.deleteLater()
            self.toolbar = None

    def displayCanvas(self, fig, layout):
        """Displays the Matplotlib figure in the layout."""
        self.canvas = FigureCanvas(fig)
        layout.addWidget(self.canvas)
        self.canvas.draw()
        # self.toolbar = NavigationToolbar(self.canvas, self.ui.mplwindow, coordinates=True)
        # layout.insertWidget(0, self.toolbar)

    def getTimeAndWindow(self):
        """Parses start time and window duration."""
        try:
            start_time_str = self.ui.timeInput.text()
            h, m, s = map(int, start_time_str.split(":")) if start_time_str else (0, 0, 0)
            start_time = h * 3600 + m * 60 + s
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Please enter time in hh:mm:ss format.")
            start_time = 0
        window_duration = self.ui.windowDurationSlider.value()
        return start_time, window_duration

    def calculate_metrics(self):
        try:
            """Calculate Basic ECG Metrics and HRV"""
            # Heart Rate (BPM)
            rr_intervals = np.diff(self.peaks) / self.data.fs  # RR intervals in seconds
            heart_rates = 60 / rr_intervals  # Convert to BPM

            self.metrics['Average Heart Rate (BPM)'] = np.mean(heart_rates)
            self.metrics['Min Heart Rate (BPM)'] = np.min(heart_rates)
            self.metrics['Max Heart Rate (BPM)'] = np.max(heart_rates)

            # Time-Domain HRV Metrics
            self.metrics['Mean RR Interval (ms)'] = np.mean(rr_intervals) * 1000
            self.metrics['Standard Deviation of NN Intervals- SDNN (ms)'] = np.std(rr_intervals) * 1000
            self.metrics['Root Mean Square of Successive Differences- RMSSD (ms)'] = np.sqrt(np.mean(np.diff(rr_intervals) ** 2)) * 1000
            self.metrics['IQR RR Interval (ms)'] = iqr(rr_intervals) * 1000

            self.updateResultsTable(self.metrics)
            # Frequency-Domain HRV Metrics (optional)
            # You can add power spectral density analysis here for LF and HF components
        except Exception as e:
            print(e)


    def updateResultsTable(self, metrics):
        self.ui.SignalresultsTable.setRowCount(len(metrics))
        for row, (metric, value) in enumerate(metrics.items()):
            self.ui.SignalresultsTable.setItem(row, 0, QTableWidgetItem(metric))
            self.ui.SignalresultsTable.setItem(row, 1, QTableWidgetItem(f"{value:.2f}"))

    def export_metrics_to_file(self):
        """Export Metrics to Excel or CSV"""
        try:
            file_path, selected_filter = QtWidgets.QFileDialog.getSaveFileName(
                None,
                "Export Results",
                "",
                "CSV Files (*.csv);;Excel Files (*.xlsx);;JSON Files (*.json)"
            )
            if not file_path:
                return  # User canceled the dialog
            # Proceed only if a file path is selected
            if file_path:
                df = pd.DataFrame(self.metrics.items(), columns=['Metric', 'Value'])
                if file_path.endswith('.csv'):
                    df.to_csv(file_path, index=False)
                elif file_path.endswith('.xlsx'):
                    df.to_excel(file_path, index=False)
                else:
                    raise ValueError("Unsupported file format. Use .csv or .xlsx.")
        except Exception as e:
            print(e)

    def clearResultsTable(self):
        self.ui.SignalresultsTable.setRowCount(0)

    # def plotSignal(self):
    #         # Clear existing canvas and toolbar if they exist
    #         try:
    #             layout = self.ui.mplwindow.layout()
    #             if self.canvas:
    #                 layout.removeWidget(self.canvas)
    #                 self.canvas.deleteLater()
    #                 self.canvas = None
    #             if self.toolbar:
    #                 layout.removeWidget(self.toolbar)
    #                 self.toolbar.deleteLater()
    #                 self.toolbar = None
    #
    #             # Determine start time and window duration
    #             try:
    #                 # Parse start time from QLineEdit (hh:mm:ss)
    #                 start_time_str = self.ui.timeInput.text()
    #                 if start_time_str:
    #                     h, m, s = map(int, start_time_str.split(':'))
    #                     start_time = h * 3600 + m * 60 + s  # Convert to seconds
    #                 else:
    #                     start_time = 0  # Default to 0 if no input is provided
    #             except ValueError:
    #                 QMessageBox.warning(self, "Invalid Input", "Please enter time in hh:mm:ss format.")
    #                 return
    #
    #             # Get the window duration from QSlider (default: 5 seconds)
    #             window_duration = self.ui.windowDurationSlider.value()
    #
    #             # Calculate start and end indices
    #             start_index = int(start_time * self.data.fs)  # Sampling rate to index
    #             end_index = start_index + int(window_duration * self.data.fs)
    #
    #             # Boundary checks
    #             if start_index < 0 or end_index > len(self.signal):
    #                 QMessageBox.warning(self, "Invalid Range", "The specified range is out of bounds.")
    #                 return
    #
    #             # Extract the segment of the signal
    #             segment = self.signal[start_index:end_index]
    #             segment_time = self.time[start_index:end_index]
    #
    #             peaks_indices = [i for i in self.peaks if start_index <= i < end_index]
    #             peaks_indices = np.array(peaks_indices) - start_index  # Adjust indices to the segment
    #
    #             # Format time to hh:mm:ss for x-axis
    #             formatted_time = [time.strftime("%H:%M:%S", time.gmtime(t)) for t in segment_time]
    #
    #             # Create subplots
    #             fig, ax = plt.subplots(1, 3, figsize=(10, 5))  # Create one figure with 3 subplots
    #
    #             # Plot Signal
    #             ax[0].plot(segment_time, segment)
    #             ax[0].plot(segment_time[peaks_indices], segment[peaks_indices], "x", label='Peaks')
    #             ax[0].set_title(f"Signal (Duration: {window_duration}s, Start: {start_time}s, Stop: {start_time + window_duration}s)")
    #             ax[0].set_xlabel('Time (s)')
    #             ax[0].set_ylabel('Amplitude')
    #             ax[0].tick_params(axis='x', rotation=45)  # Rotate x-axis labels for readability
    #             # ax[0].xaxis.set_major_locator(plt.MaxNLocator(10))  # Adjust number of x-ticks
    #             ax[0].xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    #             ax[0].tick_params(axis='x', rotation=45)  # Rotate for readability
    #             fig.autofmt_xdate()  # Auto-format the x-axis dates or time
    #
    #             # fig, ax = plt.subplots(1, 3, figsize=(10, 5))  # Create one figure with 3 subplots
    #             # ax[0].plot(self.time, self.signal)
    #             # ax[0].set_title("Loaded Signal Data")
    #             # ax[0].set_xlabel('Time [s]')
    #             # ax[0].set_ylabel("Amplitude")
    #             # for tick in ax.xaxis.get_major_ticks():
    #             #     tick.label.set_fontsize(18)  # Adjust the fontsize for x ticks
    #             #
    #             # for tick in ax.yaxis.get_major_ticks():
    #             #     tick.label.set_fontsize(18)  # Adjust the fontsize for y ticks
    #
    #             # FFT
    #             # Get the selected window and scaling values
    #             norm_value = self.ui.NormcomboBox.currentText()
    #             # Step 1: Calculate the FFT
    #             fft_values = np.fft.fft(segment, norm=norm_value)
    #             n = len(segment)
    #             # fft_values = np.fft.fft(self.signal, norm=norm_value)
    #             # # Step 2: Calculate the frequencies corresponding to the FFT values
    #             # n = len(self.signal)
    #             frequencies = np.fft.fftfreq(n, d=1 / self.data.fs)
    #
    #             # Step 3: Get the positive frequencies and the corresponding FFT values
    #             positive_frequencies = frequencies[:n // 2]
    #             positive_fft_values = np.abs(fft_values[:n // 2])  # Magnitude of the FFT
    #
    #             # Step 4: Plot the FFT
    #             ax[1].clear()
    #             ax[1].plot(positive_frequencies, positive_fft_values)
    #             ax[1].set_title(f'Frequency Spectrum of the Signal\n Norm = {norm_value}')
    #             ax[1].set_xlabel('Frequency (Hz)')
    #             ax[1].set_ylabel('Amplitude')
    #             # Periodogram
    #             scaling_value = self.ui.ScalingcomboBox.currentText()
    #             window_value = self.ui.WindowcomboBox.currentText()
    #
    #             # f, Pxx = periodogram(self.signal, fs=self.data.fs, scaling=scaling_value,
    #             #                      window=window_value)
    #             f, Pxx = periodogram(segment, fs=self.data.fs, scaling=scaling_value, window=window_value)
    #
    #             Pxx_dB = 10 * np.log10(Pxx)  # Convert power to dB
    #
    #
    #             ax[2].clear()
    #             ax[2].plot(f, Pxx_dB)
    #             ax[2].set_title(f'Periodogram of the Signal\n Window Type = {window_value}, Scaling = {scaling_value}'
    #                       )
    #             ax[2].set_xlabel('Frequency (Hz)')
    #             if scaling_value == 'spectrum':
    #                 ax[2].set_ylabel('Power(dB/Hz)')
    #             else:
    #                 ax[2].set_ylabel('Power/Frequency (dB/Hz)')
    #             # for tick in ax.xaxis.get_major_ticks():
    #             #     tick.label.set_fontsize(18)  # Adjust the fontsize for x ticks
    #             #
    #             # for tick in ax.yaxis.get_major_ticks():
    #             #     tick.label.set_fontsize(18)  # Adjust the fontsize for y ticks
    #
    #
    #
    #             fig.tight_layout()
    #             self.canvas = FigureCanvas(fig)
    #             layout = self.ui.mplwindow.layout()
    #             layout.addWidget(self.canvas)
    #             self.canvas.draw()
    #             self.toolbar = NavigationToolbar(self.canvas, self.ui.mplwindow, coordinates=True)
    #             layout.insertWidget(0, self.toolbar)  # Insert at the top of the layout
    #
    #         except Exception as e:
    #          print(e)

    def updateSliderLabel(self):
        try:
            step = 5  # Define the step size
            value = self.ui.windowDurationSlider.value()
            rounded_value = round(value / step) * step  # Round to the nearest step
            if value != rounded_value:  # Avoid infinite loops
                self.ui.windowDurationSlider.setValue(rounded_value)
            self.ui.windowDurationLabel.setText(f"Window Duration: {rounded_value} seconds")
        except Exception as e:
            print(e)

    def load_image(self):
        try:
            """Load an image and display it in the Original Image label."""
            options = QFileDialog.Options()
            file_path, _ = QFileDialog.getOpenFileName(self, "Open Image File", "",
                                                       "Images (*.png *.jpg *.bmp)", options=options)
            if file_path:
                self.original_image = cv2.imread(file_path)
                self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)

                self.processed_image = self.original_image.copy()
                self.display_image(self.original_image, self.ui.labelOriginalImage)
                self.ui.btnClearDisplay.setEnabled(True)
                self.ui.btnSaveImage.setEnabled(True)


                # Show success message box
            QMessageBox.information(self, "Success", "Image successfully loaded!")
        except Exception as e:
            print(e)

    def display_image(self, image, label):
        """Utility to display an image in a QLabel without stretching."""
        if len(image.shape) == 3:
            # Color images
            qformat = QImage.Format_RGB888
            h, w, ch = image.shape
            bytes_per_line = ch * w
            q_image = QtGui.QImage(image.data, w, h, bytes_per_line, qformat)
        else:
            # Grayscale images
            qformat = QImage.Format_Indexed8
            h, w = image.shape
            bytes_per_line = w
            q_image = QtGui.QImage(image.data, w, h,  image.strides[0], qformat)

            # Scale the pixmap while preserving aspect ratio
        scaled_pixmap = q_image.scaled(
            label.width(),
            label.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )

        pixmap = QPixmap.fromImage(scaled_pixmap)



        # Set the pixmap to the QLabel
        label.setPixmap(pixmap)

    def convert_to_grayscale(self):
        """Convert the image to grayscale."""
        try:
            if len(self.processed_image.shape) == 3:
                self.history.append(self.processed_image.copy())
                # Convert the processed image to grayscale and then to a binary image
                self.processed_image = cv2.cvtColor(self.processed_image, cv2.COLOR_RGB2GRAY)
                self.display_image(self.processed_image, self.ui.labelProcessedImage)
                self.ui.btnUndo.setEnabled(True)
                self.ui.btnRedo.setEnabled(False)
            else:
                # self.display_message("Image is already grayscale.")
                 print("Image can not be convertef to grayscale.")

        except Exception as e:
            print(e)

    def apply_thresholding(self):
        """Apply thresholding based on slider value."""
        # self.ui.finalizeThresholdCheckbox.setEnabled(True)
        try:
            if hasattr(self, 'processed_image'):
                self.history.append(self.processed_image.copy())
                gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_RGB2GRAY)
                _, thresholded = cv2.threshold(gray_image,127, 255, cv2.THRESH_OTSU)
                self.processed_image = cv2.cvtColor(thresholded, cv2.COLOR_GRAY2RGB)
                # self.ui.thresholdValueLabel.setText(f"Threshold Level: {self.ui.sliderThreshold.value()} (Min: 0, Max: 255)")
                self.display_image(self.processed_image, self.ui.labelProcessedImage)
                self.ui.btnUndo.setEnabled(True)
                self.ui.btnRedo.setEnabled(False)
        except Exception as e:
            print(e)


    def apply_blurring(self):
        """Apply a selected blur filter to the image."""
        self.history.append(self.processed_image.copy())
        if hasattr(self, 'original_image'):
            self.processed_image = cv2.GaussianBlur(self.processed_image, (5, 5), 0)
        else :
            self.processed_image = cv2.GaussianBlur(self.original_image, (5, 5), 0)
        self.display_image(self.processed_image, self.ui.labelProcessedImage)
        self.ui.btnUndo.setEnabled(True)
        self.ui.btnRedo.setEnabled(False)

    def apply_denoising(self):
        try:
            self.history.append(self.processed_image.copy())
            if len(self.processed_image.shape) == 3:  # Convert to grayscale if RGB
                gray_image = cv2.cvtColor(self.processed_image, cv2.COLOR_RGB2GRAY)
            else:
                gray_image = self.processed_image
            self.processed_image = cv2.fastNlMeansDenoising(gray_image, None, 10, 7, 21)
            self.display_image(self.processed_image, self.ui.labelProcessedImage)
            self.ui.btnUndo.setEnabled(True)
            self.ui.btnRedo.setEnabled(False)

        except Exception as e:
            print(f"Error in denoising: {e}")


    def apply_canny_edge(self):
        try:
            self.history.append(self.processed_image.copy())
            if len(self.processed_image.shape) == 3:  # Convert to grayscale if RGB
                gray_image = cv2.cvtColor(self.processed_image, cv2.COLOR_RGB2GRAY)
            else:
                gray_image = self.processed_image
            self.processed_image = cv2.Canny(gray_image, 70, 135)
            self.display_image(self.processed_image, self.ui.labelProcessedImage)
            self.ui.btnUndo.setEnabled(True)
            self.ui.btnRedo.setEnabled(False)

        except Exception as e:
            print(f"Error in edge detection: {e}")

    # Morphological Operations

    def clear_borders(self):
        """Clear borders of objects in the image."""
        try:
            self.history.append(self.processed_image.copy())
            if len(self.processed_image.shape) == 3:
                # Convert the processed image to grayscale and then to a binary image
                self.processed_image = cv2.cvtColor(self.processed_image, cv2.COLOR_RGB2GRAY)
            _, self.processed_image = cv2.threshold(self.processed_image, 127, 255, cv2.THRESH_BINARY)
            labelled = label(self.processed_image)  # label all the constituents
            cleared = clear_border(labelled)
            self.processed_image = (cleared > 0).astype(np.uint8) * 255  # Convert back to binary
            self.display_image(self.processed_image, self.ui.labelProcessedImage)
            self.ui.btnUndo.setEnabled(True)
            self.ui.btnRedo.setEnabled(False)

        except Exception as e:
            print(e)

    def fill_holes(self):
        """Fill holes in the objects of the image."""
        try:
            self.history.append(self.processed_image.copy())
            # Convert to grayscale if necessary
            if len(self.processed_image.shape) == 3:  # RGB
                gray_image = cv2.cvtColor(self.processed_image, cv2.COLOR_RGB2GRAY)
            else:
                gray_image = self.processed_image
            _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
            self.processed_image = binary_fill_holes(binary_image).astype(np.uint8) * 255
            self.display_image(self.processed_image, self.ui.labelProcessedImage)
            self.ui.btnUndo.setEnabled(True)
            self.ui.btnRedo.setEnabled(False)

        except Exception as e:
            print(e)

    def remove_small_objects(self):
        try:
            self.history.append(self.processed_image.copy())
            if len(self.processed_image.shape) == 3:  # Convert to grayscale if RGB
                gray_image = cv2.cvtColor(self.processed_image, cv2.COLOR_RGB2GRAY)
            else:
                gray_image = self.processed_image
            _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
            labelled = label(binary_image)
            cleaned = remove_small_objects(labelled, min_size=500,connectivity=8)
            self.processed_image = (cleaned > 0).astype(np.uint8) * 255  # Convert back to binary
            self.display_image(self.processed_image, self.ui.labelProcessedImage)
            self.ui.btnUndo.setEnabled(True)
            self.ui.btnRedo.setEnabled(False)

        except Exception as e:
            print(f"Error in removing small objects: {e}")

    def apply_erosion(self):
        self.history.append(self.processed_image.copy())
        self.processed_image = erosion(self.processed_image)
        self.display_image(self.processed_image, self.ui.labelProcessedImage)
        self.ui.btnUndo.setEnabled(True)
        self.ui.btnRedo.setEnabled(False)

    def apply_dilation(self):
        self.history.append(self.processed_image.copy())
        self.processed_image = dilation(self.processed_image)
        self.display_image(self.processed_image, self.ui.labelProcessedImage)
        self.ui.btnUndo.setEnabled(True)
        self.ui.btnRedo.setEnabled(False)

    def apply_sharpening(self):
        """Apply image sharpening."""
        if hasattr(self, 'processed_image') and self.processed_image is not None:
            self.history.append(self.processed_image.copy())
            try:
                # Define a sharpening kernel
                sharpening_kernel = np.array([[0, -1, 0],
                                              [-1, 5, -1],
                                              [0, -1, 0]])
                # Apply the kernel to the processed image
                sharpened_image = cv2.filter2D(self.processed_image, -1, sharpening_kernel)

                # Update the processed image and display it
                self.processed_image = sharpened_image
                self.display_image(self.processed_image, self.ui.labelProcessedImage)

                # Save the state for undo/redo
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Could not apply sharpening. Error: {e}")
        else:
            QMessageBox.warning(self, "Warning", "No image loaded to apply sharpening!")

    def calculate_properties(self):
        """Calculate region properties based on user selection."""
        if len(self.processed_image.shape) == 3:
                # Convert the processed image to grayscale and then to a binary image
            self.processed_image = cv2.cvtColor(self.processed_image, cv2.COLOR_RGB2GRAY)
        _, binary_image = cv2.threshold(self.processed_image, 127, 255, cv2.THRESH_OTSU)

        # Label connected regions
        labeled_image = label(binary_image)
        props = regionprops(labeled_image)

        # Determine selected properties
        selected_property = self.ui.comboRegionProps.currentText()
        selected_properties = []
        if selected_property == "All":
            selected_properties = ["area", "perimeter", "eccentricity"]
        else:
            selected_properties.append(
                "area" if selected_property == "Surface Area" else selected_property.lower()
            )
        # Extract properties
        data = []
        for i, region in enumerate(props, 1):
            row = {"Object ID": i}
            for prop in selected_properties:
                if hasattr(region, prop):
                    row[prop] = getattr(region, prop)
            data.append(row)

            # Populate the results table
        self.populate_table(data, selected_properties)

    def populate_table(self, data, selected_properties):
        """Populate the result table with calculated properties."""
        try:
            # Prepare the table headers
            headers = ["Object ID"] + [prop.capitalize() for prop in selected_properties]
            self.ui.resultTable.setColumnCount(len(headers))
            self.ui.resultTable.setHorizontalHeaderLabels(headers)

            # Populate table rows
            self.ui.resultTable.setRowCount(len(data))
            for row_idx, row_data in enumerate(data):
                self.ui.resultTable.setItem(row_idx, 0, QtWidgets.QTableWidgetItem(str(row_data["Object ID"])))
                for col_idx, prop in enumerate(selected_properties, start=1):
                    value = row_data.get(prop, "N/A")
                    self.ui.resultTable.setItem(row_idx, col_idx, QtWidgets.QTableWidgetItem(
                        f"{value:.2f}" if isinstance(value, float) else str(value)))
        except Exception as e:
            print(f"Error populating table: {e}")

    def export_results(self):
        """Exports the results in the table to CSV, Excel, or JSON format."""
        # Open a file dialog to choose export location and file type
        file_path, selected_filter = QtWidgets.QFileDialog.getSaveFileName(
            None,
            "Export Results",
            "",
            "CSV Files (*.csv);;Excel Files (*.xlsx);;JSON Files (*.json)"
        )
        if not file_path:
            return  # User canceled the dialog
        # Proceed only if a file path is selected
        if file_path:
            try:
                if selected_filter == "CSV Files (*.csv)" or file_path.endswith('.csv'):
                    with open(file_path, mode='w', newline='', encoding='utf-8') as file:
                        writer = csv.writer(file)
                        # Write headers
                        header = [self.ui.resultTable.horizontalHeaderItem(i).text() for i in
                                  range(self.ui.resultTable.columnCount())]
                        writer.writerow(header)
                        # Write table data
                        for row in range(self.ui.resultTable.rowCount()):
                            row_data = [
                                self.ui.resultTable.item(row, col).text() if self.ui.resultTable.item(row, col) else ''
                                for col in range(self.ui.resultTable.columnCount())
                            ]
                            writer.writerow(row_data)
                    QtWidgets.QMessageBox.information(None, "Export Successful", "Results exported successfully to CSV!")
                elif selected_filter == "Excel Files (*.xlsx)" or file_path.endswith('.xlsx'):
                    data = []
                    for row in range(self.ui.resultTable.rowCount()):
                        row_data = [
                            self.ui.resultTable.item(row, col).text() if self.ui.resultTable.item(row, col) else ''
                            for col in range(self.ui.resultTable.columnCount())
                        ]
                        data.append(row_data)
                    # Convert to DataFrame and save
                    header = [self.ui.resultTable.horizontalHeaderItem(i).text() for i in range(self.ui.resultTable.columnCount())]
                    df = pd.DataFrame(data, columns=header)
                    df.to_excel(file_path, index=False)
                    QtWidgets.QMessageBox.information(None, "Export Successful", "Results exported successfully to Excel!")
                elif selected_filter == "JSON Files (*.json)" or file_path.endswith('.json'):
                    # Extract data from table
                    data = []
                    for row in range(self.ui.resultTable.rowCount()):
                        row_data = {
                            self.ui.resultTable.horizontalHeaderItem(col).text():
                                self.resultTable.item(row, col).text() if self.ui.resultTable.item(row, col) else ''
                            for col in range(self.ui.resultTable.columnCount())
                        }
                        data.append(row_data)
                    # Write JSON
                    with open(file_path, mode='w', encoding='utf-8') as file:
                        json.dump(data, file, indent=4)
                    QtWidgets.QMessageBox.information(None, "Export Successful", "Results exported successfully to JSON!")

            except Exception as e:
                QtWidgets.QMessageBox.critical(None, "Export Failed", f"An error occurred while exporting to {selected_filter}:\n{str(e)}")

    def clear_display(self):
        # Clear image display
        self.ui.labelOriginalImage.clear()
        self.ui.labelOriginalImage.setText("Original Image")
        self.ui.labelProcessedImage.clear()
        self.ui.labelProcessedImage.setText("Processed Image")
        self.ui.btnRedo.setEnabled(False)
        self.ui.btnUndo.setEnabled(False)
        self.ui.btnSaveImage.setEnabled(False)
        # Clear results table
        self.ui.resultTable.setRowCount(0)

        # Reset internal states (if any)
        self.original_image = None
        self.processed_image = None


    def save_state(self):
        """Save the current state for undo."""
        try:
            if self.processed_image is not None:
                self.history.append(self.processed_image)
                # self.redo_stack.clear()  # Clear redo stack after a new operation
                # Enable the Undo button after an image change
                self.ui.btnUndo.setEnabled(True)
                # After updating, disable the Redo button since a new change is made
                # self.ui.btnRedo.setEnabled(False)
                # self.close()
        except Exception as e:
            print(e)

    def undo_action(self):
        """Undo the last operation."""
        try:
            if len(self.history) > 0:
                self.redo_stack.append(self.processed_image.copy())  # Save current state for redo
                self.processed_image = self.history.pop()
                self.display_image(self.processed_image, self.ui.labelProcessedImage)
                # self.update_display()
                # Enable the Redo button now that an action can be redone
                self.ui.btnRedo.setEnabled(True)

                # Disable Undo button if there are no more actions to undo
            if len(self.history) == 0:
                self.ui.btnUndo.setEnabled(False)
        except Exception as e:
            print(e)

    def redo_action(self):
        """Redo the last undone operation."""
        if len(self.redo_stack) > 0:
            self.history.append(self.processed_image.copy())  # Save current state for undo
            self.processed_image = self.redo_stack.pop()
            # self.update_display()
            self.display_image(self.processed_image, self.ui.labelProcessedImage)
            # Disable Redo button if there are no more actions to redo
        if len(self.redo_stack) == 0:
            self.ui.btnRedo.setEnabled(False)

    # def saveImage(self):
    #     filename, ext = QFileDialog.getSaveFileName(self, 'Save File', '', 'Images Files (*.png)')
    #     if filename:
    #         cv2.imwrite(filename, self.out_image)

    def saveImage(self):
        """Save the processed image."""
        if hasattr(self, 'processed_image') and self.processed_image is not None:
            try:
                # Open file dialog to save the image
                options = QFileDialog.Options()
                file_path, _ = QFileDialog.getSaveFileName(
                    self,
                    "Save Processed Image",
                    "",
                    "Images (*.png *.jpg *.bmp);;All Files (*)",
                    options=options
                )
                if file_path:
                    # Save the image in the selected file path
                    cv2.imwrite(file_path, cv2.cvtColor(self.processed_image, cv2.COLOR_RGB2BGR))
                    QMessageBox.information(self, "Success", "Image successfully saved!")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Could not save the image. Error: {e}")
        else:
            QMessageBox.warning(self, "Warning", "No processed image to save!")

    def update_display(self):
        """Update the processed image display."""
        if self.processed_image is not None:
            self.history.append(self.processed_image.copy())
            self.display_image(self.processed_image, self.ui.labelProcessedImage)
            # # Enable the Undo button after an image change
            # self.ui.btnUndo.setEnabled(True)
            # # After updating, disable the Redo button since a new change is made
            # self.ui.btnRedo.setEnabled(False)

    # Thresholding function
    # def apply_threshold(self, image, threshold):
    #     if self.ui.finalizeThresholdCheckbox.isChecked():
    #         self.history.append(self.processed_image.copy())
    #     # Convert to grayscale if necessary
    #     if len(image.shape) == 3:  # RGB
    #         gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    #     else:
    #         gray_image = image
    #
    #     # Apply thresholding
    #     _, thresholded_image = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #     return thresholded_image



    # Function to convert NumPy array to QPixmap
    def numpy_to_qpixmap(self,array):
        # Scale the array to 0-255 if necessary (e.g., if it's 0-1 or class IDs)
        if np.max(array) <= 1:
            array = (array * 255).astype(np.uint8)

        # Convert to RGB if it's single-channel
        if len(array.shape) == 2:  # Grayscale
            array = np.stack((array,) * 3, axis=-1)

        # Convert to QImage
        h, w, ch = array.shape
        qimage = QImage(array.data, w, h, ch * w, QImage.Format_RGB888)

        # Convert to QPixmap
        return QPixmap.fromImage(qimage)

    def update_threshold(self, value):
        try:
            # Apply thresholding and update display
            thresholded = self.apply_threshold(self.contrast_enhanced, value)
            pixmap = self.numpy_to_qpixmap(thresholded)
            self.ui.labelSegmentedImage.setPixmap(pixmap.scaled(self.ui.labelSegmentedImage.size(), QtCore.Qt.KeepAspectRatio))
            # self.ui.labelOriginalImage.setPixmap(self.pixmap.scaled(self.ui.labelOriginalImage.size(), QtCore.Qt.KeepAspectRatio))

        except Exception as e:
            print(e)

    def loadCSV(self):
        try:

            # Open file dialog to select CSV
            filePath, _ = QFileDialog.getOpenFileName(None, 'OpenFile', "", "Excel Files (*.xls *.xlsx *.csv)")
            if not filePath:
                return  # User canceled the file dialog
            try:
                self.data1 = pd.read_csv(filePath)
            except:
                self.data1 = pd.read_excel(filePath)
            QMessageBox.information(None, "Success", "CSV data loaded successfully!")
            self.ui.dataTable.setColumnCount(len(self.data1.columns))
            self.ui.dataTable.setHorizontalHeaderLabels(self.data1.columns.tolist())

            self.ui.dataTable.setRowCount(len(self.data1))  # Clear existing data
            for rowIdx, row in self.data1.iterrows():
                for colIdx, value in enumerate(row):
                    self.ui.dataTable.setItem(rowIdx, colIdx, QTableWidgetItem(str(value)))

        except Exception as e:
            QMessageBox.critical(None, "Error", f"Failed to load CSV: {e}")

    def connectDatabase(self):
        try:
            # Establish connection to SQLite database
            self.dbConnection = mysql.connector.connect(host="localhost",user="root", password="", database="patient_data")
            QMessageBox.information(None, "Success", "Connected to database successfully!")
        except Exception as e:
            QMessageBox.critical(None, "Error", f"Database connection failed: {e}")

    def insertData(self):
        try:
            if not hasattr(self, 'dbConnection'):
                QMessageBox.warning(None, "Warning", "No database connection. Connect to a database first.")
                return

            cursor = self.dbConnection.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS PatientData (
            PatientID VARCHAR(50) PRIMARY KEY,
            Age INT,
            Sex VARCHAR(10),
            Cholesterol FLOAT,
            BloodPressure FLOAT,
            HeartRate INT,
            Diabetes VARCHAR(10),
            FamilyHistory VARCHAR(10),
            Smoking VARCHAR(10),
            Obesity VARCHAR(10),
            AlcoholConsumption VARCHAR(10),
            ExerciseHoursPerWeek FLOAT,
            Diet VARCHAR(50),
            PreviousHeartProblems VARCHAR(10),
            MedicationUse VARCHAR(50),
            StressLevel FLOAT,
            SedentaryHoursPerDay FLOAT,
            Income FLOAT,
            BMI FLOAT,
            Triglycerides FLOAT,
            PhysicalActivityDaysPerWeek INT,
            SleepHoursPerDay FLOAT,
            Country VARCHAR(50),
            Continent VARCHAR(50),
            Hemisphere VARCHAR(50),
            HeartAttackRisk FLOAT
        )
        """
                )

            # Insert rows from the table into the database
            for rowIdx in range(self.ui.dataTable.rowCount()):
                rowData = [self.ui.dataTable.item(rowIdx, colIdx).text() for colIdx in range(self.ui.dataTable.columnCount())]
                cursor.execute(
                    """
            INSERT INTO PatientData (
                PatientID, Age, Sex, Cholesterol, BloodPressure, HeartRate, Diabetes, FamilyHistory, Smoking, 
                Obesity, AlcoholConsumption, ExerciseHoursPerWeek, Diet, PreviousHeartProblems, 
                MedicationUse, StressLevel, SedentaryHoursPerDay, Income, BMI, Triglycerides, 
                PhysicalActivityDaysPerWeek, SleepHoursPerDay, Country, Continent, Hemisphere, 
                HeartAttackRisk
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
                    rowData)

            self.dbConnection.commit()
            QMessageBox.information(None, "Success", "Data inserted into the database successfully!")
        except Exception as e:
            QMessageBox.critical(None, "Error", f"Failed to insert data: {e}")

    def updateData(self):
        try:
            if not hasattr(self, 'dbConnection'):
                QMessageBox.warning(None, "Warning", "No database connection. Connect to a database first.")
                return

            currentRow = self.ui.dataTable.currentRow()
            if currentRow == -1:
                QMessageBox.warning(None, "Warning", "Select a row to update.")
                return

            # Confirmation box
            confirm = QMessageBox.question(None, "Confirm Update", "Are you sure you want to update this record?",
                                           QMessageBox.Yes | QMessageBox.No)
            if confirm == QMessageBox.No:
                return

            # Get updated values from the table
            updatedValues = [self.ui.dataTable.item(currentRow, colIdx).text() for colIdx in
                             range(self.ui.dataTable.columnCount())]

            # Update the database
            cursor = self.dbConnection.cursor()
            cursor.execute("""
                 UPDATE PatientData SET
                Age=%s, Sex=%s, Cholesterol=%s, `BloodPressure`=%s, `HeartRate`=%s, Diabetes=%s, 
                `FamilyHistory`=%s, Smoking=%s, Obesity=%s, `AlcoholConsumption`=%s, 
                `ExerciseHoursPerWeek`=%s, Diet=%s, `PreviousHeartProblems`=%s, 
                `MedicationUse`=%s, `StressLevel`=%s, `SedentaryHoursPerDay`=%s, Income=%s, 
                BMI=%s, Triglycerides=%s, `PhysicalActivityDaysPerWeek`=%s, 
                `SleepHoursPerDay`=%s, Country=%s, Continent=%s, Hemisphere=%s, 
                `HeartAttackRisk`=%s
            WHERE `PatientID`=%s
            """, updatedValues[1:] + [updatedValues[0]])

            self.dbConnection.commit()
            QMessageBox.information(None, "Success", "Record updated successfully!")
        except Exception as e:
            QMessageBox.critical(None, "Error", f"Failed to update data: {e}")

    def retrieveData(self):
        try:
            if not hasattr(self, 'dbConnection'):
                QMessageBox.warning(None, "Warning", "No database connection. Connect to a database first.")
                return

            # Call the GUI function to get filter input
            dialogResult = self.ui.getFilterInputDialog()

            if not dialogResult:  # User cancelled the dialog
                return

            selectedField, filterValue = dialogResult
            # Construct and execute the SQL query
            cursor = self.dbConnection.cursor()
            if selectedField == "All Data":
                # Retrieve all records if "All Data" is selected
                query = "SELECT * FROM PatientData"
                cursor.execute(query)
            else:
                # Apply filtering based on user input
                if not filterValue:
                    QMessageBox.warning(None, "Warning", "Filter value cannot be empty unless 'All Data' is selected.")
                    return

                query = f"SELECT * FROM PatientData WHERE {selectedField} LIKE %s"
                cursor.execute(query, (f"{filterValue}",))


            # Fetch and display results
            results = cursor.fetchall()
            self.ui.dataTable.setRowCount(len(results))
            for rowIdx, row in enumerate(results):
                for colIdx, value in enumerate(row):
                    self.ui.dataTable.setItem(rowIdx, colIdx, QTableWidgetItem(str(value)))

            QMessageBox.information(None, "Success", f"Found {len(results)} matching records.")

        except Exception as e:
            QMessageBox.critical(None, "Error", f"Failed to retrieve data: {e}")

    # Function to Add Row
    def insertRow(self):
        rowPosition = self.ui.dataTable.rowCount()
        self.ui.dataTable.insertRow(rowPosition)

    def deleteData(self):
        try:
            if not hasattr(self, 'dbConnection'):
                QMessageBox.warning(None, "Warning", "No database connection. Connect to a database first.")
                return

            currentRow = self.ui.dataTable.currentRow()
            if currentRow == -1:
                QMessageBox.warning(None, "Warning", "Select a row to delete.")
                return

            patientID = self.ui.dataTable.item(currentRow, 0).text()  # Assuming Patient ID is the first column

            # Confirmation box
            confirm = QMessageBox.question(None, "Confirm Delete",
                                           f"Are you sure you want to delete the record for PatientID: {patientID}?",
                                           QMessageBox.Yes | QMessageBox.No)
            if confirm == QMessageBox.No:
                return
            cursor = self.dbConnection.cursor()
            cursor.execute("DELETE FROM PatientData WHERE PatientID = %s", (patientID,))

            self.dbConnection.commit()
            self.ui.dataTable.removeRow(currentRow)  # Remove row from table
            QMessageBox.information(None, "Success", "Record deleted successfully!")
        except Exception as e:
            QMessageBox.critical(None, "Error", f"Failed to delete data: {e}")


    def loadExcelData(self):
        try:
            myDlg = QFileDialog.getOpenFileName(None, 'OpenFile', "", "Excel Files (*.xls *.xlsx *.csv)")
            self.myPath = myDlg[0]  # Path + file + extension
            FileNameWithExtension = QFileInfo(myDlg[0]).fileName()  # Just the file + extension

            if myDlg[0] == myDlg[1] == '':
                # No file selected or cancel button clicked - so do nothing
                pass
            else:
                # Read and extract values from Excel
                try:
                    self.data = pd.read_csv(self.myPath)
                except:
                    self.data = pd.read_excel(self.myPath)

                # Change the text of the label to show the name of the file
                self.ui.lblExcel.setText(FileNameWithExtension)
                # Change the text of the second tab to show the path of the file
                self.ui.tabWidget.setTabText(1, self.myPath)
                # Extract headers of all columns and convert to a list
                self.df = pd.DataFrame(self.data)
                self.headers = self.df.columns.tolist()
                print(self.headers)
                # Add the list to all comboboxes
                # self.ui.cbX.addItems(self.headers)
                # self.ui.cbY.addItems(self.headers)
                # self.ui.cbMap.addItem("")
                # self.ui.cbMap.addItems(self.headers)
                # Prepare the table that will contain the Excel
                # Create N rows, N columns and add the headers
                self.ui.tableWidget.setRowCount(self.df.shape[0])
                self.ui.tableWidget.setColumnCount(self.df.shape[1])
                self.ui.tableWidget.setHorizontalHeaderLabels(self.headers)
                # A double loop to read the values and add them to the table
                for row in self.df.iterrows():
                    values = row[1]
                    for col, value in enumerate(values):
                        tableItem = QTableWidgetItem(str(value))
                        self.ui.tableWidget.setItem(row[0], col, tableItem)
                # Button to plot is now clickable
                self.ui.pbPlot.setEnabled(True)

        except IOError as e:
            print("I/O error({0}): {1}".format(e.errno, e.strerror))
            print(myDlg)
        except ValueError:
            print("Value Error.")
        except:
            print("Unexpected error:", sys.exc_info()[0])
            raise
    def cbXvaluechanged(self):
        X = self.ui.cbX.currentText()

        self.xmin = self.ui.sbXmin.setValue(self.data[X].min())
        self.xmax = self.ui.sbXmax.setValue(self.data[X].max())

    def cbYvaluechanged(self):
        Y = self.ui.cbY.currentText()

        self.ymin = self.ui.sbYmin.setValue(self.data[Y].min())
        self.ymax = self.ui.sbYmax.setValue(self.data[Y].max())


    def plotExcelData(self):
        global b_Canvas

        # Remove the previous plot
        if b_Canvas == True:
            self.rm_mpl()

        # Extract data and headers
        data = self.data
        X = self.ui.cbX.currentText()
        Y = self.ui.cbY.currentText()
        Cmap = self.ui.cbMap.currentText()

        # Extract the values from the 4 boxes to use them for the double slicing
        xmin = self.ui.sbXmin.value()
        xmax = self.ui.sbXmax.value()
        ymin = self.ui.sbYmin.value()
        ymax = self.ui.sbYmax.value()

        # Extract only the data respecting the 4 conditions
        NewData = data.loc[(data[X] > xmin) & (data[X] < xmax) & (data[Y] > ymin) & (data[Y] < ymax)]
        # It's one way, we could also use numpy, for example:
        # np.where((np.array(data[X]) > xmin) & (np.array(data[X]) < xmax))

        # Save new data in old data variable so we don't have to modify the plot part
        data = NewData

        # Prepare the plot
        fig = Figure()
        self.canvas = FigureCanvas(fig)
        self.ui.verticalLayout.addWidget(self.canvas)
        ax1f1 = fig.add_subplot(111)
        # Reformat the coordinates near the toolbar to not show values like 2.425e+04 but 24250 for example
        ax1f1.format_coord = lambda x, y: "x=%1.4f, y=%1.4f"%(x, y)

        # If no cmap then will do a normal plot (X,Y)
        if self.ui.cbMap.currentText() == "":
            ax1f1.plot(data[X], data[Y], 'o', markersize=2)
        elif self.ui.cbMap.currentText() == "Press":
            # If selecting pressure, swap the colormap so high pressure is in red instead of blue
            ax1f1.bar(data[X], data[Y], cmap='jet_r', c=data[Cmap])
        else:
            ax1f1.bar(data[X], data[Y], cmap='jet', c=data[Cmap])

        # Add a title for the figure and axis
        ax1f1.set_title('Title')
        ax1f1.set_xlabel("X", fontsize=14)
        ax1f1.set_ylabel("Y", fontsize=14)

        # If checkbox is checked, then Y axis is inverted
        if self.ui.checkBox.isChecked():
            ax1f1.invert_yaxis()

        # Draw everything and add toolbar
        self.canvas.draw()
        self.toolbar = NavigationToolbar(self.canvas, self.ui.mplwindow, coordinates=True)
        self.ui.verticalLayout.addWidget(self.toolbar)
        b_Canvas = True

    def rm_mpl(self):
        global b_Canvas
        # Remove plot part and toolbar
        self.ui.verticalLayout.removeWidget(self.canvas)
        self.canvas.close()
        self.ui.verticalLayout.removeWidget(self.toolbar)
        self.toolbar.close()
        b_Canvas = False

    def toggle_menu(self, checked):
        # Synchronize both menu buttons
        self.ui.pbMenu.setChecked(checked)
        self.ui.pbMenuIconName.setChecked(checked)

        # Show or hide widgets based on the checked state
        self.ui.icon_name_widget.setVisible(checked)
        self.ui.icon_only_widget.setHidden(checked)
        # Show or hide the MediMetrics label
        self.ui.mediMetricsLabel.setVisible(checked)

    def updateMainWindow(self):
        return

if __name__ == '__main__':
    import sys

    QApplication.setAttribute(QtCore.Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication(sys.argv)
    # To avoid weird behaviors (smaller items, ...) on big resolution screens
    app.setStyle("fusion")
    if hasattr(QtCore.Qt, 'AA_EnableHighDpiScaling'):
        QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    if hasattr(QtCore.Qt, 'AA_UseHighDpiPixmaps'):
        QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)

    window = MyMainWindow()
    window.show()
    sys.exit(app.exec_())


