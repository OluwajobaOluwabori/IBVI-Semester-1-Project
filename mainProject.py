import os, sys, time, pdb
import cv2  # pip install opencv-python
import matplotlib.pyplot as plt
import pandas as pd, math
import wfdb
from scipy.signal import periodogram,welch
from scipy.signal.windows import hamming, hann, boxcar
from scipy.signal import find_peaks,butter, filtfilt
from skimage.morphology import erosion,dilation,remove_small_objects
from scipy.ndimage import binary_fill_holes
from skimage.measure import label, regionprops, regionprops_table
from skimage.segmentation import clear_border
from scipy.ndimage import binary_fill_holes
import csv
import json

import plotly.graph_objects as go
from PyQt5.QtWebEngineWidgets import QWebEngineView
import plotly.express as px


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
from skimage.io import imread, imshow
import skimage.morphology as morph
from scipy.stats import iqr
from PyQt5.QtWidgets import QApplication, QMainWindow, QDesktopWidget
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

        # Perform analysis


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
        self.ui.pbVideoProcessing1.clicked.connect(self.switch_to_VideoProcessingPage)
        self.ui.pbVideoProcessing2.clicked.connect(self.switch_to_VideoProcessingPage)

        # Link the clicked action from the button to a function to read excel file
        self.ui.LoadBtnsignal.clicked.connect(self.loadDatasignal)
        self.ui.updateTimeButton.clicked.connect(self.plotSignal)
        self.ui.windowDurationSlider.valueChanged.connect(self.updateSliderLabel)

        self.ui.WindowcomboBox.currentTextChanged.connect(self.plotPeriodogram)
        self.ui.ScalingcomboBox.currentTextChanged.connect(self.plotPeriodogram)
        self.ui.NormcomboBox.currentTextChanged.connect(self.plotFFT)

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
        self.ui.btnBoundary.clicked.connect(self.apply_boundary)
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
        self.analyze_patient_data()

        # Connect Signals to Slots
        self.ui.btnLoadVideo.clicked.connect(self.load_video)
        self.ui.btnvideoGrayscale.clicked.connect(self.convert_video_to_grayscale)
        self.ui.btnvideoEdgeDetection.clicked.connect(self.apply_edge_detection)
        self.ui.btnVisualizeChannels.clicked.connect(self.visualize_channels)
        self.ui.btnvideoThreshold.clicked.connect(self.apply_videothresholding)
        self.ui.btnSaveVideo.clicked.connect(self.save_video)
        self.ui.btnClearvideoDisplay.clicked.connect(self.clear_videodisplay)
        self.ui.btnPlayVideo.clicked.connect(self.play_video)
        self.ui.btnPauseVideo.clicked.connect(self.pause_video)
        self.ui.btnStopVideo.clicked.connect(self.stop_video)

        self.is_paused = False

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

    def switch_to_ImageProcessingPage(self):
        self.ui.stackedWidget.setCurrentIndex(3)
    def switch_to_VideoProcessingPage(self):
        self.ui.stackedWidget.setCurrentIndex(4)


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

            QMessageBox.critical(self, "Error", f"Failed to load data: {e}")
        except Exception as e:
            print(e)


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
            ax.set_title(f"FFT Plot\n Norm = {norm_value}")
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
            ax.set_title(f"Periodogram Plot\n Window Type = {window_value}, Scaling = {scaling_value}")
            ax.set_xlabel("Frequency (Hz)")
            if scaling_value == 'spectrum':
                ax.set_ylabel('Power(dB)')
            else:
                ax.set_ylabel('Power/Frequency (dB/Hz)')

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
            self.metrics['Interquartile range of RR intervals (ms)'] = iqr(rr_intervals) * 1000

            self.updateResultsTable(self.metrics)

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
        try:
            if hasattr(self, 'processed_image'):
                self.history.append(self.processed_image.copy())
                gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_RGB2GRAY)
                threshold_value = self.ui.thresholdValueInput.value()
                _, thresholded = cv2.threshold(gray_image,threshold_value, 255, cv2.THRESH_BINARY)
                self.processed_image = cv2.cvtColor(thresholded, cv2.COLOR_GRAY2RGB)
                self.display_image(self.processed_image, self.ui.labelProcessedImage)
                self.ui.btnUndo.setEnabled(True)
                self.ui.btnRedo.setEnabled(False)
        except Exception as e:
            print(e)


    def apply_blurring(self):
        """Apply a selected blur filter to the image."""
        try:
            self.history.append(self.processed_image.copy())
            kernel_size = int(self.ui.blurValueInput.currentText())  # Get selected kernel size
            if hasattr(self, 'original_image'):
                self.processed_image = cv2.GaussianBlur(self.processed_image, (kernel_size, kernel_size), 0)
            else :
                self.processed_image = cv2.GaussianBlur(self.original_image, (kernel_size, kernel_size), 0)
            self.display_image(self.processed_image, self.ui.labelProcessedImage)
            self.ui.btnUndo.setEnabled(True)
            self.ui.btnRedo.setEnabled(False)
        except Exception as e:
            print(f"Error in denoising: {e}")

    def apply_denoising(self):
        try:
            self.history.append(self.processed_image.copy())

            self.processed_image = cv2.fastNlMeansDenoising(self.processed_image, None, 10, 7, 21)
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
            self.processed_image = cv2.Canny(gray_image, 100, 200)
            self.display_image(self.processed_image, self.ui.labelProcessedImage)
            self.ui.btnUndo.setEnabled(True)
            self.ui.btnRedo.setEnabled(False)

        except Exception as e:
            print(f"Error in edge detection: {e}")

    def apply_boundary(self):
        try:
            self.history.append(self.processed_image.copy())
            if len(self.processed_image.shape) == 3:  # Convert to grayscale if RGB
                gray_image = cv2.cvtColor(self.processed_image, cv2.COLOR_RGB2GRAY)
            else:
                gray_image = self.processed_image
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

            eroded_image = cv2.erode(gray_image, kernel, iterations=1)
            self.processed_image = gray_image - eroded_image
            self.display_image(self.processed_image, self.ui.labelProcessedImage)
            self.ui.btnUndo.setEnabled(True)
            self.ui.btnRedo.setEnabled(False)

        except Exception as e:
            print(f"Error in extracting boundary: {e}")

    # Morphological Operations

    def clear_borders(self):
        """Clear borders of objects in the image."""
        try:
            self.history.append(self.processed_image.copy())
            if len(self.processed_image.shape) == 3:
                # Convert the processed image to grayscale and then to a binary image
                self.processed_image = cv2.cvtColor(self.processed_image, cv2.COLOR_RGB2GRAY)
            _, self.processed_image = cv2.threshold(self.processed_image, 120, 255, cv2.THRESH_BINARY)
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
            if len(self.processed_image.shape) == 3 or len(np.unique(self.processed_image)) > 2:  # RGB
                gray_image = cv2.cvtColor(self.processed_image, cv2.COLOR_RGB2GRAY)
                seed = np.copy(gray_image)
                seed[1:-1, 1:-1] = 0  # Set the interior pixels to 0

                # Dilate the seed image iteratively until it reaches the original image
                dilated = seed.copy()
                while True:
                    dilated_prev = dilated.copy()
                    dilated = cv2.dilate(dilated, np.ones((5, 5), np.uint8))
                    dilated = np.minimum(dilated, gray_image)
                    if np.array_equal(dilated, dilated_prev):
                        break

                # Subtract the dilated result to get the filled holes
                self.processed_image = gray_image - dilated
            else:
                gray_image = self.processed_image
            _, binary_image = cv2.threshold(gray_image, 120, 255, cv2.THRESH_BINARY)
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
            min_size = self.ui.minSizeInput.value()
            cleaned = remove_small_objects(labelled, min_size=min_size,connectivity=8)
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
        self.history = []


    def save_state(self):
        """Save the current state for undo."""
        try:
            if self.processed_image is not None:
                self.history.append(self.processed_image)
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

        except Exception as e:
            print(e)

    def load_video(self):
        """Load a video file and display the first frame."""
        try:
            options = QFileDialog.Options()
            file_path, _ = QFileDialog.getOpenFileName(self, "Open Video File", "",
                                                       "Videos (*.mp4 *.avi *.mov)", options=options)
            if file_path:
                self.video_path = file_path
                self.cap = cv2.VideoCapture(file_path)

                if not self.cap.isOpened():
                    raise Exception("Failed to load video.")

                self.original_video_frames = []
                while True:
                    ret, frame = self.cap.read()
                    if not ret: break
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    self.original_video_frames.append(frame)

                self.processed_video_frames = [f.copy() for f in self.original_video_frames]
                self.display_frame(self.original_video_frames[0], self.ui.labelOriginalVideo)
                QMessageBox.information(self, "Success", "Video loaded!")
        except Exception as e:
            print(f"load_video error: {e}")

    def play_video(self):
        """Play/resume video from current position."""
        try:
            # Only reset frame index if not paused
            if not hasattr(self, 'play_frame_index') or not self.is_paused:
                self.play_frame_index = 0

            self.is_paused = False  # Clear pause state

            # Stop existing timer if any
            if hasattr(self, 'play_timer') and self.play_timer.isActive():
                self.play_timer.stop()

            self.play_timer = QtCore.QTimer(self)

            def update_frames():
                if self.play_frame_index < len(self.original_video_frames):
                    # Update original frame
                    self.display_frame(
                        self.original_video_frames[self.play_frame_index],
                        self.ui.labelOriginalVideo
                    )
                    # Update processed frame
                    if self.processed_video_frames:
                        is_gray = len(self.processed_video_frames[self.play_frame_index].shape) == 2
                        self.display_frame(
                            self.processed_video_frames[self.play_frame_index],
                            self.ui.labelProcessedVideo,
                            is_gray=is_gray
                        )
                    self.play_frame_index += 1
                else:
                    self.play_timer.stop()
                    self.play_frame_index = 0  # Reset for next play

            self.play_timer.timeout.connect(update_frames)
            self.play_timer.start(50)
        except Exception as e:
            print(f"play_video error: {e}")

    def pause_video(self):
        """Pause video while retaining current position."""
        try:
            if hasattr(self, 'play_timer') and self.play_timer.isActive():
                self.play_timer.stop()
                self.is_paused = True  # Set pause state
        except Exception as e:
            print(f"pause_video error: {e}")

    def stop_video(self):
        """Stop all playback and reset displays."""
        try:
            # Stop all possible timers
            for timer_name in ['play_timer', 'display_timer']:
                if hasattr(self, timer_name):
                    timer = getattr(self, timer_name)
                    if timer.isActive(): timer.stop()

            # Reset frame indices
            self.play_frame_index = 0

            # Show first frames
            if self.original_video_frames:
                self.display_frame(self.original_video_frames[0], self.ui.labelOriginalVideo)
            if self.processed_video_frames:
                is_gray = len(self.processed_video_frames[0].shape) == 2
                self.display_frame(self.processed_video_frames[0], self.ui.labelProcessedVideo, is_gray)
        except Exception as e:
            print(f"stop_video error: {e}")
        except Exception as e:
            print(f"Error in stop_video: {e}")

    def update_frame(self, frames, label, is_gray=False):
        """Update a single frame on the specified QLabel."""
        try:
            if self.frame_index < len(frames):
                frame = frames[self.frame_index]
                if is_gray:
                    q_image = QImage(frame.data, frame.shape[1], frame.shape[0], frame.strides[0],
                                     QImage.Format_Grayscale8)
                else:
                    q_image = QImage(frame.data, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)

                label.setPixmap(QPixmap.fromImage(q_image).scaled(label.width(), label.height(), Qt.KeepAspectRatio))
                self.frame_index += 1
            else:
                self.frame_index = 0  # Loop back to the start
        except Exception as e:
            print(f"Error in update_frame: {e}")

    def convert_video_to_grayscale(self):
        """Convert video to grayscale."""
        try:
            self.stop_video()
            self.processed_video_frames = [
                cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                for frame in self.original_video_frames
            ]
            # Immediately show first processed frame
            self.display_frame(self.processed_video_frames[0], self.ui.labelProcessedVideo, is_gray=True)
        except Exception as e:
            print(f"grayscale error: {e}")

    def apply_edge_detection(self):
        """Apply edge detection to video."""
        try:
            self.stop_video()
            self.processed_video_frames = [
                cv2.Canny(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), 100, 200)
                for frame in self.original_video_frames
            ]
            self.display_frame(self.processed_video_frames[0], self.ui.labelProcessedVideo, is_gray=True)
        except Exception as e:
            print(f"edge detection error: {e}")

    def apply_videothresholding(self):
        """Apply thresholding to the video and display the processed video."""
        try:
            self.stop_video()
            self.processed_video_frames = [
                cv2.threshold(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), self.ui.thresholdVideoInput.value(), 255, cv2.THRESH_BINARY)[1]
                for frame in self.original_video_frames
            ]
            self.display_video(self.processed_video_frames[0], self.ui.labelProcessedVideo, is_gray=True)
        except Exception as e:
            print(f"Error in apply_thresholding: {e}")

    def display_video(self, frames, label, is_gray=False):
        """Play a video sequence on a specific label."""
        try:
            # Stop any existing display timer
            if hasattr(self, 'display_timer') and self.display_timer.isActive():
                self.display_timer.stop()

            self.display_timer = QtCore.QTimer(self)
            self.display_frame_index = 0

            def update_display():
                if self.display_frame_index < len(frames):
                    self.display_frame(frames[self.display_frame_index], label, is_gray)
                    self.display_frame_index += 1
                else:
                    self.display_timer.stop()

            # Show first frame immediately
            if frames:
                self.display_frame(frames[0], label, is_gray)

            self.display_timer.timeout.connect(update_display)
            self.display_timer.start(50)
        except Exception as e:
            print(f"display_video error: {e}")

    def visualize_channels(self):
        try:
            # Open the video file
            video_path = self.video_path
            cap = cv2.VideoCapture(video_path)

            # Define the color ranges for blue, red, and green in HSV
            lower_blue = np.array([100, 150, 50])  # Adjust the ranges for better detection
            upper_blue = np.array([140, 255, 255])

            lower_red1 = np.array([0, 150, 50])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([160, 150, 50])
            upper_red2 = np.array([180, 255, 255])

            lower_green = np.array([35, 100, 100])
            upper_green = np.array([85, 255, 255])

            # Process each frame of the video
            while True:
                ret, frame = cap.read()
                if not ret:
                    # Release and reinitialize the video capture to restart playback
                    cap.release()
                    cap = cv2.VideoCapture(video_path)
                    continue

                # Convert the frame to HSV
                hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

                # Create masks for each color
                blue_mask = cv2.inRange(hsv_frame, lower_blue, upper_blue)
                red_mask1 = cv2.inRange(hsv_frame, lower_red1, upper_red1)
                red_mask2 = cv2.inRange(hsv_frame, lower_red2, upper_red2)
                red_mask = red_mask1 | red_mask2  # Combine both red ranges
                green_mask = cv2.inRange(hsv_frame, lower_green, upper_green)

                # Extract regions for each color
                blue_region = cv2.bitwise_and(frame, frame, mask=blue_mask)
                red_region = cv2.bitwise_and(frame, frame, mask=red_mask)
                green_region = cv2.bitwise_and(frame, frame, mask=green_mask)
                # Display the results
                cv2.imshow('Original Video', frame)
                cv2.imshow('Blue Region', blue_region)
                cv2.imshow('Red Region', red_region)
                cv2.imshow('Green Region', green_region)

                # Press 'q' to exit
                if cv2.waitKey(50) & 0xFF == ord('q'):
                    break

            # Release resources
            cap.release()
            cv2.destroyAllWindows()
        except Exception as e:
            print(e)

    def create_color_window(self, color_name, frames):
        """Create and display a window for a specific color."""
        try:
            # Create a new window
            window = QtWidgets.QWidget()
            window.setWindowTitle(f"{color_name} Regions")
            layout = QtWidgets.QVBoxLayout(window)

            video_label = QtWidgets.QLabel(window)
            video_label.setAlignment(QtCore.Qt.AlignCenter)
            layout.addWidget(video_label)

            # Timer for frame playback
            timer = QtCore.QTimer(window)

            def update_frame():
                if frames:
                    frame = frames.pop(0)  # Get the next frame
                    q_image = QImage(frame.data, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)
                    video_label.setPixmap(QPixmap.fromImage(q_image).scaled(
                        video_label.width(), video_label.height(), Qt.KeepAspectRatio))
                # else:
                #     timer.stop()  # Stop when all frames are played

            timer.timeout.connect(update_frame)
            timer.start(100)  # Adjust playback speed

            window.show()
            return window  # Return the window reference
        except Exception as e:
            print(f"Error in create_color_window: {e}")

    def save_video(self):
        """Save the processed video."""
        try:
            if not self.processed_video_frames:
                QMessageBox.warning(self, "Error", "No processed video to save!")
                return

            file_path, _ = QFileDialog.getSaveFileName(self, "Save Processed Video", "",
                                                       "Videos (*.mp4 *.avi *.mov)")
            if file_path:
                height, width = self.processed_video_frames[0].shape[:2]
                is_color = len(self.processed_video_frames[0].shape) == 3
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4
                out = cv2.VideoWriter(file_path, fourcc, 20.0, (width, height), is_color)

                for frame in self.processed_video_frames:
                    if not is_color:
                        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                    out.write(frame)

                out.release()
                QMessageBox.information(self, "Success", "Processed video saved successfully!")
        except Exception as e:
            print(f"Error in save_video: {e}")

    def clear_videodisplay(self):
        """Clear the video display and reset internal states."""
        self.ui.labelOriginalVideo.clear()
        self.ui.labelOriginalVideo.setText("Original Video")
        self.ui.labelProcessedVideo.clear()
        self.ui.labelProcessedVideo.setText("Processed Video")
        self.original_video_frames = []
        self.processed_video_frames = []

    def display_frame(self, frame, label, is_gray=False):
        """Display a single frame on the specified QLabel."""
        try:
            if is_gray:
                q_image = QImage(frame.data, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_Grayscale8)
            else:
                q_image = QImage(frame.data, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)

            label.setPixmap(QPixmap.fromImage(q_image).scaled(label.width(), label.height(), Qt.KeepAspectRatio))
        except Exception as e:
            print(f"Error in display_frame: {e}")

    def show_video(self, frames, title, is_gray=False):
        """Display video frames in a new persistent window."""
        try:
            if not frames:
                raise ValueError("No frames provided to display.")

            window = QtWidgets.QWidget()
            window.setWindowTitle(title)
            layout = QtWidgets.QVBoxLayout(window)

            video_label = QtWidgets.QLabel(window)
            video_label.setAlignment(QtCore.Qt.AlignCenter)
            layout.addWidget(video_label)

            current_frame = 0

            def update_frame():
                nonlocal current_frame
                if current_frame < len(frames):
                    frame = frames[current_frame]
                    if is_gray:
                        q_image = QImage(frame.data, frame.shape[1], frame.shape[0], frame.strides[0],
                                         QImage.Format_Grayscale8)
                    else:
                        q_image = QImage(frame.data, frame.shape[1], frame.shape[0], frame.strides[0],
                                         QImage.Format_RGB888)
                    video_label.setPixmap(QPixmap.fromImage(q_image).scaled(video_label.width(), video_label.height(),
                                                                            Qt.KeepAspectRatio))
                    current_frame += 1
                else:
                    timer.stop()

            timer = QtCore.QTimer(window)
            timer.timeout.connect(update_frame)
            timer.start(50)

            window.show()
            return window  # Return the window reference
        except Exception as e:
            print(f"Error in show_video: {e}")

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

                query = f"SELECT * FROM PatientData WHERE {selectedField} = %s"
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

    # Data analysis
    def collect_data_from_database(self):
        """
        Collect data from the database.

        :param db_path: Path to the SQLite database.
        :param table_name: Table name containing patient data.
        :return: DataFrame containing the table data.
        """
        dbConnection = mysql.connector.connect(host="localhost", user="root", password="", database="patient_data")
        # cursor = dbConnection.cursor()
        query = f"SELECT * FROM patientdata"

        data = pd.read_sql(query, dbConnection)
        dbConnection.close()
        return data

    def generate_dashboard(self, data):
        # Total Patients
        total_patients = len(data)
        # Gender Distribution Pie Chart
        gender_counts = data['Sex'].value_counts()
        gender_fig = go.Figure(data=[go.Pie(labels=gender_counts.index, values=gender_counts.values)])
        gender_fig.update_layout(title="Gender Distribution",width=400, height=400, margin=dict(t=50, l=50, r=50, b=50))
        gender_fig.update_traces(marker=dict(colors=['#6A0DAD', '#5DADE2']))
        import plotly.colors as pc
        # Generate a color scale with the same length as unique_continents
        continent_colors = pc.qualitative.Prism[:6]  # Choose a qualitative palette

        country_fig = px.sunburst(data, path=['Continent', 'Country'], values=data['PatientID'].value_counts().values, color='Continent', color_discrete_sequence=continent_colors)
        country_fig.update_traces(marker=dict(colors=['#6A0DAD', '#5DADE2', '#8E44AD', '#3498DB']))
        country_fig.update_layout(title="Continent & Country Distribution",width=400, height=400, margin=dict(t=50, l=50, r=50, b=50))

        income_fig = px.violin(data, y="Income", x="Smoking", box=True, points="outliers",
                        hover_data=data.columns)
        income_fig.update_layout(title="Income Distribution",width=400, height=400, margin=dict(t=50, l=50, r=50, b=50))  # Dark blue-gray background

        income_fig.update_traces(marker=dict(color='#7B68EE'), box_visible=True, line_color='#6A0DAD')



        diet_fig = go.Figure(data=[go.Pie(labels=["Healthy", "Average", "Unhealthy"], values=data['Diet'].value_counts())])
        diet_fig.update_layout(title="Diet Distribution", width=400, height=400,
                                   margin=dict(t=50, l=50, r=50, b=50))
        diet_fig.update_traces(marker=dict(colors=['#6A0DAD', '#5DADE2']))

        medication_fig = go.Figure(
            data=[go.Pie(labels=["0", "1"], values=data['MedicationUse'].value_counts())])
        medication_fig.update_layout(title="Medication Use Distribution", width=400, height=400,
                               margin=dict(t=50, l=50, r=50, b=50))
        medication_fig.update_traces(marker=dict(colors=['#6A0DAD', '#5DADE2']))

        smoking_fig = go.Figure(
            data=[go.Pie(labels=["0", "1"], values=data['Smoking'].value_counts())])
        smoking_fig.update_layout(title="Smoking Distribution", width=400, height=400,
                                  margin=dict(t=50, l=50, r=50, b=50))
        smoking_fig.update_traces(marker=dict(colors=['#6A0DAD', '#5DADE2']))

        familyhistory_fig = go.Figure(
            data=[go.Pie(labels=["0", "1"], values=data['FamilyHistory'].value_counts())])
        familyhistory_fig.update_layout(title="FamilyHistory Distribution", width=400, height=400,
                                  margin=dict(t=50, l=50, r=50, b=50))
        familyhistory_fig.update_traces(marker=dict(colors=['#6A0DAD', '#5DADE2']))

        # Histograms
        histograms = []
        for col in ['Age', 'Cholesterol', 'HeartRate', 'BMI']:
            hist_fig = go.Figure(data=[go.Histogram(x=data[col], nbinsx=20)])
            hist_fig.update_traces(marker=dict(color='#6A0DAD'))
            hist_fig.update_layout(title=f"{col} Distribution", xaxis_title=col, yaxis_title="Count",width=400, height=400, margin=dict(t=50, l=50, r=50, b=50))
            histograms.append(hist_fig.to_html(full_html=False))

        # Boxplots
        boxplots = []
        for col in [ 'Triglycerides','BMI', 'StressLevel']:
            box_fig = go.Figure(data=[go.Box(y=data[col])])
            box_fig.update_traces(marker=dict(color='#6A0DAD'),  # Deep purple boxes
                                  boxmean=True,  # Show mean line
                                  line_color='#6A0DAD',  # Purple border
                                  fillcolor='#8A2BE2',  # Slightly lighter fill
                                  jitter=0.3,  # Spread points for visibility
                                  whiskerwidth=0.8)
            box_fig.update_layout(title=f"{col} Boxplot", yaxis_title=col,width=400, height=400, margin=dict(t=50, l=50, r=50, b=50))
            boxplots.append(box_fig.to_html(full_html=False))

        # Diabetes and Heart Attack Risk Distribution
        diabetes_fig = go.Figure(data=[go.Pie(labels=["Yes", "No"], values=data['Diabetes'].value_counts())])
        diabetes_fig.update_layout(title="Diabetes Distribution",width=400, height=400, margin=dict(t=50, l=50, r=50, b=50))
        diabetes_fig.update_traces(marker=dict(colors=['#6A0DAD', '#5DADE2']))

        heart_risk_fig = go.Figure(
            data=[go.Pie(labels=["At Risk", "Not At Risk"], values=data['HeartAttackRisk'].value_counts())])
        heart_risk_fig.update_layout(title="Heart Attack Risk Distribution",width=400, height=400, margin=dict(t=50, l=50, r=50, b=50))
        heart_risk_fig.update_traces(marker=dict(colors=['#6A0DAD', '#5DADE2']))

        # Dashboard HTML
        dashboard_html = f"""
        <h2>Data Summary</h2>
        <h3>Total Patients: {total_patients}</h3>
        <div style="display: flex; flex-wrap: wrap; gap: 20px;">
            <div style="flex: 1; min-width: 100px;">{gender_fig.to_html(full_html=False)}</div>
            <div style="flex: 1; min-width: 100px;">{country_fig.to_html(full_html=False)}</div>
            <div style="flex: 1; min-width: 100px;">{diet_fig.to_html(full_html=False)}</div>
        </div>
        <div style="display: flex; flex-wrap: wrap; gap: 20px;">
            <div style="flex: 1; min-width: 100px;">{smoking_fig.to_html(full_html=False)}</div>
            <div style="flex: 1; min-width: 100px;">{medication_fig.to_html(full_html=False)}</div>
            <div style="flex: 1; min-width: 100px;">{familyhistory_fig.to_html(full_html=False)}</div>
            
        </div>
        <div style="display: flex; flex-wrap: wrap; gap: 20px;">
            <div style="flex: 1; min-width: 100px;">{diabetes_fig.to_html(full_html=False)}</div>
            <div style="flex: 1; min-width: 100px;">{income_fig.to_html(full_html=False)}</div>
            <div style="flex: 1; min-width: 100px;">{heart_risk_fig.to_html(full_html=False)}</div>
        
        </div>
        <h2>Histograms</h2>
        <div style="display: flex; flex-wrap: wrap; gap: 20px;">
            {''.join(f'<div style="flex: 1; min-width: 100px;">{hist}</div>' for hist in histograms)}
        </div>
        <h2>Boxplots</h2>
        <div style="display: flex; flex-wrap: wrap; gap: 20px;">
            {''.join(f'<div style="flex: 1; min-width: 100px;">{box}</div>' for box in boxplots)}
        </div>
        """
        # Save the HTML to a file
        with open("dashboard.html", "w", encoding="utf-8") as file:
            file.write(dashboard_html)
        print("Dashboard HTML saved as 'dashboard.html'.")
        return dashboard_html

    def generate_gender_analysis(self, df):
        """
           Generate HTML for visualizing features by patient gender.

           :param df: Pandas DataFrame containing the patient data.
           :return: Combined HTML containing multiple plots.
           """
        # Feature 1: BMI Distribution by Gender
        bmi_fig = px.box(df, x="Sex", y="BMI", title="BMI Distribution by Gender",color='Sex')
        bmi_fig.update_traces(marker=dict(color='#6A0DAD'),  # Deep purple boxes
                              boxmean=True,  # Show mean line
                              line_color='#6A0DAD',  # Purple border
                              fillcolor='#8A2BE2',  # Slightly lighter fill
                              jitter=0.3,  # Spread points for visibility
                              whiskerwidth=0.8)
        bmi_html = bmi_fig.to_html(full_html=False, include_plotlyjs='cdn')

        # Feature 2: Stress Level Distribution by Gender
        stress_fig = px.histogram(df, x="StressLevel", color="Sex", barmode="group",
                                  title="Stress Level Distribution by Gender", facet_col="Sex", text_auto=True, color_discrete_map={"Male": "#6A0DAD", "Female": "#5DADE2"})
        stress_html = stress_fig.to_html(full_html=False)

        # Feature 3: Heart Attack Risk by Gender
        heart_attack_fig = px.histogram(
            df, x="Sex", color="HeartAttackRisk", barmode="group",
            title="Heart Attack Risk by Gender", text_auto=True,
            color_discrete_map={1.0: "#6A0DAD", 0.0: "#5DADE2"}
        )
        heart_attack_fig.update_layout(width=400, height=400, margin=dict(t=50, l=50, r=50, b=50))
        heart_attack_html = heart_attack_fig.to_html(full_html=False)

        # Feature 4: Smoking Habits by Gender
        smoking_fig = px.histogram(
            df, x="Sex", color="Smoking", barmode="group",
            title="Smoking Habits by Gender", text_auto=True,
            color_discrete_map={"1": "#6A0DAD", "0": "#5DADE2"}
        )
        smoking_fig.update_layout(width=400, height=400, margin=dict(t=50, l=50, r=50, b=50))
        smoking_html = smoking_fig.to_html(full_html=False)

        # Feature 5: Alcohol Consumption by Gender
        alcohol_fig = px.histogram(
            df, x="Sex", color="AlcoholConsumption", barmode="group",
            title="Alcohol Consumption by Gender", text_auto=True,
            color_discrete_map={"1": "#6A0DAD", "0": "#5DADE2"}
        )
        alcohol_fig.update_layout(width=400, height=400, margin=dict(t=50, l=50, r=50, b=50))
        alcohol_html = alcohol_fig.to_html(full_html=False)

        # Feature 6: Diabetes Distribution by Gender
        diabetes_fig = px.histogram(
            df, x="Sex", color="Diabetes", barmode="group",
            title="Diabetes Distribution by Gender", text_auto=True,
            color_discrete_map={"1": "#6A0DAD", "0": "#5DADE2"}
        )
        diabetes_fig.update_layout(width=400, height=400, margin=dict(t=50, l=50, r=50, b=50))
        diabetes_html = diabetes_fig.to_html(full_html=False)

        # Feature 7: Family History by Gender
        family_fig = px.histogram(
            df, x="Sex", color="FamilyHistory", barmode="group",
            title="Family History by Gender", text_auto=True,
            color_discrete_map={"1": "#6A0DAD", "0": "#5DADE2"}
        )
        family_fig.update_layout(width=400, height=400, margin=dict(t=50, l=50, r=50, b=50))
        family_html = family_fig.to_html(full_html=False)

        # Feature 8: Previous Heart Problems by Gender
        previous_heart_problems_fig = px.histogram(
            df, x="Sex", color="PreviousHeartProblems", barmode="group",
            title="Previous Heart Problems by Gender", text_auto=True,
            color_discrete_map={"1": "#6A0DAD", "0": "#5DADE2"}
        )
        previous_heart_problems_fig.update_layout(width=400, height=400, margin=dict(t=50, l=50, r=50, b=50))
        previous_heart_problems_html = previous_heart_problems_fig.to_html(full_html=False)

        # Combine all HTML plots
        combined_html = f"""
            <html>
            <head>
                <style>
                    body {{
                        font-family: Arial, sans-serif;
                        margin: 0;
                        padding: 0;
                    }}
                    .chart-container {{
                        margin: 20px auto;
                        max-width: 900px;
                    }}
                </style>
            </head>
            <body>
            <div style="display: flex; flex-wrap: wrap; gap: 20px;">
               <div style="flex: 1; min-width: 100px;">{heart_attack_html}</div>
                <div style="flex: 1; min-width: 100px;">{smoking_html}</div>
                <div style="flex: 1; min-width: 100px;">{alcohol_html}</div>
                </div>
                <div style="display: flex; flex-wrap: wrap; gap: 20px;">
                <div style="flex: 1; min-width: 100px;">{diabetes_html}</div>
                <div style="flex: 1; min-width: 100px;">{family_html}</div>
                <div style="flex: 1; min-width: 100px;">{previous_heart_problems_html}</div>
                </div>
        
                <div class="chart-container">{bmi_html}</div>
                <div class="chart-container">{stress_html}</div>
            </body>
            </html>
            """

        # Save combined HTML to a file
        with open("gender_analysis.html", "w", encoding="utf-8") as f:
            f.write(combined_html)
        return "gender_analysis.html"

    def generate_smoking_analysis(self, df):
        """
           Generate HTML for visualizing features by patient gender.

           :param df: Pandas DataFrame containing the patient data.
           :return: Combined HTML containing multiple plots.
           """
        # Feature 1: BMI Distribution by Smoking
        bmi_fig = px.box(df, x="Smoking", y="BMI", title="BMI Distribution by Smoking", color='Smoking')
        bmi_fig.update_traces(marker=dict(color='#6A0DAD'),  # Deep purple boxes
                              boxmean=True,  # Show mean line
                              line_color='#6A0DAD',  # Purple border
                              fillcolor='#8A2BE2',  # Slightly lighter fill
                              jitter=0.3,  # Spread points for visibility
                              whiskerwidth=0.8)
        bmi_html = bmi_fig.to_html(full_html=False, include_plotlyjs='cdn')

        # Feature 2: Stress Level Distribution by Smoking
        stress_fig = px.histogram(df, x="StressLevel", color="Smoking", barmode="group",
                                  title="Stress Level Distribution by Smoking", facet_col="Smoking", text_auto=True, color_discrete_map={"1": "#6A0DAD", "0": "#5DADE2"})
        stress_html = stress_fig.to_html(full_html=False)

        # Feature 3: Heart Attack Risk by Smoking
        heart_attack_fig = px.histogram(df, x="Smoking", color="HeartAttackRisk", barmode="group",
                                        title="Heart Attack Risk by Smoking", text_auto=True,
            color_discrete_map={1.0: "#6A0DAD", 0.0: "#5DADE2"})
        heart_attack_fig.update_layout(width=400, height=400, margin=dict(t=50, l=50, r=50, b=50))
        heart_attack_html = heart_attack_fig.to_html(full_html=False)

        # Feature 4: Smoking Habits by Smoking
        smoking_fig = px.histogram(df, x="Smoking", color="Sex", barmode="group",
                                   title="Smoking Habits by Gender", text_auto=True, color_discrete_map={"Male": "#6A0DAD", "Female": "#5DADE2"})
        smoking_fig.update_layout(width=400, height=400, margin=dict(t=50, l=50, r=50, b=50))
        smoking_html = smoking_fig.to_html(full_html=False)

        # Feature 5: Alcohol Consumption by Smoking
        alcohol_fig = px.histogram(df, x="Smoking", color="AlcoholConsumption", barmode="group",
                                   title="Alcohol Consumption by Smoking", text_auto=True, color_discrete_map={"1": "#6A0DAD", "0": "#5DADE2"})
        alcohol_fig.update_layout(width=400, height=400, margin=dict(t=50, l=50, r=50, b=50))
        alcohol_html = alcohol_fig.to_html(full_html=False)

        # Feature 6: Diabetes Habits by Smoking
        diabetes_fig = px.histogram(df, x="Smoking", color="Diabetes", barmode="group",
                                    title="Diabetes Distribution by Gender", text_auto=True, color_discrete_map={"1": "#6A0DAD", "0": "#5DADE2"})
        diabetes_fig.update_layout(width=400, height=400, margin=dict(t=50, l=50, r=50, b=50))
        diabetes_html = diabetes_fig.to_html(full_html=False)

        # Feature 7: FamilyHistory by Smoking
        family_fig = px.histogram(df, x="Smoking", color="FamilyHistory", barmode="group",
                                  title="Family History by Smoking", text_auto=True, color_discrete_map={"1": "#6A0DAD", "0": "#5DADE2"})
        family_fig.update_layout(width=400, height=400, margin=dict(t=50, l=50, r=50, b=50))
        family_html = family_fig.to_html(full_html=False)
        # Feature 8: PreviousHeartProblems by Gender
        PreviousHeartProblems_fig = px.histogram(df, x="Smoking", color="PreviousHeartProblems", barmode="group",
                                                 title="PreviousHeartProblems by Smoking", text_auto=True, color_discrete_map={"1": "#6A0DAD", "0": "#5DADE2"})
        PreviousHeartProblems_fig.update_layout(width=400, height=400, margin=dict(t=50, l=50, r=50, b=50))
        PreviousHeartProblems_html = PreviousHeartProblems_fig.to_html(full_html=False)

        # Combine all HTML plots
        combined_html = f"""
            <html>
            <head>
                <style>
                    body {{
                        font-family: Arial, sans-serif;
                        margin: 0;
                        padding: 0;
                    }}
                    .chart-container {{
                        margin: 20px auto;
                        max-width: 900px;
                    }}
                </style>
            </head>
            <body>
            <div style="display: flex; flex-wrap: wrap; gap: 20px;">
               <div style="flex: 1; min-width: 100px;">{heart_attack_html}</div>
                <div style="flex: 1; min-width: 100px;">{smoking_html}</div>
                <div style="flex: 1; min-width: 100px;">{alcohol_html}</div>
                </div>
                <div style="display: flex; flex-wrap: wrap; gap: 20px;">
                <div style="flex: 1; min-width: 100px;">{diabetes_html}</div>
                <div style="flex: 1; min-width: 100px;">{family_html}</div>
                <div style="flex: 1; min-width: 100px;">{PreviousHeartProblems_html}</div>
                </div>

                <div class="chart-container">{bmi_html}</div>
                <div class="chart-container">{stress_html}</div>
            </body>
            </html>
            """

        # Save combined HTML to a file
        with open("smoking_analysis.html", "w", encoding="utf-8") as f:
            f.write(combined_html)
        return "smoking_analysis.html"

    def generate_familyhistory_analysis(self, df):
        """
           Generate HTML for visualizing features by patient familyhistory.

           :param df: Pandas DataFrame containing the patient data.
           :return: Combined HTML containing multiple plots.
           """
        # Feature 1: BMI Distribution by Gender
        bmi_fig = px.box(df, x="FamilyHistory", y="BMI", title="BMI Distribution by FamilyHistory")
        bmi_fig.update_traces(marker=dict(color='#6A0DAD'),  # Deep purple boxes
                              boxmean=True,  # Show mean line
                              line_color='#6A0DAD',  # Purple border
                              fillcolor='#8A2BE2',  # Slightly lighter fill
                              jitter=0.3,  # Spread points for visibility
                              whiskerwidth=0.8)
        bmi_html = bmi_fig.to_html(full_html=False, include_plotlyjs='cdn')

        # Feature 2: Stress Level Distribution by FamilyHistory
        stress_fig = px.histogram(df, x="StressLevel", color="FamilyHistory", barmode="group",
                                  title="Stress Level Distribution by Gender", facet_col="FamilyHistory", text_auto=True,
            color_discrete_map={'1': "#6A0DAD", '0': "#5DADE2"})
        stress_html = stress_fig.to_html(full_html=False)

        # Feature 3: Heart Attack Risk by FamilyHistory
        heart_attack_fig = px.histogram(df, x="FamilyHistory", color="HeartAttackRisk", barmode="group",
                                        title="Heart Attack Risk by FamilyHistory", text_auto=True,
            color_discrete_map={1.0: "#6A0DAD", 0.0: "#5DADE2"})
        heart_attack_fig.update_layout(width=400, height=400, margin=dict(t=50, l=50, r=50, b=50))
        heart_attack_html = heart_attack_fig.to_html(full_html=False)

        # Feature 4: Smoking Habits by Gender
        smoking_fig = px.histogram(df, x="FamilyHistory", color="Smoking", barmode="group",
                                   title="FamilyHistory Habits by Smoking", text_auto=True,
            color_discrete_map={'1': "#6A0DAD", '0': "#5DADE2"})
        smoking_fig.update_layout(width=400, height=400, margin=dict(t=50, l=50, r=50, b=50))
        smoking_html = smoking_fig.to_html(full_html=False)

        # Feature 5: Alcohol Consumption by Gender
        alcohol_fig = px.histogram(df, x="FamilyHistory", color="AlcoholConsumption", barmode="group",
                                   title="Alcohol Consumption by FamilyHistory", text_auto=True,color_discrete_map={'1': "#6A0DAD", '0': "#5DADE2"})
        alcohol_fig.update_layout(width=400, height=400, margin=dict(t=50, l=50, r=50, b=50))
        alcohol_html = alcohol_fig.to_html(full_html=False)

        # Feature 6: Diabetes Habits by Gender
        diabetes_fig = px.histogram(df, x="FamilyHistory", color="Diabetes", barmode="group",
                                    title="Diabetes Distribution by FamilyHistory", text_auto=True, color_discrete_map={'1': "#6A0DAD", '0': "#5DADE2"})
        diabetes_fig.update_layout(width=400, height=400, margin=dict(t=50, l=50, r=50, b=50))
        diabetes_html = diabetes_fig.to_html(full_html=False)

        # Feature 7: Gender by FamilyHistory
        gender_fig = px.histogram(df, x="FamilyHistory", color="Sex", barmode="group",
                                  title="Gender by FamilyHistory", text_auto=True,color_discrete_map={'Male': "#6A0DAD", 'Female': "#5DADE2"})
        gender_fig.update_layout(width=400, height=400, margin=dict(t=50, l=50, r=50, b=50))
        gender_html = gender_fig.to_html(full_html=False)
        # Feature 8: PreviousHeartProblems by FamilyHistory
        PreviousHeartProblems_fig = px.histogram(df, x="FamilyHistory", color="PreviousHeartProblems", barmode="group",
                                                 title="PreviousHeartProblems by FamilyHistory", text_auto=True, color_discrete_map={'1': "#6A0DAD", '0': "#5DADE2"})
        PreviousHeartProblems_fig.update_layout(width=400, height=400, margin=dict(t=50, l=50, r=50, b=50))
        PreviousHeartProblems_html = PreviousHeartProblems_fig.to_html(full_html=False)

        # Combine all HTML plots
        combined_html = f"""
            <html>
            <head>
                <style>
                    body {{
                        font-family: Arial, sans-serif;
                        margin: 0;
                        padding: 0;
                    }}
                    .chart-container {{
                        margin: 20px auto;
                        max-width: 900px;
                    }}
                </style>
            </head>
            <body>
            <div style="display: flex; flex-wrap: wrap; gap: 20px;">
               <div style="flex: 1; min-width: 100px;">{heart_attack_html}</div>
                <div style="flex: 1; min-width: 100px;">{smoking_html}</div>
                <div style="flex: 1; min-width: 100px;">{alcohol_html}</div>
                </div>
                <div style="display: flex; flex-wrap: wrap; gap: 20px;">
                <div style="flex: 1; min-width: 100px;">{diabetes_html}</div>
                <div style="flex: 1; min-width: 100px;">{gender_html}</div>
                <div style="flex: 1; min-width: 100px;">{PreviousHeartProblems_html}</div>
                </div>

                <div class="chart-container">{bmi_html}</div>
                <div class="chart-container">{stress_html}</div>
            </body>
            </html>
            """

        # Save combined HTML to a file
        with open("familyhistory_analysis.html", "w", encoding="utf-8") as f:
            f.write(combined_html)
        return "familyhistory_analysis"


    def plot_cholesterol_distribution(self, df):
        """
        Plots cholesterol distribution with users having heart attack risk in red and others in blue.

        :param df: Pandas DataFrame containing 'Cholesterol' and 'HeartAttackRisk' columns.
        """
        # Ensure required columns are present
        if 'Cholesterol' not in df.columns or 'HeartAttackRisk' not in df.columns:
            raise ValueError("DataFrame must contain 'Cholesterol' and 'HeartAttackRisk' columns.")

        # Plot the histogram
        fig = px.histogram(
            df,
            x='Cholesterol',
            color='HeartAttackRisk',
            color_discrete_map={
                0.0: '#5DADE2',  # Users without heart attack risk
                1.0: '#6A0DAD'  # Users with heart attack risk
            },
            nbins=30,  # Adjust bin size as needed
            title="Heart Attack Risk by Cholesterol Distribution",
            labels={'Cholesterol': 'Cholesterol Levels', 'HeartAttackRisk': 'Heart Attack Risk'}
        )


        # Update layout for better visualization
        fig.update_layout(
            xaxis_title="Cholesterol Levels",
            yaxis_title="Count",
            legend_title="Heart Attack Risk",
            height=500,
            width=500,
            margin=dict(t=50, l=50, r=50, b=50),
            title_x=0.5
        )
        cholesterol_html = fig.to_html(full_html=False)


        # Feature 2: Heart Attack Risk by Diabetes
        Diabetes_fig = px.histogram(df, x="Diabetes", color="HeartAttackRisk", barmode="group",
                                        title="Heart Attack Risk by Diabetes", text_auto=True, color_discrete_map={1.0: "#6A0DAD", 0.0: "#5DADE2"})
        Diabetes_html = Diabetes_fig.to_html(full_html=False)

        # Feature 3: Heart Attack Risk by AlcoholConsumption
        AlcoholConsumption_fig = px.histogram(df, x="AlcoholConsumption", color="HeartAttackRisk", barmode="group",
                                    title="Heart Attack Risk by AlcoholConsumption", text_auto=True, color_discrete_map={1.0: "#6A0DAD", 0.0: "#5DADE2"})
        AlcoholConsumption_html = AlcoholConsumption_fig.to_html(full_html=False)

        # Feature 3: Heart Attack Risk by BMI

        # Plot the histogram
        bmi_fig = px.histogram(
            df,
            x='BMI',
            color='HeartAttackRisk',
            color_discrete_map={
                0.0: '#5DADE2',  # Users without heart attack risk
                1.0: '#6A0DAD'  # Users with heart attack risk
            },
            nbins=30,  # Adjust bin size as needed
            title="Heart Attack Risk by BMI Distribution",
            labels={'BMI': 'BMI Values', 'HeartAttackRisk': 'Heart Attack Risk'}
        )

        # Update layout for better visualization
        bmi_fig.update_layout(
            xaxis_title="BMI Values",
            yaxis_title="Count",
            legend_title="Heart Attack Risk",
            height=500,
            width=500,
            margin=dict(t=50, l=50, r=50, b=50),
            title_x=0.5
        )
        bmi_html = bmi_fig.to_html(full_html=False)


        # Combine all HTML plots
        combined_html = f"""
                    <html>
                    <head>
                        <style>
                            body {{
                                font-family: Arial, sans-serif;
                                margin: 0;
                                padding: 0;
                            }}
                            .chart-container {{
                                margin: 20px auto;
                                max-width: 900px;
                            }}
                        </style>
                    </head>
                    <body>
                    <div style="display: flex; flex-wrap: wrap; gap: 20px;">
                       <div style="flex: 1; min-width: 100px;">{AlcoholConsumption_html}</div>
                       <div style="flex: 1; min-width: 100px;">{Diabetes_html}</div>
                       </div>
                       <div style="display: flex; flex-wrap: wrap; gap: 20px;">
                       <div style="flex: 1; min-width: 100px;">{cholesterol_html}</div>
                       <div style="flex: 1; min-width: 100px;">{bmi_html}</div>
                        </div>
                    </body>
                    </html>
                    """

        # Save combined HTML to a file
        with open("cholesterol_analysis.html", "w", encoding="utf-8") as f:
            f.write(combined_html)
        return "cholesterol_analysis"



    def generate_correlation_heatmap(self, df):
        try:
            """
            Generate a correlation heatmap for the given DataFrame.
    
            :param df: Pandas DataFrame containing the dataset.
            :return: Path to the generated HTML file.
            """
            # Compute the correlation matrix
            # Include only numeric columns, including binary columns
            df['Sex'] = df['Sex'].apply(lambda x: 1 if x == 'Female' else 0)

            df.Diet = df['Diet'].apply(lambda x: 0 if x == 'Unhealthy' else (1 if x == 'Average' else 2))
            df.FamilyHistory = df.FamilyHistory.apply(lambda x: 1 if x == '1' else 0)  # check the dataset
            df.Diabetes = df.Diabetes.apply(lambda x: 1 if x == '1' else 0)
            df.AlcoholConsumption = df.AlcoholConsumption.apply(lambda x: 1 if x == '1' else 0)

            numeric_columns = df.select_dtypes(include=['number'])
            numeric_columns = numeric_columns.drop(columns=['Diet', 'StressLevel', 'PhysicalActivityDaysPerWeek','SleepHoursPerDay', 'ExerciseHoursPerWeek'])
            corr_matrix = numeric_columns.corr(numeric_only=True,method='spearman')

            # Create a heatmap using Plotly
            heatmap_fig = go.Figure(
                data=go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns,
                    y=corr_matrix.columns,
                    text=corr_matrix.round(2).values,  # Add text labels (rounded to 2 decimal places)
                    texttemplate="%{text}",  # Format the text
                    textfont=dict(size=10),
                    # colorscale="Bluered",
                    zmin=-1,
                    zmax=1,
                    colorscale='Purples',
                    colorbar=dict(title="Correlation"),
                )
            )

            # Update layout
            heatmap_fig.update_layout(
                title="Correlation Heatmap",
                xaxis_title="Features",
                yaxis_title="Features",
                height=600,
                width=800,
                margin=dict(t=50, l=50, r=50, b=50),
                title_x=0.5
            )

            # Save the plot to an HTML file
            heatmap_html = heatmap_fig.to_html(full_html=True)

            # Combine all HTML plots
            combined_html = f"""
                <html>
                <head>
                    <style>
                        body {{
                            font-family: Arial, sans-serif;
                            margin: 0;
                            padding: 0;
                        }}
                        .chart-container {{
                            margin: 20px auto;
                            max-width: 900px;
                        }}
                    </style>
                </head>
                <body>
                            <div class="chart-container">{heatmap_html}</div>
                         </body>
                </html>
                """

            # Save combined HTML to a file
            with open("correlation_analysis.html", "w", encoding="utf-8") as f:
                f.write(combined_html)
        except Exception as e:
            print(e)
        return combined_html


    def analyze_patient_data(self):
        """
        Combine data collection and analysis.

        :param db_path: Path to the SQLite database.
        :param table_name: Table name containing patient data.
        :return: Dictionary containing analysis results.
        """
        df = self.collect_data_from_database()
        self.df = df
        dashboard_html = self.generate_dashboard(df)



        gender_html = self.generate_gender_analysis(df)
        smoking_html = self.generate_smoking_analysis(df)
        familyhistory_html = self.generate_familyhistory_analysis(df)
        cholesterol_html = self.plot_cholesterol_distribution(df)

        # Create a QWebEngineView and set the HTML content
        web_view = QWebEngineView()
        web_view.setHtml(dashboard_html)
        # Add QWebEngineView to the Overview tab layout
        file_path = "C:/Users/user/OneDrive - UPEC/ProjectS1/dashboard.html"  # Replace with your file path
        web_view.load(QUrl.fromLocalFile(file_path))
        self.ui.layout_overview.addWidget(web_view)


        # Create QWebEngineView to display the HTML
        web_view2 = QWebEngineView()
        web_view2.setHtml(gender_html)
        web_view2.load(QUrl.fromLocalFile("C:/Users/user/OneDrive - UPEC/ProjectS1/gender_analysis.html"))

        # Add the QWebEngineView to the tab layout
        self.ui.layout_sex.addWidget(web_view2)

        # Create QWebEngineView to display the HTML
        web_view3 = QWebEngineView()
        web_view3.setHtml(smoking_html)
        web_view3.load(QUrl.fromLocalFile("C:/Users/user/OneDrive - UPEC/ProjectS1/smoking_analysis.html"))

        # Add the QWebEngineView to the tab layout
        self.ui.layout_Smoking.addWidget(web_view3)

        # Create QWebEngineView to display the HTML
        web_view4 = QWebEngineView()
        web_view4.setHtml(familyhistory_html)
        web_view4.load(QUrl.fromLocalFile("C:/Users/user/OneDrive - UPEC/ProjectS1/familyhistory_analysis.html"))

        # Add the QWebEngineView to the tab layout
        self.ui.layout_Family.addWidget(web_view4)

        # Create QWebEngineView to display the HTML
        web_view5 = QWebEngineView()
        web_view5.setHtml(cholesterol_html)
        web_view5.load(QUrl.fromLocalFile("C:/Users/user/OneDrive - UPEC/ProjectS1/cholesterol_analysis.html"))

        # Add the QWebEngineView to the tab layout
        self.ui.layout_risk_factors.addWidget(web_view5)

        # Generate the correlation heatmap
        variables = ["BMI", "HeartAttackRisk", "ExerciseHoursPerWeek", "Cholesterol", "Triglycerides", "Smoking", "Diabetes", "FamilyHistory"]

        # self.populate_correlation_dropdown(df)
        heatmap_file = self.generate_correlation_heatmap(df)



        # Create QWebEngineViews for the plots

        heatmap_view = QWebEngineView()
        heatmap_view.setHtml(heatmap_file)
        heatmap_view.load(QUrl.fromLocalFile("C:/Users/user/OneDrive - UPEC/ProjectS1/correlation_analysis.html"))


        # Add the views to the tab layout
        self.ui.layout_correlation.addWidget(heatmap_view)

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


