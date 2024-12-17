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
from scipy.signal import find_peaks
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

import numpy as np
from numpy import *
from ProjectS1v1GUI import Ui_MainWindow
# import tensorflow as tf
# from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate
from skimage.io import imread, imshow
import skimage.morphology as morph

b_Canvas = False

class MyMainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MyMainWindow, self).__init__(parent)
        QMainWindow.__init__(self, parent)
        self.processed_image = None
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.ui.icon_name_widget.setHidden(True)


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

        self.ui.WindowcomboBox.currentTextChanged.connect(self.plotSignal)
        self.ui.ScalingcomboBox.currentTextChanged.connect(self.plotSignal)
        self.ui.NormcomboBox.currentTextChanged.connect(self.plotSignal)
        self.ui.pbValidate.clicked.connect(self.updateMainWindow)

        # Signals and Slots
        self.ui.btnLoadImage.clicked.connect(self.load_image)
        self.ui.btnGrayscale.clicked.connect(self.convert_to_grayscale)
        self.ui.btnDenoise.clicked.connect(self.apply_denoising)
        self.ui.btnBlur.clicked.connect(self.apply_blurring)
        self.ui.btnEdgeDetection.clicked.connect(self.apply_canny_edge)
        self.ui.sliderThreshold.valueChanged.connect(self.apply_thresholding)


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
        self.ui.btnRedo.setEnabled(False)
        self.ui.btnUndo.setEnabled(False)
        self.ui.btnClearDisplay.setEnabled(False)



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


        self.canvas = None
        self.toolbar = None

        self.history = []  # Stack for undo
        self.redo_stack = []  # Stack for redo






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

                # Detect R-peaks with a height threshold
                threshold = np.mean(self.signal) + 0.5 * np.std(self.signal)
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

    def plotSignal(self):
            # Clear existing canvas and toolbar if they exist
            try:
                layout = self.ui.mplwindow.layout()
                if self.canvas:
                    layout.removeWidget(self.canvas)
                    self.canvas.deleteLater()
                    self.canvas = None
                if self.toolbar:
                    layout.removeWidget(self.toolbar)
                    self.toolbar.deleteLater()
                    self.toolbar = None

                # Determine start time and window duration
                try:
                    # Parse start time from QLineEdit (hh:mm:ss)
                    start_time_str = self.ui.timeInput.text()
                    if start_time_str:
                        h, m, s = map(int, start_time_str.split(':'))
                        start_time = h * 3600 + m * 60 + s  # Convert to seconds
                    else:
                        start_time = 0  # Default to 0 if no input is provided
                except ValueError:
                    QMessageBox.warning(self, "Invalid Input", "Please enter time in hh:mm:ss format.")
                    return

                # Get the window duration from QSlider (default: 5 seconds)
                window_duration = self.ui.windowDurationSlider.value()

                # Calculate start and end indices
                start_index = int(start_time * self.data.fs)  # Sampling rate to index
                end_index = start_index + int(window_duration * self.data.fs)

                # Boundary checks
                if start_index < 0 or end_index > len(self.signal):
                    QMessageBox.warning(self, "Invalid Range", "The specified range is out of bounds.")
                    return

                # Extract the segment of the signal
                segment = self.signal[start_index:end_index]
                segment_time = self.time[start_index:end_index]

                peaks_indices = [i for i in self.peaks if start_index <= i < end_index]
                peaks_indices = np.array(peaks_indices) - start_index  # Adjust indices to the segment

                # Format time to hh:mm:ss for x-axis
                formatted_time = [time.strftime("%H:%M:%S", time.gmtime(t)) for t in segment_time]

                # Create subplots
                fig, ax = plt.subplots(1, 3, figsize=(10, 5))  # Create one figure with 3 subplots

                # Plot Signal
                ax[0].plot(segment_time, segment)
                ax[0].plot(segment_time[peaks_indices], segment[peaks_indices], "x", label='Peaks')
                ax[0].set_title(f"Signal (Duration: {window_duration}s, Start: {start_time}s, Stop: {start_time + window_duration}s)")
                ax[0].set_xlabel('Time (s)')
                ax[0].set_ylabel('Amplitude')
                ax[0].tick_params(axis='x', rotation=45)  # Rotate x-axis labels for readability
                # ax[0].xaxis.set_major_locator(plt.MaxNLocator(10))  # Adjust number of x-ticks
                ax[0].xaxis.set_major_locator(plt.MaxNLocator(integer=True))
                ax[0].tick_params(axis='x', rotation=45)  # Rotate for readability
                fig.autofmt_xdate()  # Auto-format the x-axis dates or time

                # fig, ax = plt.subplots(1, 3, figsize=(10, 5))  # Create one figure with 3 subplots
                # ax[0].plot(self.time, self.signal)
                # ax[0].set_title("Loaded Signal Data")
                # ax[0].set_xlabel('Time [s]')
                # ax[0].set_ylabel("Amplitude")
                # for tick in ax.xaxis.get_major_ticks():
                #     tick.label.set_fontsize(18)  # Adjust the fontsize for x ticks
                #
                # for tick in ax.yaxis.get_major_ticks():
                #     tick.label.set_fontsize(18)  # Adjust the fontsize for y ticks

                # FFT
                # Get the selected window and scaling values
                norm_value = self.ui.NormcomboBox.currentText()
                # Step 1: Calculate the FFT
                fft_values = np.fft.fft(segment, norm=norm_value)
                n = len(segment)
                # fft_values = np.fft.fft(self.signal, norm=norm_value)
                # # Step 2: Calculate the frequencies corresponding to the FFT values
                # n = len(self.signal)
                frequencies = np.fft.fftfreq(n, d=1 / self.data.fs)

                # Step 3: Get the positive frequencies and the corresponding FFT values
                positive_frequencies = frequencies[:n // 2]
                positive_fft_values = np.abs(fft_values[:n // 2])  # Magnitude of the FFT

                # Step 4: Plot the FFT
                ax[1].clear()
                ax[1].plot(positive_frequencies, positive_fft_values)
                ax[1].set_title(f'Frequency Spectrum of the Signal\n Norm = {norm_value}')
                ax[1].set_xlabel('Frequency (Hz)')
                ax[1].set_ylabel('Amplitude')
                # Periodogram
                scaling_value = self.ui.ScalingcomboBox.currentText()
                window_value = self.ui.WindowcomboBox.currentText()

                # f, Pxx = periodogram(self.signal, fs=self.data.fs, scaling=scaling_value,
                #                      window=window_value)
                f, Pxx = periodogram(segment, fs=self.data.fs, scaling=scaling_value, window=window_value)

                Pxx_dB = 10 * np.log10(Pxx)  # Convert power to dB


                ax[2].clear()
                ax[2].plot(f, Pxx_dB)
                ax[2].set_title(f'Periodogram of the Signal\n Window Type = {window_value}, Scaling = {scaling_value}'
                          )
                ax[2].set_xlabel('Frequency (Hz)')
                if scaling_value == 'spectrum':
                    ax[2].set_ylabel('Power(dB/Hz)')
                else:
                    ax[2].set_ylabel('Power/Frequency (dB/Hz)')
                # for tick in ax.xaxis.get_major_ticks():
                #     tick.label.set_fontsize(18)  # Adjust the fontsize for x ticks
                #
                # for tick in ax.yaxis.get_major_ticks():
                #     tick.label.set_fontsize(18)  # Adjust the fontsize for y ticks



                fig.tight_layout()
                self.canvas = FigureCanvas(fig)
                layout = self.ui.mplwindow.layout()
                layout.addWidget(self.canvas)
                self.canvas.draw()
                self.toolbar = NavigationToolbar(self.canvas, self.ui.mplwindow, coordinates=True)
                layout.insertWidget(0, self.toolbar)  # Insert at the top of the layout

            except Exception as e:
             print(e)

    def updateSliderLabel(self):
        try:
            step = 5  # Define the step size
            value = self.ui.windowDurationSlider.value()
            rounded_value = round(value / step) * step  # Round to the nearest step
            if value != rounded_value:  # Avoid infinite loops
                self.ui.windowDurationSlider.setValue(rounded_value)
            self.ui.windowDurationSlider.setText(f"{rounded_value} seconds")
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
                # self.save_state()

                # Show success message box
            QMessageBox.information(self, "Success", "Image successfully loaded!")
        except Exception as e:
            print(e)

    def display_image(self, image, label):
        """Utility to display an image in a QLabel."""
        if len(image.shape) == 3:
            # Color images with alpha channel
            if (image.shape[2]) == 4:
                qformat = QImage.Format_RGBA8888
            # Color images
            else:
                qformat = QImage.Format_RGB888
            h, w, ch = image.shape
            bytes_per_line = ch * w  # channel * width, also known as strides
            q_image = QtGui.QImage(image.data, w, h, bytes_per_line, qformat)
        else:
            # Grayscale images
            qformat = QImage.Format_Indexed8
            q_image = QImage(image, image.shape[1], image.shape[0], image.strides[0], qformat)

        # height, width, channel = image.shape
        # bytes_per_line = 3 * width
        # q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)

        label.setPixmap(QPixmap.fromImage(q_image).scaled(label.width(), label.height(), Qt.KeepAspectRatio))

    def convert_to_grayscale(self):
        """Convert the image to grayscale."""
        try:
            # Convert the processed image to grayscale and then to a binary image
            self.processed_image = cv2.cvtColor(self.processed_image, cv2.COLOR_RGB2GRAY)
            self.display_image(self.processed_image, self.ui.labelProcessedImage)
            self.ui.btnRedo.setEnabled(False)
            self.save_state()

        except Exception as e:
            print(f'Invalid format: Cannot convert to grayscale!{e}')

    def apply_thresholding(self):
        """Apply thresholding based on slider value."""
        if hasattr(self, 'original_image'):
            gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_RGB2GRAY)
            _, thresholded = cv2.threshold(gray_image, self.ui.sliderThreshold.value(), 255, cv2.THRESH_BINARY)
            self.processed_image = cv2.cvtColor(thresholded, cv2.COLOR_GRAY2RGB)
            self.ui.thresholdValueLabel.setText(f"Threshold Level: {self.ui.sliderThreshold.value()} (Min: 0, Max: 255)")
            self.display_image(self.processed_image, self.ui.labelProcessedImage)
            self.ui.btnRedo.setEnabled(False)
            self.save_state()


    def apply_blurring(self):
        """Apply a selected blur filter to the image."""
        if hasattr(self, 'original_image'):
            self.processed_image = cv2.GaussianBlur(self.processed_image, (5, 5), 0)
        else :
            self.processed_image = cv2.GaussianBlur(self.original_image, (5, 5), 0)
        self.display_image(self.processed_image, self.ui.labelProcessedImage)
        self.ui.btnRedo.setEnabled(False)
        self.save_state()

    def apply_denoising(self):
        try:
            if len(self.processed_image.shape) == 3:  # Convert to grayscale if RGB
                gray_image = cv2.cvtColor(self.processed_image, cv2.COLOR_RGB2GRAY)
            else:
                gray_image = self.processed_image
            self.processed_image = cv2.fastNlMeansDenoising(gray_image, None, 10, 7, 21)
            self.display_image(self.processed_image, self.ui.labelProcessedImage)
            self.ui.btnRedo.setEnabled(False)
            self.save_state()

        except Exception as e:
            print(f"Error in denoising: {e}")


    def apply_canny_edge(self):
        try:
            if len(self.processed_image.shape) == 3:  # Convert to grayscale if RGB
                gray_image = cv2.cvtColor(self.processed_image, cv2.COLOR_RGB2GRAY)
            else:
                gray_image = self.processed_image
            self.processed_image = cv2.Canny(gray_image, 70, 135)
            self.display_image(self.processed_image, self.ui.labelProcessedImage)
            self.save_state()
            self.ui.btnRedo.setEnabled(False)

        except Exception as e:
            print(f"Error in edge detection: {e}")

    # Morphological Operations

    def clear_borders(self):
        """Clear borders of objects in the image."""
        try:
            if len(self.processed_image.shape) == 3:
                # Convert the processed image to grayscale and then to a binary image
                self.processed_image = cv2.cvtColor(self.processed_image, cv2.COLOR_RGB2GRAY)
            _, self.processed_image = cv2.threshold(self.processed_image, 127, 255, cv2.THRESH_BINARY)
            labelled = label(self.processed_image)  # label all the constituents
            cleared = clear_border(labelled)
            self.processed_image = (cleared > 0).astype(np.uint8) * 255  # Convert back to binary
            self.display_image(self.processed_image, self.ui.labelProcessedImage)
            self.save_state()
            self.ui.btnRedo.setEnabled(False)

        except Exception as e:
            print(e)

    def fill_holes(self):
        """Fill holes in the objects of the image."""
        try:
            # Convert to grayscale if necessary
            if len(self.processed_image.shape) == 3:  # RGB
                gray_image = cv2.cvtColor(self.processed_image, cv2.COLOR_RGB2GRAY)
            else:
                gray_image = self.processed_image
            _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
            self.processed_image = binary_fill_holes(binary_image).astype(np.uint8) * 255
            self.display_image(self.processed_image, self.ui.labelProcessedImage)
            self.save_state()
            self.ui.btnRedo.setEnabled(False)

        except Exception as e:
            print(e)

    def remove_small_objects(self):
        try:
            if len(self.processed_image.shape) == 3:  # Convert to grayscale if RGB
                gray_image = cv2.cvtColor(self.processed_image, cv2.COLOR_RGB2GRAY)
            else:
                gray_image = self.processed_image
            _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
            labelled = label(binary_image)
            cleaned = remove_small_objects(labelled, min_size=500)
            self.processed_image = (cleaned > 0).astype(np.uint8) * 255  # Convert back to binary
            self.display_image(self.processed_image, self.ui.labelProcessedImage)
            self.save_state()
            self.ui.btnRedo.setEnabled(False)

        except Exception as e:
            print(f"Error in removing small objects: {e}")

    def apply_erosion(self):
        self.processed_image = erosion(self.processed_image)
        self.display_image(self.processed_image, self.ui.labelProcessedImage)
        self.save_state()

    def apply_dilation(self):
        self.processed_image = dilation(self.processed_image)
        self.display_image(self.processed_image, self.ui.labelProcessedImage)
        self.ui.btnRedo.setEnabled(False)
        self.save_state()

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

        # Clear results table
        self.ui.resultTable.setRowCount(0)

        # Reset internal states (if any)
        self.original_image = None
        self.processed_image = None


    def save_state(self):
        """Save the current state for undo."""
        try:
            if self.processed_image is not None:
                self.history.append(self.processed_image.copy())
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
                # self.display_image(self.processed_image, self.ui.labelProcessedImage)
                self.update_display()
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
    def apply_threshold(self, image, threshold):

        # Convert to grayscale if necessary
        if len(image.shape) == 3:  # RGB
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray_image = image

        # Apply thresholding
        _, thresholded_image = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresholded_image



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
                # try:
                #     self.data = pd.read_csv(self.myPath)
                # except:
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


