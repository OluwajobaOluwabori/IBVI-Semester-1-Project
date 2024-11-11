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
from PeriodogramWindow import Ui_Form as Ui_FormPeriodogram
from FFTWindow import Ui_Form as Ui_FormFFT
# from RotateWindow import Ui_Form as Ui_FormRotate
# from BrightnessWindow import Ui_Form as Ui_FormBright
# from SaturationWindow import Ui_Form as Ui_FormSaturation
b_Canvas = False

class PeriodogramWindow(QWidget):
    def __init__(self, parent):
        super(PeriodogramWindow, self).__init__()
        self.parent = parent
        QWidget.__init__(self)
        Ui_FormPeriodogram.__init__(self)

        self.p_ui = Ui_FormPeriodogram()
        self.p_ui.setupUi(self)

        self.p_ui.WindowcomboBox.currentTextChanged.connect(self.update_periodogram_plot)
        self.p_ui.ScalingcomboBox.currentTextChanged.connect(self.update_periodogram_plot)
        self.p_ui.pbValidatePeriodogram.clicked.connect(self.update_periodogram_plot)
        # self.displayImage(image=self.parent.out_image, image2=self.parent.out_image)
        # self.show()
        self.canvas = None
        self.bGammaChanged = False



    def update_periodogram_plot(self):
        # Remove the previous canvas if it exists
        if self.canvas is not None:
            self.p_ui.verticalLayout.removeWidget(self.canvas)
            self.canvas.deleteLater()
            self.canvas = None
        # Remove previous toolbar if it exists
        if hasattr(self, 'toolbar'):
            self.p_ui.verticalLayout.removeWidget(self.toolbar)
            self.toolbar.deleteLater()
        try:
            # Get the selected window and scaling values
            scaling_value = self.p_ui.ScalingcomboBox.currentText()
            window_value = self.p_ui.WindowcomboBox.currentText()

            # Calculate periodogram based on parameters
            if window_value.lower() == 'hamming':
                window_func = hamming(len(self.parent.signal))
            elif window_value.lower() == 'hann':
                window_func = hann(len(self.parent.signal))
            else:
                window_func = boxcar(len(self.parent.signal))  # Default to 'boxcar'

            f, Pxx = periodogram(self.parent.signal, fs=self.parent.data.fs, scaling=scaling_value, window=window_func)
            Pxx_dB = 10 * np.log10(Pxx)  # Convert power to dB

            # Plot within Periodogramlabel using a FigureCanvas
            fig = Figure(figsize=(12, 6))
            ax = fig.add_subplot(111)
            ax.plot(f, Pxx_dB)
            ax.set_title(f'Periodogram of the Signal: Window Type = {window_value}, Scaling = {scaling_value}', fontsize=24)
            ax.set_xlabel('Frequency (Hz)', fontsize=22)
            if scaling_value == 'spectrum':
                ax.set_ylabel('Power(dB/Hz)', fontsize=22)
            else:
                ax.set_ylabel('Power/Frequency (dB/Hz)', fontsize=22)
            for tick in ax.xaxis.get_major_ticks():
                tick.label.set_fontsize(18)  # Adjust the fontsize for x ticks

            for tick in ax.yaxis.get_major_ticks():
                tick.label.set_fontsize(18)  # Adjust the fontsize for y ticks
            # Embed the plot in the QLabel (Periodogramlabel)

            self.canvas = FigureCanvas(fig)
            self.toolbar = NavigationToolbar(self.canvas, self)  # Create the toolbar

            self.p_ui.Periodogramlabel.setLayout(QVBoxLayout())
            self.p_ui.Periodogramlabel.layout().addWidget(self.toolbar)  # Add the toolbar first

            self.p_ui.Periodogramlabel.layout().addWidget(self.canvas)
            # self.p_ui.verticalLayout.addWidget(self.canvas)
            self.canvas.draw()
        except Exception as e:
            print(f"Error updating periodogram plot: {e}")

    def plotPeriodogram(self):
        """Method to plot periodogram in Periodogramlabel based on selected parameters."""
        pass

    def rm_mpl(self):
        global b_Canvas
        if b_Canvas:
            self.p_ui.verticalLayout.removeWidget(self.canvas)
            self.canvas.deleteLater()
            b_Canvas = False

class FFTWindow(QWidget):
    def __init__(self, parent):
        super(FFTWindow, self).__init__()
        self.parent = parent
        QWidget.__init__(self)
        Ui_FormFFT.__init__(self)

        self.p_ui = Ui_FormFFT()
        self.p_ui.setupUi(self)

        self.p_ui.comboBox.currentTextChanged.connect(self.update_FFT_plot)
        self.p_ui.pbValidateFFT.clicked.connect(self.update_FFT_plot)
        # self.displayImage(image=self.parent.out_image, image2=self.parent.out_image)
        # self.show()
        self.canvas = None
        self.bGammaChanged = False


    def update_FFT_plot(self):
        # Remove the previous canvas if it exists
        if self.canvas is not None:
            self.p_ui.verticalLayout.removeWidget(self.canvas)
            self.canvas.deleteLater()
            self.canvas = None
        # Remove previous toolbar if it exists
        if hasattr(self, 'toolbar'):
            self.p_ui.verticalLayout.removeWidget(self.toolbar)
            self.toolbar.deleteLater()
        try:
            # Get the selected window and scaling values
            norm_value = self.p_ui.comboBox.currentText()

            # Step 1: Calculate the FFT
            fft_values = np.fft.fft(self.parent.signal,norm=norm_value)
            # Step 2: Calculate the frequencies corresponding to the FFT values
            n = len(self.parent.signal)
            frequencies = np.fft.fftfreq(n, d=1 / self.parent.data.fs)

            # Step 3: Get the positive frequencies and the corresponding FFT values
            positive_frequencies = frequencies[:n // 2]
            positive_fft_values = np.abs(fft_values[:n // 2])  # Magnitude of the FFT

            # Step 4: Plot the FFT
            fig = Figure(figsize=(12, 6))
            ax = fig.add_subplot(111)
            ax.plot(positive_frequencies, positive_fft_values)
            ax.set_title(f'Frequency Spectrum of the Signal: Norm = {norm_value}', fontsize=24)
            ax.set_xlabel('Frequency (Hz)', fontsize=22)
            ax.set_ylabel('Amplitude', fontsize=22)
            for tick in ax.xaxis.get_major_ticks():
                tick.label.set_fontsize(18)  # Adjust the fontsize for x ticks

            for tick in ax.yaxis.get_major_ticks():
                tick.label.set_fontsize(18)  # Adjust the fontsize for y ticks

            # Embed the plot in the QLabel (Periodogramlabel)

            self.canvas = FigureCanvas(fig)
            self.toolbar = NavigationToolbar(self.canvas, self)  # Create the toolbar

            self.p_ui.FFTlabel.setLayout(QVBoxLayout())
            self.p_ui.FFTlabel.layout().addWidget(self.toolbar)  # Add the toolbar first

            self.p_ui.FFTlabel.layout().addWidget(self.canvas)
            # self.p_ui.verticalLayout.addWidget(self.canvas)
            self.canvas.draw()
        except Exception as e:
            print(f"Error updating periodogram plot: {e}")

    def FFT(self):
        try:
            # Step 1: Calculate the FFT
            fft_values = np.fft.fft(self.signal)
            # Step 2: Calculate the frequencies corresponding to the FFT values
            n = len(self.signal)
            frequencies = np.fft.fftfreq(n, d=1 / self.data.fs)

            # Step 3: Get the positive frequencies and the corresponding FFT values
            positive_frequencies = frequencies[:n // 2]
            positive_fft_values = np.abs(fft_values[:n // 2])  # Magnitude of the FFT

            # Step 4: Plot the FFT
            plt.figure(figsize=(10, 6))
            plt.plot(positive_frequencies, positive_fft_values)
            plt.title('Frequency Spectrum of ECG Signal')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Amplitude')
            plt.grid(True)
            plt.show()
        except Exception as e:
            print(e)

class MyMainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MyMainWindow, self).__init__(parent)
        QMainWindow.__init__(self, parent)
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
        self.ui.LoadBtn.clicked.connect(self.loadData)
        self.ui.WindowcomboBox.currentTextChanged.connect(self.loadData)
        self.ui.ScalingcomboBox.currentTextChanged.connect(self.loadData)

        # Link the clicked action from the button to a function to plot data
        # self.ui.pbPlot.clicked.connect(self.plotData)
        # self.ui.pbFFT.clicked.connect(self.showFFTWindow)
        # self.ui.pbPeriodogram.clicked.connect(self.showPeriodogramWindow)


        # Button can't be clicked because no reason to if Excel not yet loaded
        # self.ui.pbPlot.setEnabled(False)
        # self.ui.pbFFT.setEnabled(False)
        # self.ui.pbPeriodogram.setEnabled(False)

        self.canvas = None


        # Initialize periodogram window variable
        self.periodogram_window = None
        self.FFT_window = None


    def showPeriodogramWindow(self):
        """Method to open PeriodogramWindow as a separate widget."""
        if self.periodogram_window is None:  # Open only if not already open
            self.periodogram_window = PeriodogramWindow(self)
            self.periodogram_window.show()
        else:
            # Bring the existing window to the front
            self.periodogram_window.raise_()
            self.periodogram_window.activateWindow()

    def showFFTWindow(self):
        """Method to open PeriodogramWindow as a separate widget."""
        if self.FFT_window is None:  # Open only if not already open
            self.FFT_window = FFTWindow(self)
            self.FFT_window.show()
        else:
            # Bring the existing window to the front
            self.FFT_window.raise_()
            self.FFT_window.activateWindow()

    def switch_to_dataManagementPage(self):
        self.ui.stackedWidget.setCurrentIndex(0)
    def switch_to_DataAnalysisPage(self):
        self.ui.stackedWidget.setCurrentIndex(1)
    def switch_to_SpectrumAnalysisPage(self):
        self.ui.stackedWidget.setCurrentIndex(2)
        self.ui.LoadBtn.clicked.connect(self.loadData)
        # self.ui.pbPeriodogram.clicked.connect(self.Periodogram)
    def switch_to_ImageProcessingPage(self):
        self.ui.stackedWidget.setCurrentIndex(3)


    def loadData(self):
        try:
            myDlg = QFileDialog.getOpenFileName(None, 'OpenFile', "", "Record Files (*.dat)")

            self.myPath = myDlg[0][:-4] # Path + file + extension
            FileNameWithExtension = QFileInfo(myDlg[0]).fileName()  # Just the file + extension
            print(self.myPath)

            if myDlg[0] == myDlg[1] == '':
                # No file selected or cancel button clicked - so do nothing
                pass
            else:
                # Read and extract values from Excel
                self.data = wfdb.rdrecord(self.myPath)
                self.signal = self.data.p_signal[:, 0]
                self.data.fs = self.data.fs
                self.data.n_sig = self.data.n_sig
                fig = Figure(figsize=(12, 12))
                ax = fig.add_subplot(121)
                ax.plot(self.signal)
                ax.set_title("Loaded Signal Data")
                ax.set_xlabel("Samples")
                ax.set_ylabel("Amplitude")
                # for tick in ax.xaxis.get_major_ticks():
                #     tick.label.set_fontsize(18)  # Adjust the fontsize for x ticks
                #
                # for tick in ax.yaxis.get_major_ticks():
                #     tick.label.set_fontsize(18)  # Adjust the fontsize for y ticks


                self.canvas = FigureCanvas(fig)
                layout = self.ui.mplwindow.layout()
                layout.addWidget(self.canvas)
                self.canvas.draw()
                self.toolbar = NavigationToolbar(self.canvas, self.ui.mplwindow, coordinates=True)
                layout.addWidget(self.toolbar)

                # Periodogram
                scaling_value = self.ui.ScalingcomboBox.currentText()
                window_value = self.ui.WindowcomboBox.currentText()

                f, Pxx = periodogram(self.signal, fs=self.data.fs, scaling=scaling_value,
                                     window=window_value)
                Pxx_dB = 10 * np.log10(Pxx)  # Convert power to dB

                # Plot within Periodogramlabel using a FigureCanvas
                fig = Figure(figsize(12, 12))
                ax = fig.add_subplot(122)
                ax.plot(f, Pxx_dB)
                ax.set_title(f'Periodogram of the Signal: Window Type = {window_value}, Scaling = {scaling_value}'
                          )
                ax.set_xlabel('Frequency (Hz)')
                if scaling_value == 'spectrum':
                    ax.set_ylabel('Power(dB/Hz)')
                else:
                    ax.set_ylabel('Power/Frequency (dB/Hz)')
                # for tick in ax.xaxis.get_major_ticks():
                #     tick.label.set_fontsize(18)  # Adjust the fontsize for x ticks
                #
                # for tick in ax.yaxis.get_major_ticks():
                #     tick.label.set_fontsize(18)  # Adjust the fontsize for y ticks
                # Embed the plot in the QLabel (Periodogramlabel)
                self.canvas = FigureCanvas(fig)
                layout = self.ui.mplwindow.layout()
                layout.addWidget(self.canvas)
                self.canvas.draw()
                self.ui.verticalLayout_5.addWidget(self.toolbar)



            #     # Button to plot is now clickable
            # self.ui.pbPlot.setEnabled(True)
            # self.ui.pbFFT.setEnabled(True)
            # self.ui.pbPeriodogram.setEnabled(True)
            # self.ui.pbPlot.setEnabled(True)


        except Exception as e:

            print(f"Error loading data: {e}")




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


