# import sys
# from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
# from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
# import matplotlib.pyplot as plt
# import numpy as np

# class MainWindow(QMainWindow):
#     def __init__(self):
#         super().__init__()
#         self.setWindowTitle("Matplotlib in PyQt5 Example")
#         self.setGeometry(100, 100, 800, 600)

#         # Tạo một widget chính để chứa biểu đồ
#         central_widget = QWidget()
#         self.setCentralWidget(central_widget)

#         # Tạo một layout dọc để chứa FigureCanvas
#         layout = QVBoxLayout(central_widget)

#         # Tạo một Figure và FigureCanvas
#         self.figure = plt.Figure(figsize=(6, 4))
#         self.canvas = FigureCanvas(self.figure)
#         layout.addWidget(self.canvas)

#         # Vẽ biểu đồ mẫu
#         self.plot_sample_data()

#     def plot_sample_data(self):
#         ax = self.figure.add_subplot(111)
#         x = np.linspace(0, 10, 100)
#         y = np.sin(x)
#         ax.plot(x, y)
#         ax.set_title('Sin Wave')
#         ax.set_xlabel('X axis')
#         ax.set_ylabel('Y axis')
#         self.canvas.draw()

# if __name__ == '__main__':
#     app = QApplication(sys.argv)
#     window = MainWindow()
#     window.show()
#     sys.exit(app.exec_())


import sys
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel
from PyQt5.QtGui import QPixmap
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QApplication, QMainWindow, QFrame, QLabel, QVBoxLayout,QFileDialog,QListWidgetItem,QScrollArea
from PyQt5.QtWidgets import QApplication, QMainWindow, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget
from PyQt5.QtCore import Qt

import io 
from PyQt5.QtCore import QFile
from PyQt5.uic import loadUi
import pandas as pd 
from torchvision import transforms
from ultralytics import YOLO
import geopandas as gpd
import shapely.geometry
from PIL import Image
from matplotlib.colors import Normalize
from matplotlib.cm import get_cmap
import numpy as np 


model = YOLO("best.pt")  
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

def predict(image_path):
    img = Image.open(image_path)
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    transform = transforms.Compose([
        transforms.Resize((128, 128)), 
        transforms.ToTensor(),          
    ])
    img = transform(img)
    results = model(img)
    return results

class MyMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Load UI từ file .ui
        ui_file = QFile("untitled.ui")
        ui_file.open(QFile.ReadOnly)
        loadUi(ui_file, self)
        ui_file.close()

        self.setWindowTitle("SHIP")

        # Tạo một Figure và vẽ biểu đồ
        self.figure = plt.Figure(figsize=(6, 4))

        # Chuyển đổi Figure thành QPixmap
        pixmap = self.figure_to_pixmap(self.figure)

        # Hiển thị QPixmap trên QLabel
        self.label.setPixmap(pixmap)
        
        self.pushButton.clicked.connect(self.openfile)
        self.pushButton_2.clicked.connect(self.close)
        self.comboBox.currentTextChanged.connect(self.combobox_click)


    # def plot_sample_data(self):
    #     ax = self.figure.add_subplot(111)
    #     x = [1, 2, 3, 4, 5]
    #     y = [2, 3, 5, 7, 11]
    #     ax.plot(x, y)
    #     ax.set_title('Sample Plot')
    #     ax.set_xlabel('X axis')
    #     ax.set_ylabel('Y axis')
    #     self.figure.tight_layout()
    def combobox_click(self):
        try:
            mmsi=self.comboBox.currentText()
            mmsi=int(mmsi)
            
            ship_static_info = self.df[self.df['MMSI'] == mmsi]
            # record = ship_static_info.iloc[0]
            
            static_columns = ['MMSI', 'IMO', 'Callsign', 'Name', 'Ship type', 'Cargo type', 'Width', 'Length',
                           'Type of position fixing device', 'Draught', 'Destination', 'ETA']
            mode_values = ship_static_info[static_columns].mode().iloc[0]
            self.tableWidget.setRowCount(len(static_columns))
            self.tableWidget.setColumnCount(2)  # 2 cột: Tên thông tin tĩnh và Giá trị thông tin tĩnh
            self.tableWidget.setHorizontalHeaderLabels(['Attribute', 'Value'])

            row = 0
            for row, attribute in enumerate(static_columns):
                attribute_item = QTableWidgetItem(attribute)
                self.tableWidget.setItem(row, 0, attribute_item)
                
                #value_item = QTableWidgetItem(str(record[attribute]))
                value_item = QTableWidgetItem(str(mode_values[attribute]))
                
                self.tableWidget.setItem(row, 1, value_item) 
            self.tableWidget.resizeColumnsToContents()
            self.plot_ais_data(self.df,mmsi)
            
        except:
            pass
    def openfile(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_name, _ = QFileDialog.getOpenFileName(self, "Open CSV File", "", "CSV Files (*.csv);;All Files (*)", options=options)
        
        if file_name:
           self.df = pd.read_csv(file_name)
           #self.plot_ais_data(self.df, self.mmsi)
           unique_mmsi = self.df['MMSI'].unique().astype(str)
           self.comboBox.clear()
           self.comboBox.addItems(unique_mmsi)
           self.label_4.setText(file_name)
    def close(self):
        exit()
    def plot_ais_data(self, df, mmsi):
        ship = df[df['MMSI'] == mmsi]
        latitude = list(ship['Latitude'])
        longitude = list(ship['Longitude'])
        cog = list(ship['COG'])
        ship['# Timestamp'] = pd.to_datetime(ship['# Timestamp'], format='%d/%m/%Y %H:%M:%S')
        ship['Timestamp_Num'] = ship['# Timestamp'].astype('int64') / 10**9
        timestamp = list(ship['Timestamp_Num'])
        combined_data = list(zip(latitude, longitude, timestamp, cog))
        # Sắp xếp theo timestamp (index 2 của tuple)
        sorted_data = sorted(combined_data, key=lambda x: x[2])

        # Rã zip để lấy lại các danh sách đã sắp xếp
        latitude, longitude, timestamp, cog = zip(*sorted_data)
        latitude = list(latitude)
        longitude = list(longitude)
        timestamp = list(timestamp)
        results=self.PREDICT(latitude,longitude,timestamp)
        for result in results:
            # Get the names of the classes
            class_names = result.names
            # Get the probabilities from the Probs object
            probs = result.probs.cpu().numpy()  # convert to numpy array if needed
            top1_class_index = probs.top1
            top1_confidence = probs.top1conf
            
        self.tableWidget_2.setRowCount(3)  # Số hàng
        self.tableWidget_2.setColumnCount(2)  # Số cột
        self.tableWidget_2.setHorizontalHeaderLabels(['Class Name', 'Confidence'])
        for result in results:
            if result.probs is not None:
                # Lấy các tên lớp
                class_names = result.names
                
                # Lấy các chỉ số của top 5 lớp và xác suất tương ứng từ đối tượng probs
                top5_class_indices = result.probs.top5
                top5_confidences = result.probs.top5conf.cpu().numpy()  # Chuyển đổi thành numpy array nếu cần thiết
                
                # In ra top 1, top 2 và top 3 lớp, xác suất và tên
                for row in range(len(top5_class_indices )):
                    class_index = top5_class_indices [row]
                    class_name=class_names[class_index]
                    confidence = top5_confidences[row]

                    item_class_name = QTableWidgetItem(str(class_name))
                    item_confidence = QTableWidgetItem(str(f"{confidence:.5f}"))

                    self.tableWidget_2.setItem(row, 0, item_class_name)
                    self.tableWidget_2.setItem(row, 1, item_confidence)
            else:
                print("No classification probabilities found for this result.")
        self.tableWidget_2.resizeColumnsToContents()
                

        
        
        
        cog = list(cog)
        norm = Normalize(vmin=min(timestamp), vmax=max(timestamp))
        cmap = get_cmap('viridis')  # Chọn một colormap, ví dụ 'viridis'

        # Vẽ bản đồ thế giới
        fig, ax = plt.subplots(figsize=(10, 8))
        base = world.plot(ax=ax, color='lightblue', edgecolor='black')
        ax.plot(longitude, latitude, color='blue', alpha=0.7)
        #scatter = ax.scatter(longitude, latitude, c=timestamp, cmap=cmap, norm=norm, marker='o', alpha=0.6)
        ax.set_ylabel('Latitude')
        ax.set_xlabel('Longitude')
        ax.set_aspect('equal', adjustable='datalim')
        plt.xlim(min(longitude)-(max(longitude)-min(longitude))/5, max(longitude) + (max(longitude)-min(longitude))/5)
        plt.ylim(min(latitude) -(max(latitude)-min(latitude))/5, max(latitude) + (max(latitude)-min(latitude))/5)
        
        # plt.xlim(0, 20)
        # plt.ylim(50,70)
        ax.axis('on')
        ax.text(
            0.95, 0.95,  # tọa độ (x, y) của văn bản
            f'MMSI:{mmsi}\n{class_names[top1_class_index]}: {top1_confidence:.2f}', 
            horizontalalignment='right',  # căn chỉnh văn bản theo chiều ngang
            verticalalignment='top',  # căn chỉnh văn bản theo chiều dọc
            transform=ax.transAxes,  # sử dụng hệ tọa độ của biểu đồ
            fontsize=12,  # kích thước chữ
            bbox=dict(facecolor='white', alpha=0.5)  # nền trắng với độ trong suốt
        )
        # Vẽ mũi tên tại mỗi điểm thứ 20
        step = max(1, len(latitude) // 20)

        for j in range(0, len(latitude), step):
            lat = latitude[j]
            lon = longitude[j]
            angle = cog[j]
            dx = np.cos(np.radians(angle)) * 0.01 
            dy = np.sin(np.radians(angle)) * 0.01
            
            ax.arrow(lon, lat, dx, dy, color='red', head_width= 0.01, head_length=0.02)
        fig.tight_layout()
        pixmap = self.figure_to_pixmap(fig)
        self.label.setPixmap(pixmap)

    def figure_to_pixmap(self, figure):
        # Render the figure to a buffer
        buf = io.BytesIO()
        figure.savefig(buf, format='png')
        buf.seek(0)

        # Convert buffer to QPixmap
        image = QImage.fromData(buf.getvalue())
        pixmap = QPixmap.fromImage(image)
        return pixmap
    def PREDICT(self,latitude, longitude, timestamp):
        norm = Normalize(vmin=min(timestamp), vmax=max(timestamp))
        cmap = get_cmap('viridis') 
        plt.figure(figsize=(10, 8))
        plt.plot(latitude, longitude, color='black', alpha=0.5)
        plt.scatter(latitude, longitude, c=timestamp, cmap=cmap, norm=norm, marker='o', alpha=0.6)
        plt.ylabel('Longitude')
        plt.xlabel('Latitude')
        plt.gca().set_aspect('equal', adjustable='datalim')
        plt.xlim(min(latitude), max(latitude))
        plt.ylim(min(longitude), max(longitude))
        plt.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.savefig(f'temp.jpg', bbox_inches='tight', pad_inches=0, dpi=50)
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=50)
        buf.seek(0)
        pixmap = QPixmap()
        pixmap.loadFromData(buf.getvalue())
        
        original_width = pixmap.width()
        original_height = pixmap.height()
        scaled_pixmap = pixmap.scaled(original_width // 1.8, original_height // 1.5, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.label_5.setPixmap(scaled_pixmap)

        #self.label_5.setPixmap(pixmap)
        results=predict("temp.jpg")
        plt.close()
        return results 
if __name__ == '__main__':

    app = QApplication(sys.argv)
    mainWindow = MyMainWindow()
    mainWindow.show()
    sys.exit(app.exec_())
