import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import pandas as pd
import time

class SmartPageReader:
    def __init__(self):
        self.image = None
        self.coordinates = {'circles': [], 'crosses': []}
        self.svm_model = None
        self.scaler = StandardScaler()
        self.current_shape = 'cross'  # Başlangıç şekli
    
    def capture_image(self):
        """Kameradan fotoğraf çek"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            raise Exception("Kamera açılamadı!")
        
        print("Fotoğraf çekmek için SPACE tuşuna basın")
        print("Çıkmak için ESC tuşuna basın")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            cv2.imshow('Camera', frame)
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC
                break
            elif key == 32:  # SPACE
                self.image = frame.copy()
                print("Fotoğraf çekildi!")
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        if self.image is None:
            raise Exception("Fotoğraf çekilemedi!")
        
        return self.image
    
    def mark_shapes(self):
        """Kullanıcının şekilleri işaretlemesine izin ver"""
        if self.image is None:
            raise Exception("Önce fotoğraf çekilmeli!")
        
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                if self.current_shape == 'circle':
                    cv2.circle(self.image, (x, y), 10, (0, 255, 0), 2)
                    self.coordinates['circles'].append((x, y))
                    print(f"Daire eklendi: ({x}, {y})")
                else:
                    cv2.line(self.image, (x-10, y), (x+10, y), (0, 0, 255), 2)
                    cv2.line(self.image, (x, y-10), (x, y+10), (0, 0, 255), 2)
                    self.coordinates['crosses'].append((x, y))
                    print(f"Artı eklendi: ({x}, {y})")
        
        window_name = 'Mark Shapes'
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, mouse_callback)
        
        print("\nŞekil işaretleme moduna hoş geldiniz!")
        print("'c' tuşu - Daire işaretleme modu")
        print("'x' tuşu - Artı işaretleme modu")
        print("'SPACE' tuşu - İşaretlemeyi tamamla ve kaydet")
        print("'ESC' tuşu - İptal")
        
        while True:
            # Mevcut modu göster
            img_with_text = self.image.copy()
            mode_text = f"Mod: {'Daire' if self.current_shape == 'circle' else 'Artı'}"
            cv2.putText(img_with_text, mode_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            cv2.imshow(window_name, img_with_text)
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC
                self.coordinates['circles'] = []
                self.coordinates['crosses'] = []
                print("İşlem iptal edildi!")
                break
            elif key == ord('c'):
                self.current_shape = 'circle'
                print("Daire işaretleme modu aktif")
            elif key == ord('x'):
                self.current_shape = 'cross'
                print("Artı işaretleme modu aktif")
            elif key == 32:  # SPACE
                if len(self.coordinates['circles']) == 0 and len(self.coordinates['crosses']) == 0:
                    print("Hiç şekil işaretlenmedi! İşaretleme yapmadan çıkılamaz.")
                    continue
                self.save_coordinates()
                print("İşaretleme tamamlandı ve veriler kaydedildi!")
                break
        
        cv2.destroyAllWindows()
    
    def save_coordinates(self):
        """Koordinatları CSV dosyasına kaydet"""
        data = []
        
        # Daireleri ekle
        for x, y in self.coordinates['circles']:
            data.append({'x': x, 'y': y, 'shape': 'circle'})
        
        # Artıları ekle
        for x, y in self.coordinates['crosses']:
            data.append({'x': x, 'y': y, 'shape': 'cross'})
        
        # DataFrame oluştur ve kaydet
        df = pd.DataFrame(data)
        df.to_csv('dataset.csv', index=False)
        print(f"\nVeriler dataset.csv dosyasına kaydedildi:")
        print(f"Toplam {len(self.coordinates['circles'])} daire ve {len(self.coordinates['crosses'])} artı işareti.")

    def train_model(self):
        """EV2MD modülü ile model eğitimi"""
        if len(self.coordinates['circles']) == 0 or len(self.coordinates['crosses']) == 0:
            raise Exception("Önce şekilleri işaretleyin!")

        # Veri noktalarını hazırla
        X = []  # koordinatlar
        y = []  # etiketler (0: daire, 1: artı)

        # Daireleri ekle
        for x, y_coord in self.coordinates['circles']:
            X.append([x, y_coord])
            y.append(0)

        # Artıları ekle
        for x, y_coord in self.coordinates['crosses']:
            X.append([x, y_coord])
            y.append(1)

        X = np.array(X)
        y = np.array(y)

        # Verileri ölçeklendir
        X_scaled = self.scaler.fit_transform(X)

        # SVM modelini eğit
        self.svm_model = SVC(kernel='rbf', C=1.0)
        self.svm_model.fit(X_scaled, y)

    def draw_decision_boundary(self):
        """Karar sınırını çiz ve sonucu kaydet"""
        if self.svm_model is None:
            raise Exception("Önce model eğitilmeli!")

        # Görüntü boyutlarını al
        h, w = self.image.shape[:2]
        
        # Daha yoğun bir ızgara oluştur (daha düzgün sınırlar için)
        xx, yy = np.meshgrid(np.linspace(0, w-1, 200),
                            np.linspace(0, h-1, 200))
        
        # Izgara noktalarını düzleştir
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        
        # Noktaları ölçeklendir
        grid_points_scaled = self.scaler.transform(grid_points)
        
        # Karar sınırını hesapla
        Z = self.svm_model.predict(grid_points_scaled)
        Z = Z.reshape(xx.shape)

        # Sonuç görüntüsünü hazırla
        result_image = self.image.copy()

        # Matplotlib ile karar sınırını çiz
        plt.figure(figsize=(12, 12))
        plt.contour(xx, yy, Z, levels=[0.5], colors='blue', linewidths=3)
        boundary = plt.gca().collections[0].get_paths()[0].vertices
        plt.close()

        # Sınırı görüntü üzerine çiz
        points = boundary.astype(np.int32)
        cv2.polylines(result_image, [points], False, (255, 0, 0), 2)

        # İşaretlenen noktaları tekrar çiz
        # Daireleri çiz
        for (x, y) in self.coordinates['circles']:
            cv2.circle(result_image, (x, y), 10, (0, 255, 0), 2)  # Yeşil daire
            cv2.circle(result_image, (x, y), 2, (0, 255, 0), -1)  # Merkez nokta

        # Artıları çiz
        for (x, y) in self.coordinates['crosses']:
            cv2.line(result_image, (x-10, y), (x+10, y), (0, 0, 255), 2)  # Yatay çizgi
            cv2.line(result_image, (x, y-10), (x, y+10), (0, 0, 255), 2)  # Dikey çizgi
            cv2.circle(result_image, (x, y), 2, (0, 0, 255), -1)  # Merkez nokta

        # Sonucu PNG olarak kaydet
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        output_filename = f'decision_boundary_{timestamp}.png'
        cv2.imwrite(output_filename, result_image)
        print(f"\nSonuç görüntüsü kaydedildi: {output_filename}")

        return result_image

def main():
    reader = SmartPageReader()
    
    try:
        # 1. Adım: Fotoğraf çek
        print("\nKamera açılıyor...")
        reader.capture_image()
        
        # 2. Adım: Şekilleri işaretle
        print("\nŞekilleri işaretleme modu başlıyor...")
        reader.mark_shapes()
        
        # 3. Adım: Model eğitimi ve karar sınırı çizimi
        print("\nModel eğitiliyor...")
        reader.train_model()
        
        print("\nKarar sınırı çiziliyor...")
        result = reader.draw_decision_boundary()
        
        # Sonucu göster
        cv2.imshow('Decision Boundary', result)
        print("\nKarar sınırı çizildi. Çıkmak için herhangi bir tuşa basın...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"Hata: {e}")
        return

if __name__ == "__main__":
    main() 