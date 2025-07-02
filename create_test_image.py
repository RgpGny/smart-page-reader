import cv2
import numpy as np

def create_test_image():
    # Beyaz arka planlı görüntü oluştur
    image = np.ones((500, 500, 3), dtype=np.uint8) * 255
    
    # Artı işaretleri çiz
    crosses = [
        (100, 100), (150, 150), (200, 100),
        (100, 200), (150, 250), (200, 200)
    ]
    
    for (x, y) in crosses:
        # Yatay çizgi
        cv2.line(image, (x-10, y), (x+10, y), (0,0,0), 2)
        # Dikey çizgi
        cv2.line(image, (x, y-10), (x, y+10), (0,0,0), 2)
    
    # Daireler çiz
    circles = [
        (300, 100), (350, 150), (400, 100),
        (300, 200), (350, 250), (400, 200)
    ]
    
    for (x, y) in circles:
        cv2.circle(image, (x, y), 10, (0,0,0), 2)
    
    # Görüntüyü kaydet
    cv2.imwrite('input_image.jpg', image)
    
    print("Test görüntüsü oluşturuldu: input_image.jpg")

if __name__ == "__main__":
    create_test_image() 