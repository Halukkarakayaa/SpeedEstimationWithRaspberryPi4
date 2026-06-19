"""
Step 1 & 2: YOLOv8n Vehicle Detection + ByteTrack + EMA Filtering
Mode: Pre-recorded Video File Mode (Raspberry Pi Optimized)
"""

import cv2
import time
from ultralytics import YOLO
import math
import numpy as np
from collections import deque


# ─────────────────────────────────────────────
# CONFIGURATION — AYARLAR
# ─────────────────────────────────────────────

# --- VİDEO DOSYASI AYARLARI ---
# --- CANLI KAMERA AYARLARI ---
CAMERA_INDEX = 0                         # USB veya Pi Kamerası
CAPTURE_WIDTH = 1280                     # Kamera genişliği
CAPTURE_HEIGHT = 720                     # Kamera yüksekliği
SAVE_OUTPUT_VIDEO = True                 # İşlenmiş görüntüyü MP4 olarak kaydet
TARGET_VIDEO_PATH = "canli_kayit8.mp4" # Kaydedilecek dosyanın adı
HEADLESS_MODE = False                     # True: Monitörsüz çalışır, sadece kaydeder

INFER_SIZE     = 320      # YOLOv8 inference boyutu (Pi için ideal)
CONF_THRESHOLD = 0.1     # Minimum tespit güveni
DEVICE         = "cpu"    # Pi 4'te GPU yok

VEHICLE_CLASSES = [2, 3, 5, 7]
WINDOW_TITLE = "Vehicle Detection — Video Mode"
DETECT_EVERY_N = 1   


# ─────────────────────────────────────────────
# SPEED ESTIMATION CONFIGURATION (13x39 Metre)
# ─────────────────────────────────────────────
# Videonuzdaki 13x39 metrelik alanın köşe pikselleri
SOURCE_PTS = np.array([
    [116,183],   # Sol üst
    [372,150],   # Sağ üst
    [970,388],   # Sağ alt 
    [484,575]     # Sol alt
], dtype=np.float32)

# Referans alanın GERÇEK boyutları (Metre)
DEST_PTS = np.array([
    [0, 0],       # Sol üst 
    [10, 0],      # Sağ üst (Genişlik: 25m)
    [10, 17],     # Sağ alt (Genişlik: 25m, Uzunluk: 39m)
    [0, 17]       # Sol alt (Uzunluk: 39m)
], dtype=np.float32)




MATRIX = cv2.getPerspectiveTransform(SOURCE_PTS, DEST_PTS)

# Geçmiş konum ve zaman verilerini tutacağımız bellekler
vehicle_history = {} 
vehicle_speeds = {}  

# ─────────────────────────────────────────────
# FILTERING & TRACKING CONFIGURATION
# ─────────────────────────────────────────────
EMA_ALPHA = 0.6  
ema_states = {}  
smoothed_boxes_to_draw = [] 

#locked_classes = {}       # Araçların ilk tespit edilen sınıflarını kilitler
lost_tracks = {}          # Araçların kaç karedir kayıp olduğunu sayar
MAX_LOST_FRAMES = 15      # Aracı silmeden önce beklenecek maksimum kare sayısı


# def get_color(track_id):
#     # ID'ye göre benzersiz ve sabit bir renk (BGR) üretir
#     return (int((track_id * 83) % 255), int((track_id * 137) % 255), int((track_id * 211) % 255))
# Sınıflara göre sabit renk tanımları (BGR formatında)
# Siyah ve beyaz zeminlerde maksimum görünürlük sağlar
# Siyah ve beyaz zeminlerde maksimum görünürlük sağlar (BGR Formatı)
CLASS_COLORS = {
    2: (255, 200, 0),     # Araba (Car) -> Mavi
    3: (0, 255, 0),     # Motosiklet (Motorcycle) -> Yeşil
    5: (255, 0, 255),   # Otobüs (Bus) -> Pembe / Macenta
    7: (0, 255, 255)    # Kamyon/Tır (Truck) -> Sarı
}
# ─────────────────────────────────────────────
# LOAD MODEL & VIDEO
# ─────────────────────────────────────────────

print("[INFO] YOLOv8n modeli yükleniyor...")
model = YOLO("yolov8n_ncnn_model_320", task="detect")
print("[INFO] Model yüklendi. Video dosyası açılıyor...")

# # Kamerayı değil, videoyu açıyoruz
# cap = cv2.VideoCapture(SOURCE_VIDEO_PATH)

# if not cap.isOpened():
#     raise RuntimeError(f"[ERROR] Video dosyası açılamadı: {SOURCE_VIDEO_PATH}")
# # Videonun kendi özelliklerini okuyoruz
# video_fps = cap.get(cv2.CAP_PROP_FPS)
# if video_fps == 0 or math.isnan(video_fps):
#      video_fps = 4.5 # Eğer videodan okunamazsa varsayılan

# video_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("[INFO] Canlı kamera başlatılıyor...")
cap = cv2.VideoCapture(CAMERA_INDEX)

# Kamerayı mecburi 720p ve hızlı okuma formatına alıyoruz
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

if not cap.isOpened():
    raise RuntimeError(f"[ERROR] {CAMERA_INDEX} indexli kamera açılamadı! Bağlantıyı kontrol et.")

video_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Pi'nin tahmini işlem hızı (Kaydedilen videonun çok hızlı/ağır olmaması için)
video_fps = 4.5 
scale_factor = video_height / 720.0

print(f"[INFO] Kamera bağlandı: {video_width}x{video_height} | Kayıt FPS: {video_fps}")




# scale_factor = video_height / 720.0

# print(f"[INFO] Video yüklendi: {video_width}x{video_height} @ {video_fps} FPS")

# Video Kaydedici (VideoWriter) Ayarları
out = None
if SAVE_OUTPUT_VIDEO:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(TARGET_VIDEO_PATH, fourcc, video_fps, (video_width, video_height))
    print(f"[INFO] Çıktı şu dosyaya kaydedilecek: {TARGET_VIDEO_PATH}")

# Performans ölçümü için değişkenler
processing_fps = 0.0 
processed_frame_count = 0
fps_update_interval = 0.5 
last_fps_time = time.time()

frame_counter = 0

# COLOUR & FONT CONSTANTS
COLOR_BOX   = (0, 255, 0)
COLOR_LABEL = (0, 255, 0)
COLOR_FPS   = (0, 200, 255)
COLOR_BG    = (0, 0, 0)
FONT        = cv2.FONT_HERSHEY_SIMPLEX
BOX_THICKNESS = max(2, int(2 * scale_factor))

def draw_label(frame, text, x, y, base_scale=0.55, color=COLOR_LABEL, base_thickness=1):
    # YENİ: Ölçek çarpanını uygula
    dynamic_scale = base_scale * scale_factor
    dynamic_thickness = max(1, int(base_thickness * scale_factor))
    
    (tw, th), baseline = cv2.getTextSize(text, FONT, dynamic_scale, dynamic_thickness)
    cv2.rectangle(frame, (x, y - th - baseline - int(2 * scale_factor)), (x + tw + int(4 * scale_factor), y + baseline), COLOR_BG, cv2.FILLED)
    cv2.putText(frame, text, (x + int(2 * scale_factor), y - int(2 * scale_factor)), FONT, dynamic_scale, color, dynamic_thickness, cv2.LINE_AA)

# print("[INFO] İşlem başladı. Görüntüyü izlerken çıkmak için klavyeden 'q' tuşuna basın.")
# cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
if not HEADLESS_MODE:
    print("[INFO] İşlem başladı. Çıkmak için klavyeden 'q' tuşuna basın.")
    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
else:
    print("[INFO] Headless (Ekransız) Mod aktif. Görüntü sadece dosyaya kaydediliyor...")
    print("[INFO] Durdurmak için klavyeden CTRL+C tuşlarına basın.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[INFO] Videonun sonuna gelindi.")
        break 

    frame_counter += 1

    # Videonun içindeki sanal zaman
    current_video_time = frame_counter / video_fps

    if frame_counter % DETECT_EVERY_N == 0:
        
        results = model.track(
            source=frame, imgsz=INFER_SIZE, conf=CONF_THRESHOLD,
            classes=VEHICLE_CLASSES, verbose=False, device=DEVICE,
            tracker="bytetrack.yaml", persist=True
        )
        last_result = results[0]
        smoothed_boxes_to_draw = [] 

        if last_result.boxes is not None and last_result.boxes.id is not None:
            boxes = last_result.boxes.xyxy.cpu().numpy()
            confs = last_result.boxes.conf.cpu().numpy()
            clss = last_result.boxes.cls.cpu().numpy()
            track_ids = last_result.boxes.id.int().cpu().numpy()

            current_ids_in_frame = set()

            for box, conf, cls, track_id in zip(boxes, confs, clss, track_ids):
                # Aracı ekranın neresinde olursa olsun genel takibe ekliyoruz (Geç algılamayı önler)
                current_ids_in_frame.add(track_id)
                x1, y1, x2, y2 = box

                # EMA Filtresi ile Kutu Yumuşatma
                if track_id in ema_states:
                    ex1, ey1, ex2, ey2 = ema_states[track_id]
                    nx1 = EMA_ALPHA * x1 + (1 - EMA_ALPHA) * ex1
                    ny1 = EMA_ALPHA * y1 + (1 - EMA_ALPHA) * ey1
                    nx2 = EMA_ALPHA * x2 + (1 - EMA_ALPHA) * ex2
                    ny2 = EMA_ALPHA * y2 + (1 - EMA_ALPHA) * ey2
                    ema_states[track_id] = [nx1, ny1, nx2, ny2]
                else:
                    ema_states[track_id] = [x1, y1, x2, y2]

                smooth_x1, smooth_y1, smooth_x2, smooth_y2 = ema_states[track_id]
                bottom_center_x = (smooth_x1 + smooth_x2) / 2
                bottom_center_y = smooth_y2

                # # YENİ: Sınıf Kilitleme Mekanizması
                # if track_id not in locked_classes:
                #     locked_classes[track_id] = model.names[int(cls)] # İlk kararı kilitle
                # class_name = locked_classes[track_id] # Her zaman kilitli olanı kullan
                
                # YENİ: Araç ekranda görüldüğü için kayıp sayacını sıfırla
                lost_tracks[track_id] = 0
                
                # --- YENİ EKLENEN ADIM: ARAÇ ALANIN İÇİNDE Mİ? ---
                is_inside = cv2.pointPolygonTest(SOURCE_PTS, (float(bottom_center_x), float(bottom_center_y)), False)

                if is_inside >= 0:
                    # ARAÇ ALANIN İÇİNDE (Veya çizginin tam üstünde)
                    # Gerçek dünya koordinatlarına (Metre) çevir
                    pt = np.array([[[bottom_center_x, bottom_center_y]]], dtype=np.float32)
                    transformed_pt = cv2.perspectiveTransform(pt, MATRIX)
                    real_x, real_y = transformed_pt[0][0] 

                    # if track_id not in vehicle_history:
                    #     vehicle_history[track_id] = deque(maxlen=15)
                    #     vehicle_speeds[track_id] = 0.0

                    # # Güncel konumu ve VİDEO ZAMANINI hafızaya ekle
                    # vehicle_history[track_id].append((current_video_time, real_x, real_y))
                    # history = vehicle_history[track_id]

                    # # Hesaplama için yeterli veri birikmişse
                    # if len(history) > 3:
                    #     old_video_time, old_x, old_y = history[0]
                    #     time_diff = current_video_time - old_video_time
                        
                    #     # Videoda en az 0.3 saniyelik bir ilerleme olduysa
                    #     if time_diff > 0.3:
                    #         dist_meters = math.sqrt((real_x - old_x)**2 + (real_y - old_y)**2)
                    #         speed_ms = dist_meters / time_diff
                    #         vehicle_speeds[track_id] = speed_ms * 3.6 # km/h'a çevir

                    # current_speed = vehicle_speeds[track_id]
                    #class_name = model.names[int(cls)]
                    if track_id not in vehicle_history:
                        # YENİ: Hafızayı 15 kareden 45 kareye çıkarıyoruz (Yaklaşık 1.5 saniye)
                        vehicle_history[track_id] = deque(maxlen=45) 
                        vehicle_speeds[track_id] = 0.0

                    # Güncel konumu ve VİDEO ZAMANINI hafızaya ekle
                    vehicle_history[track_id].append((current_video_time, real_x, real_y))
                    history = vehicle_history[track_id]

                    # YENİ: Hesaplama yapmak için kameranın aracı en az 15 kare görmesini bekle
                    # Hızlı Tepki: 15 kare beklemiyoruz, 3. kareden itibaren hesaplamaya başlıyoruz
                    if len(history) > 3:
                        old_video_time, old_x, old_y = history[0]
                        time_diff = current_video_time - old_video_time 
                        
                        # Sıfıra bölünmeyi engellemek için ufak bir zaman farkı (yaklaşık 0.1 sn) yeterli
                        if time_diff > 0.1:
                            dist_meters = math.sqrt((real_x - old_x)**2 + (real_y - old_y)**2)
                            
                            speed_ms = dist_meters / time_diff
                            calculated_speed = speed_ms * 3.6
                            
                            # KURAL: 5 km/h altı titreme kabul edilir, üstü hıza yansıtılır
                            if calculated_speed < 5.0:
                                vehicle_speeds[track_id] = 0.0
                            else:
                                prev_speed = vehicle_speeds[track_id]
                                if prev_speed == 0.0:
                                    vehicle_speeds[track_id] = calculated_speed
                                else:
                                    # Yeni hıza daha çabuk tepki vermesi için %50-%50 oranını kullanıyoruz
                                    vehicle_speeds[track_id] = (prev_speed * 0.5) + (calculated_speed * 0.5)
                    
                    current_speed = vehicle_speeds[track_id]
                    
                    # SADECE alanın içindeki araçları çizim listesine ekliyoruz
                    smoothed_boxes_to_draw.append((ema_states[track_id], track_id, int(cls), float(conf), current_speed))
                
                else:
                    # ARAÇ ALANIN DIŞINDA
                    # Çizim yapılmaz. Eğer araç daha önce alana girip şimdi çıktıysa hız geçmişini sıfırla.
                    if track_id in vehicle_history:
                        del vehicle_history[track_id]
                        if track_id in vehicle_speeds:
                            del vehicle_speeds[track_id]

            # Ekranda tamamen kaybolan araçları temizle
            # YENİ: Ekranda kaybolan araçları HEMAN SİLME, sayacı artır.
            keys_to_update = [tid for tid in ema_states if tid not in current_ids_in_frame]
            for tid in keys_to_update:
                lost_tracks[tid] = lost_tracks.get(tid, 0) + 1
                
                # Sadece sabır sınırını (15 kare) aşanları kalıcı olarak sil
                if lost_tracks[tid] > MAX_LOST_FRAMES:
                    del ema_states[tid]
                    if tid in vehicle_history: del vehicle_history[tid]
                    if tid in vehicle_speeds: del vehicle_speeds[tid]
                    #if tid in locked_classes: del locked_classes[tid]
                    del lost_tracks[tid]
        else:
            ema_states.clear()

    # --- Görselleştirme ---
    # Hedef Alanı Kırmızı ile Çizme
    pts = SOURCE_PTS.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(frame, [pts], isClosed=True, color=(0, 0, 255), thickness=2)

    # for s_box, track_id, confidence, speed in smoothed_boxes_to_draw:
    #     x1, y1, x2, y2 = map(int, s_box)
    #     box_color = get_color(track_id)
    #     cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, BOX_THICKNESS)
    #     label_text = f"#{track_id} | {speed:.1f} km/h"
    #     draw_label(frame, label_text, x1, y2 + int(20 * scale_factor), base_scale=0.5, color=box_color)

    for s_box, track_id, cls_id, confidence, speed in smoothed_boxes_to_draw:
        x1, y1, x2, y2 = map(int, s_box)
        
        # Aracın sınıfına göre belirlenmiş rengi çek (Listede yoksa Beyaz yap)
        box_color = CLASS_COLORS.get(cls_id, (255, 255, 255))
        
        # Kutuyu sınıfa özel sabit renkle çiz
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, BOX_THICKNESS)
        
        # Ekrana yazdırılacak metin (Artık 2, 7 değil; 1, 2, 3 gibi benzersiz ID'ler yazacak)
        label_text = f"#{track_id} | {speed:.1f} km/h"
        
        # Yazı arkaplanını ve kutuyu aynı parlak renkle çiz
        draw_label(frame, label_text, x1, y2 + int(20 * scale_factor), base_scale=0.5, color=box_color)

    # İşlemci Hızını (FPS) Hesapla
    processed_frame_count += 1
    now = time.time()
    elapsed = now - last_fps_time

    if elapsed >= fps_update_interval:
        processing_fps = processed_frame_count / elapsed
        processed_frame_count = 0
        last_fps_time = now

    draw_label(frame, f"Islem FPS: {processing_fps:.1f}", x=int(10*scale_factor), y=int(30*scale_factor), base_scale=0.8, color=COLOR_FPS, base_thickness=2)
    draw_label(frame, f"Alandaki Araclar: {len(smoothed_boxes_to_draw)}", x=int(10*scale_factor), y=int(65*scale_factor), base_scale=0.6, color=(255, 220, 0), base_thickness=1)

    # Video dosyasına kaydet
    if SAVE_OUTPUT_VIDEO and out is not None:
        out.write(frame)

    # cv2.imshow(WINDOW_TITLE, frame)
    
    # # Videonun Pi sınırlarında maksimum hızda işlenmesi için (1ms bekleme)
    # if cv2.waitKey(1) & 0xFF == ord("q"):
    #     print("[INFO] İşlem kullanıcı tarafından durduruldu.")
    #     break
    
    # Eğer Headless mod kapalıysa (monitör varsa) ekranda göster
    if not HEADLESS_MODE:
        cv2.imshow(WINDOW_TITLE, frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("[INFO] İşlem kullanıcı tarafından (Q) durduruldu.")
            break
    

cap.release()

if SAVE_OUTPUT_VIDEO and out is not None:
    out.release()
cv2.destroyAllWindows()
print("[INFO] İşlem başarıyla tamamlandı. Dosyalar kapatıldı.")