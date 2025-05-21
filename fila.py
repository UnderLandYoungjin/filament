import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque
import time
import os

# 결과 저장 디렉토리 생성 - 숫자 증가 로직 추가
base_dir = "measurement_results"
counter = 0
results_dir = base_dir

# 이미 존재하는 디렉토리면 넘버링 추가
while os.path.exists(results_dir):
    counter += 1
    results_dir = f"{base_dir}_{counter:02d}"  # 01, 02 등 두 자리 숫자 포맷

os.makedirs(results_dir, exist_ok=True)
print(f"결과가 '{results_dir}' 디렉토리에 저장됩니다.")

# 영상 파일을 읽기 위해 VideoCapture 객체 생성
video_up = cv2.VideoCapture('up.mp4')

# 동영상 정보 가져오기
fps = video_up.get(cv2.CAP_PROP_FPS)
width = int(video_up.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video_up.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(video_up.get(cv2.CAP_PROP_FRAME_COUNT))

# 초기 보정값: 최초 프레임에서 측정된 픽셀 거리를 실제 1.75 mm와 매칭
initial_distance_mm = 1.75  # mm 단위 초기 기준 거리
initial_pixel_distance = None  # 최초 측정 픽셀 거리 (아직 미정)
pixel_to_mm_ratio = None       # 픽셀 당 mm 변환 비율 (초기 프레임에서 설정)

# 필라멘트 직경 데이터를 저장할 리스트 초기화
diameters_data = []

# 이동 평균 필터 (잡음을 줄이기 위함)
moving_average_window = 5
previous_diameters = deque(maxlen=moving_average_window)

# 실시간 그래프를 위한 데이터
real_time_frames = deque(maxlen=100)
real_time_diameters = deque(maxlen=100)

# 측정 정확도 평가를 위한 변수
std_deviation = 0
min_diameter = float('inf')
max_diameter = 0

# 측정 통계
measurement_statistics = {
    'min': float('inf'),
    'max': 0,
    'mean': 0,
    'std': 0,
    'stable_periods': 0
}

# 색상 및 테마 상수
MAIN_COLOR = (0, 165, 255)  # 주황색
SECONDARY_COLOR = (0, 255, 255)  # 노란색
TEXT_COLOR = (255, 255, 255)  # 흰색
ERROR_COLOR = (0, 0, 255)  # 빨간색
SUCCESS_COLOR = (0, 255, 0)  # 녹색
DARK_BG = (40, 40, 40)  # 어두운 배경

# HUD 관련 상수
MARGIN = 10
INFO_BAR_HEIGHT = 40
GRAPH_HEIGHT = 150
GRAPH_WIDTH = 300

# HUD에 실시간 그래프 그리기 함수
def draw_graph(frame, data_x, data_y, x_pos, y_pos, width, height):
    # 그래프 배경
    cv2.rectangle(frame, (x_pos, y_pos), (x_pos + width, y_pos + height), DARK_BG, -1)
    cv2.rectangle(frame, (x_pos, y_pos), (x_pos + width, y_pos + height), MAIN_COLOR, 1)
    
    # 데이터가 충분하지 않으면 종료
    if len(data_y) < 2:
        return
    
    # 그래프 Y축 범위 설정 (1.5-2.0mm가 보통의 필라멘트 범위)
    y_min, y_max = 1.5, 2.0
    if data_y:
        actual_min = min(data_y)
        actual_max = max(data_y)
        # 실제 데이터 범위가 기본 범위를 벗어나면 조정
        if actual_min < y_min or actual_max > y_max:
            margin = (actual_max - actual_min) * 0.1
            y_min = max(1.0, actual_min - margin)
            y_max = min(3.0, actual_max + margin)
    
    # 눈금 그리기
    for i in range(5):
        y_tick = y_pos + int(height * (1 - i / 4))
        tick_value = y_min + (y_max - y_min) * (i / 4)
        cv2.line(frame, (x_pos, y_tick), (x_pos + 5, y_tick), TEXT_COLOR, 1)
        cv2.putText(frame, f"{tick_value:.2f}", (x_pos + 7, y_tick + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, TEXT_COLOR, 1)
    
    # 기준선 (1.75mm) 표시
    ref_y = y_pos + int(height * (1 - (1.75 - y_min) / (y_max - y_min)))
    cv2.line(frame, (x_pos, ref_y), (x_pos + width, ref_y), (0, 255, 0), 1, cv2.LINE_AA)
    
    # 그래프 그리기
    points = []
    for i in range(len(data_y)):
        norm_x = i / (len(data_y) - 1 or 1)
        norm_y = (data_y[i] - y_min) / (y_max - y_min)
        px = x_pos + int(norm_x * width)
        py = y_pos + int(height * (1 - norm_y))
        points.append((px, py))
    
    for i in range(1, len(points)):
        color = SUCCESS_COLOR if abs(data_y[i] - 1.75) < 0.05 else MAIN_COLOR
        cv2.line(frame, points[i-1], points[i], color, 2, cv2.LINE_AA)
    
    # 그래프 제목
    cv2.putText(frame, "Filament Diameter (mm)", (x_pos, y_pos - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, SECONDARY_COLOR, 1)

# 현재 시간 및 프레임 정보 표시 함수
def draw_info_bar(frame, frame_count, diameter, process_time):
    # 상단 정보 바 배경
    cv2.rectangle(frame, (0, 0), (frame.shape[1], INFO_BAR_HEIGHT), DARK_BG, -1)
    
    # 시간 정보
    current_time = time.strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, current_time, (MARGIN, 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1)
    
    # 프레임 정보
    frame_info = f"Frame: {frame_count}/{total_frames} ({frame_count/total_frames*100:.1f}%)"
    cv2.putText(frame, frame_info, (frame.shape[1] - 230, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1)
    
    # 중앙에 현재 직경 표시
    if diameter > 0:
        color = SUCCESS_COLOR if abs(diameter - 1.75) < 0.05 else SECONDARY_COLOR
        cv2.putText(frame, f"Diameter: {diameter:.3f} mm", (frame.shape[1]//2 - 80, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # 처리 시간
    cv2.putText(frame, f"Process: {process_time*1000:.1f} ms", (frame.shape[1] - 230, INFO_BAR_HEIGHT + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1)

# 측정 통계 표시 함수
def draw_statistics(frame, stats, x_pos, y_pos):
    # 통계 패널 배경
    panel_width = 200
    panel_height = 150
    cv2.rectangle(frame, (x_pos, y_pos), (x_pos + panel_width, y_pos + panel_height), DARK_BG, -1)
    cv2.rectangle(frame, (x_pos, y_pos), (x_pos + panel_width, y_pos + panel_height), MAIN_COLOR, 1)
    
    # 제목
    cv2.putText(frame, "Measurement Statistics", (x_pos + 10, y_pos + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, SECONDARY_COLOR, 1)
    
    # 통계 데이터
    cv2.putText(frame, f"Min: {stats['min']:.3f} mm", (x_pos + 10, y_pos + 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1)
    cv2.putText(frame, f"Max: {stats['max']:.3f} mm", (x_pos + 10, y_pos + 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1)
    cv2.putText(frame, f"Mean: {stats['mean']:.3f} mm", (x_pos + 10, y_pos + 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1)
    cv2.putText(frame, f"Std Dev: {stats['std']:.3f} mm", (x_pos + 10, y_pos + 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1)
    
    # 품질 지표 (허용 오차 내 안정 기간)
    stability = "Good" if stats['std'] < 0.02 else "Fair" if stats['std'] < 0.05 else "Poor"
    color = SUCCESS_COLOR if stability == "Good" else SECONDARY_COLOR if stability == "Fair" else ERROR_COLOR
    cv2.putText(frame, f"Stability: {stability}", (x_pos + 10, y_pos + 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

# 주기적 보정 함수 - 측정 안정성 향상
def recalibrate_pixel_to_mm(pixel_distances, current_mm, target_mm=1.75, weight=0.1):
    """매 N프레임마다 보정 비율을 조정하여 장기간 안정성 확보"""
    global pixel_to_mm_ratio
    
    if not pixel_distances or pixel_to_mm_ratio is None:
        return pixel_to_mm_ratio
    
    # 중앙값 기반 이상치 제거
    median_distance = np.median(pixel_distances)
    valid_distances = [d for d in pixel_distances if 0.8 * median_distance <= d <= 1.2 * median_distance]
    
    if not valid_distances:
        return pixel_to_mm_ratio
    
    avg_pixel_distance = np.mean(valid_distances)
    
    # 목표 직경에서 현재 직경까지의 오차
    error_ratio = target_mm / current_mm
    
    # 보정 비율 조정 (급격한 변화 방지를 위한 가중평균)
    new_ratio = target_mm / avg_pixel_distance
    adjusted_ratio = (1 - weight) * pixel_to_mm_ratio + weight * new_ratio
    
    # 보정 비율 변화가 너무 크지 않은지 확인
    if 0.9 < (adjusted_ratio / pixel_to_mm_ratio) < 1.1:
        return adjusted_ratio
    else:
        # 변화가 너무 크면 더 작은 가중치 사용
        return (1 - weight/2) * pixel_to_mm_ratio + (weight/2) * new_ratio

# 각 프레임에서 필라멘트의 직경을 측정하는 함수
def process_frame(frame, frame_count):
    global initial_pixel_distance, pixel_to_mm_ratio
    start_time = time.time()
    
    # 원본 이미지 저장
    original_frame = frame.copy()
    
    # 1. 영상 전처리: 흑백 변환 및 Gaussian 블러 적용
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 2. 캐니 엣지 검출
    edges = cv2.Canny(blurred, 50, 150)

    # 3. 서브픽셀 정확도를 위해 이미지 업샘플링 (배율 2 적용)
    edges = cv2.resize(edges, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # 4. 노이즈 감소 (Gaussian 필터 재적용)
    edges = cv2.GaussianBlur(edges, (3, 3), 0)

    # 5. 바운딩 박스 설정 (업샘플된 이미지 기준, 가운데 영역)
    height, width = edges.shape
    box_width = 120    # 바운딩 박스 너비 (픽셀)
    box_height = 240   # 바운딩 박스 높이 (픽셀)
    box_x_start = width // 2 - box_width // 2  # 가운데로 위치 변경
    box_x_end = box_x_start + box_width
    box_y_start = height // 2 - box_height // 2
    box_y_end = box_y_start + box_height
    roi = edges[box_y_start:box_y_end, box_x_start:box_x_end]

    # 6. ROI 내 여러 세로선에서 직경 측정
    num_lines = 7  # 측정 포인트 수
    step = roi.shape[1] // (num_lines + 1)
    diameters = []
    edge_points = []  # 엣지 포인트 저장을 위한 리스트

    # 원본 영상에 결과를 표시하기 위한 복사본 (HUD 추가)
    processed_image = original_frame.copy()
    
    # 바운딩 박스 표시
    box_start = (box_x_start // 2, box_y_start // 2)
    box_end = (box_x_end // 2, box_y_end // 2)
    cv2.rectangle(processed_image, box_start, box_end, MAIN_COLOR, 2)
    
    # 측정 영역 강조 효과
    overlay = processed_image.copy()
    cv2.rectangle(overlay, box_start, box_end, MAIN_COLOR, -1)
    processed_image = cv2.addWeighted(overlay, 0.2, processed_image, 0.8, 0)

    for i in range(1, num_lines + 1):
        x = i * step
        line_profile = roi[:, x]
        # 해당 열에서 캐니 엣지가 나타나는 위치(행 인덱스) 찾기
        edge_positions = np.where(line_profile > 0)[0]

        if len(edge_positions) >= 2:
            # 엣지 클러스터링 - 유사한 위치의 엣지들을 그룹화
            clusters = []
            current_cluster = [edge_positions[0]]
            
            for j in range(1, len(edge_positions)):
                if edge_positions[j] - edge_positions[j-1] < 5:  # 인접한 위치면 같은 클러스터
                    current_cluster.append(edge_positions[j])
                else:
                    clusters.append(current_cluster)
                    current_cluster = [edge_positions[j]]
            
            if current_cluster:
                clusters.append(current_cluster)
            
            # 각 클러스터의 평균 위치를 실제 엣지 위치로 사용
            if len(clusters) >= 2:
                top_cluster = clusters[0]
                bottom_cluster = clusters[-1]
                
                top_edge = int(np.mean(top_cluster)) + box_y_start
                bottom_edge = int(np.mean(bottom_cluster)) + box_y_start
                pixel_distance = bottom_edge - top_edge
                diameters.append(pixel_distance)
                
                # 엣지 포인트 저장
                x_original = int((box_x_start + x) / 2)
                top_edge_original = int(top_edge / 2)
                bottom_edge_original = int(bottom_edge / 2)
                edge_points.append((x_original, top_edge_original, bottom_edge_original))

    diameter_mm = 0
    if len(diameters) > 0:
        # 이상치 제거 (중앙값에서 크게 벗어나는 측정값 제외)
        median_diameter = np.median(diameters)
        filtered_diameters = [d for d in diameters if 0.8 * median_diameter <= d <= 1.2 * median_diameter]
        
        if filtered_diameters:
            # 여러 열에서 측정된 픽셀 거리의 평균 계산
            pixel_distance = np.mean(filtered_diameters)
            
            # 최초 프레임에서 보정: 측정된 픽셀 거리를 1.75 mm에 대응시킴
            if initial_pixel_distance is None:
                initial_pixel_distance = pixel_distance
                pixel_to_mm_ratio = initial_distance_mm / initial_pixel_distance
                print(f"초기 보정 완료: 픽셀-mm 비율 = {pixel_to_mm_ratio:.6f}")
            
            # 주기적 보정 (50프레임마다 수행)
            elif frame_count % 50 == 0 and frame_count > 0:
                # 현재 픽셀-mm 비율 저장
                old_ratio = pixel_to_mm_ratio
                # 현재 직경 계산 (mm)
                current_diameter = pixel_distance * pixel_to_mm_ratio
                # 이전 측정값 기준으로 비율 조정
                pixel_to_mm_ratio = recalibrate_pixel_to_mm(
                    [d for d in filtered_diameters], 
                    current_diameter, 
                    initial_distance_mm
                )
                # 변경 비율 출력
                change_percent = 100 * (pixel_to_mm_ratio - old_ratio) / old_ratio
                if abs(change_percent) > 0.1:
                    print(f"프레임 {frame_count}: 보정 비율 업데이트 {old_ratio:.6f} → {pixel_to_mm_ratio:.6f} ({change_percent:+.2f}%)")
            
            # 현재 프레임의 직경 계산 (mm)
            current_diameter = pixel_distance * pixel_to_mm_ratio
            
            # 이동 평균 필터 적용 (노이즈 완화)
            previous_diameters.append(current_diameter)
            diameter_mm = np.mean(previous_diameters)
            
            # 엣지 및 측정 표시 - 애니메이션 효과 추가
            for x_original, top_edge_original, bottom_edge_original in edge_points:
                # 펄싱 효과를 위한 시간 기반 알파 값
                alpha = 0.5 + 0.5 * np.sin(frame_count * 0.1)
                
                # 측정 라인 그리기
                cv2.line(
                    processed_image, 
                    (x_original, top_edge_original), 
                    (x_original, bottom_edge_original), 
                    MAIN_COLOR, 2
                )
                
                # 포인트 강조 (펄싱 효과)
                point_color = tuple([int(c * alpha + (1-alpha) * 255) for c in SECONDARY_COLOR])
                cv2.circle(processed_image, (x_original, top_edge_original), 4, point_color, -1)
                cv2.circle(processed_image, (x_original, bottom_edge_original), 4, point_color, -1)
                
                # 측정 거리 표시 라인
                mid_y = (top_edge_original + bottom_edge_original) // 2
                arrow_length = 10
                cv2.arrowedLine(
                    processed_image,
                    (x_original - arrow_length, top_edge_original), 
                    (x_original, top_edge_original),
                    SECONDARY_COLOR, 2, cv2.LINE_AA, tipLength=0.3
                )
                cv2.arrowedLine(
                    processed_image, 
                    (x_original - arrow_length, bottom_edge_original), 
                    (x_original, bottom_edge_original),
                    SECONDARY_COLOR, 2, cv2.LINE_AA, tipLength=0.3
                )
            
            # 실시간 그래프 데이터 업데이트
            real_time_frames.append(frame_count)
            real_time_diameters.append(diameter_mm)
            
            # 측정 통계 업데이트
            measurement_statistics['min'] = min(measurement_statistics['min'], diameter_mm)
            measurement_statistics['max'] = max(measurement_statistics['max'], diameter_mm)
            
            # 충분한 데이터가 수집됐을 때 평균과 표준편차 계산
            if len(diameters_data) > 0:
                all_diameters = [d['up_mm'] for d in diameters_data] + [diameter_mm]
                measurement_statistics['mean'] = np.mean(all_diameters)
                measurement_statistics['std'] = np.std(all_diameters)
    
    # 중앙 마커 표시 (십자선)
    center_x, center_y = width // 4, height // 4  # 업샘플링 고려
    marker_size = 20
    cv2.line(processed_image, (center_x - marker_size, center_y), (center_x + marker_size, center_y), 
             SECONDARY_COLOR, 1, cv2.LINE_AA)
    cv2.line(processed_image, (center_x, center_y - marker_size), (center_x, center_y + marker_size), 
             SECONDARY_COLOR, 1, cv2.LINE_AA)
    
    # 좌표계 그리드 (옵션)
    grid_spacing = 50
    for x in range(0, width, grid_spacing):
        cv2.line(processed_image, (x//2, 0), (x//2, height//2), DARK_BG, 1, cv2.LINE_AA)
    for y in range(0, height, grid_spacing):
        cv2.line(processed_image, (0, y//2), (width//2, y//2), DARK_BG, 1, cv2.LINE_AA)
    
    # HUD 요소 추가
    process_time = time.time() - start_time
    
    # 정보 바 추가
    draw_info_bar(processed_image, frame_count, diameter_mm, process_time)
    
    # 실시간 그래프 추가 - 오른쪽 중단으로 위치 변경
    draw_graph(processed_image, real_time_frames, real_time_diameters, 
               processed_image.shape[1] - GRAPH_WIDTH - MARGIN, 
               processed_image.shape[0] // 2 - GRAPH_HEIGHT - MARGIN, 
               GRAPH_WIDTH, GRAPH_HEIGHT)
    
    # 측정 통계 추가
    draw_statistics(processed_image, measurement_statistics, 
                   processed_image.shape[1] - 220, 
                   processed_image.shape[0] - GRAPH_HEIGHT - MARGIN)

    return diameter_mm, processed_image

# 결과 시각화를 위한 Matplotlib 그래프 초기화
plt.figure(figsize=(12, 8))
# 다크 배경 스타일을 제거하고 기본 밝은 배경 사용

# 결과 저장을 위한 비디오 인코더
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(os.path.join(results_dir, 'filament_measurement_result.avi'), 
                      fourcc, fps, (width, height))

# 각 프레임 단위로 처리
frame_count = 0
try:
    while True:
        ret_up, frame_up = video_up.read()
        if not ret_up:
            break

        # 진행 상황 표시
        progress = (frame_count / total_frames) * 100
        if frame_count % 10 == 0:
            print(f"\rProcessing: {progress:.1f}% complete", end="")
        
        dia_up, processed_frame = process_frame(frame_up, frame_count)
        
        if dia_up > 0:
            diameters_data.append({'frame': frame_count, 'up_mm': dia_up})

        # 결과 비디오에 프레임 저장
        out.write(processed_frame)
        
        # 화면에 표시 (크기 조정 옵션)
        display_frame = processed_frame
        cv2.imshow('Filament Diameter Measurement', display_frame)
        
        frame_count += 1
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):  # 's' 키를 누르면 현재 프레임 저장
            cv2.imwrite(os.path.join(results_dir, f'frame_{frame_count}.jpg'), processed_frame)
            print(f"\nFrame saved: frame_{frame_count}.jpg")
    
    print("\nProcessing complete!")

except KeyboardInterrupt:
    print("\nProcessing interrupted by user.")

finally:
    # 리소스 해제
    video_up.release()
    out.release()
    cv2.destroyAllWindows()

    # 결과 데이터를 CSV 파일로 저장
    if diameters_data:
        df = pd.DataFrame(diameters_data)
        df.to_csv(os.path.join(results_dir, 'filament_diameters_up.csv'), index=False)
        
        # 최종 결과 출력
        mean_diameter = df['up_mm'].mean()
        std_diameter = df['up_mm'].std()
        min_diameter = df['up_mm'].min()
        max_diameter = df['up_mm'].max()
        
        print(f"측정 완료!")
        print(f"총 {len(diameters_data)}개 프레임의 직경 데이터가 저장되었습니다.")
        print(f"평균 직경: {mean_diameter:.3f} mm (표준편차: {std_diameter:.3f} mm)")
        print(f"최소 직경: {min_diameter:.3f} mm, 최대 직경: {max_diameter:.3f} mm")
        
        # 최종 그래프 생성 및 저장 (흰색 배경으로 변경)
        plt.subplot(2, 1, 1)
        plt.plot(df['frame'], df['up_mm'], 'o-', color='#FF8000', alpha=0.7, markersize=3)  # 주황색 조정
        plt.axhline(y=1.75, color='green', linestyle='--', alpha=0.7, label='Target (1.75mm)')
        plt.axhline(y=mean_diameter, color='red', linestyle='-', alpha=0.5, label=f'Mean ({mean_diameter:.3f}mm)')
        plt.fill_between(df['frame'], mean_diameter - std_diameter, mean_diameter + std_diameter, 
                        color='red', alpha=0.1, label=f'±1σ ({std_diameter:.3f}mm)')
        plt.grid(True, alpha=0.3)
        plt.title('Filament Diameter Measurement Over Time', color='black')  # 제목 색상을 검은색으로
        plt.ylabel('Diameter (mm)', color='black')  # 라벨 색상을 검은색으로
        plt.legend(facecolor='white', edgecolor='gray')  # 범례 배경을 흰색으로
        plt.tick_params(colors='black')  # 눈금 색상을 검은색으로
        
        plt.subplot(2, 1, 2)
        plt.hist(df['up_mm'], bins=30, color='#4682B4', alpha=0.7, edgecolor='white')  # 스틸블루 색상
        plt.axvline(x=1.75, color='green', linestyle='--', alpha=0.7, label='Target (1.75mm)')
        plt.axvline(x=mean_diameter, color='red', linestyle='-', alpha=0.7, label=f'Mean ({mean_diameter:.3f}mm)')
        plt.grid(True, alpha=0.3)
        plt.title('Diameter Distribution', color='black')  # 제목 색상을 검은색으로
        plt.xlabel('Diameter (mm)', color='black')  # x축 라벨 색상을 검은색으로
        plt.ylabel('Frequency', color='black')  # y축 라벨 색상을 검은색으로
        plt.legend(facecolor='white', edgecolor='gray')  # 범례 배경을 흰색으로
        plt.tick_params(colors='black')  # 눈금 색상을 검은색으로
        
        # 그림 전체 스타일 조정
        plt.tight_layout()
        # 배경색 명시적으로 설정
        plt.gcf().patch.set_facecolor('white')
        for ax in plt.gcf().get_axes():
            ax.set_facecolor('white')  # 각 서브플롯의 배경색도 흰색으로 설정
        
        plt.savefig(os.path.join(results_dir, 'filament_measurement_results.png'), dpi=150)
        
        print(f"결과가 {results_dir} 디렉토리에 저장되었습니다.")
    else:
        print("측정된 직경 데이터가 없습니다.")
