import os
import re
import tempfile

from flask import Flask, render_template, request, redirect, url_for, send_file, flash
from fpdf import FPDF

import cv2
import boto3
from ultralytics import YOLO

app = Flask(__name__)
app.secret_key = "your_secret_key"  # 플라스크 세션/메시지 사용을 위해 필요합니다.
DATABASE = 'problems.db'

# UPLOAD_FOLDER: 크롭된 문제 이미지가 저장될 폴더 (static 폴더 내에 uploads 폴더)
UPLOAD_FOLDER = os.path.join("static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

from google.cloud import storage
import uuid

# GCS 클라이언트 초기화 (서비스 계정 JSON 파일 경로를 환경 변수로 설정)
def upload_image_to_gcs(local_file_path, bucket_name, destination_blob_name=None):
    client = storage.Client()  # 환경 변수 GOOGLE_APPLICATION_CREDENTIALS가 설정되어 있어야 함
    bucket = client.bucket(bucket_name)
    if destination_blob_name is None:
        destination_blob_name = f"uploads/{uuid.uuid4().hex}.jpg"
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(local_file_path)
    # 퍼블릭 접근을 허용하고 URL 리턴 (필요 시)
    blob.make_public()
    return blob.public_url


# AWS Rekognition 설정 (본인 AWS 키로 교체)
rekognition = boto3.client(
    'rekognition',
    aws_access_key_id='AKIA3CN75QU2QUS7XLFO',
    aws_secret_access_key='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',
    region_name='ap-northeast-2'
)

# YOLO 모델 로드 (모델 경로도 필요에 따라 수정)
model_path = os.path.join(os.path.dirname(__file__), "models", "yolov8_trained_model.pt")
print("모델 파일 경로:", model_path)
model = YOLO(model_path)
print("모델 로드 완료")

# DB 연결 함수
import os
import pymysql  # MySQL connector. PostgreSQL이면 psycopg2 사용

def get_db_connection():
    connection = pymysql.connect(
        unix_socket=os.environ.get('DB_HOST'),  # DB_HOST 환경 변수에 '/cloudsql/INSTANCE_CONNECTION_NAME' 값이 들어있어야 함
        user=os.environ.get('DB_USER'),
        password=os.environ.get('DB_PASS', ''),
        db=os.environ.get('DB_NAME'),
        charset='utf8mb4',
        cursorclass=pymysql.cursors.DictCursor,
        connect_timeout=20
    )
    return connection



#########################################
# 이미지 처리 및 DB에 문제 추가 함수
#########################################
def process_image(file, student_id):
    try:
        import tempfile, os, re
        # 1. 업로드된 파일을 임시 파일로 저장
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            file.save(tmp.name)
            temp_filename = tmp.name

        # 2. 원본 이미지 로드
        original_img = cv2.imread(temp_filename)
        if original_img is None:
            os.remove(temp_filename)
            print("이미지 로드 실패")
            return

        image_h, image_w = original_img.shape[:2]
        img = original_img.copy()

        # 3. YOLO 예측 실행
        results = model.predict(temp_filename, conf=0.05)
        print("YOLO 예측 결과:", results)
        q_boxes = []
        for result in results:
            for box in result.boxes.data:
                x1, y1, x2, y2, conf, cls = box.tolist()
                if int(cls) == 0:
                    q_box = (int(x1), int(y1), int(x2), int(y2))
                    q_boxes.append(q_box)
                    width = q_box[2] - q_box[0]
                    x1_new = q_box[0] + int(width * 0.1)
                    img[q_box[1]:q_box[3], x1_new:q_box[2]] = (255, 255, 255)

        # 4. 처리된 이미지 저장 (OCR 전용)
        processed_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
        cv2.imwrite(processed_file.name, img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        processed_file.close()

        with open(processed_file.name, "rb") as f:
            image_bytes = f.read()
        response = rekognition.detect_text(Image={'Bytes': image_bytes})
        print("OCR 응답:", response)
        ocr_results = []
        for item in response['TextDetections']:
            text = item['DetectedText']
            bbox = item['Geometry']['BoundingBox']
            x = int(bbox['Left'] * image_w)
            y = int(bbox['Top'] * image_h)
            w = int(bbox['Width'] * image_w)
            h = int(bbox['Height'] * image_h)
            ocr_results.append((text, x, y, w, h))

        # 5. 각 q_box에 대해 OCR 결과를 분석하고, DB에 문제 추가
        for q_box in q_boxes:
            q_x1, q_y1, q_x2, q_y2 = q_box
            min_dist = float('inf')
            best_match = None

            for text, x, y, w, h in ocr_results:
                center_x = x + (w // 2)
                center_y = y + (h // 2)
                if not (q_x1 <= center_x <= q_x2 and q_y1 <= center_y <= q_y2):
                    continue
                clean_text = re.sub(r'\D', '', text)
                if not clean_text.isdigit():
                    continue
                dist = (center_x - q_x1) ** 2 + (center_y - q_y1) ** 2
                if dist < min_dist:
                    min_dist = dist
                    best_match = clean_text

            # 로그를 통해 확인 (디버깅용)
            print(f"q_box: {q_box}, best_match: {best_match}")

            # 기본값을 설정 (원한다면)
            if not best_match:
                best_match = "0"

            page_number = "1"
            sorted_ocr = sorted(
                [ (text, x, y, w, h) for text, x, y, w, h in ocr_results if re.sub(r'\D','',text).isdigit() ],
                key=lambda item: item[2],
                reverse=True
            )
            if sorted_ocr:
                page_number = re.sub(r'\D','', sorted_ocr[0][0])
            
            # 6. 크롭 이미지 생성: q_box 영역을 원본 이미지에서 크롭
            crop = original_img[q_y1:q_y2, q_x1:q_x2]
            import uuid
            crop_filename = f"student_{student_id}_{uuid.uuid4().hex}.jpg"
            local_crop_path = os.path.join(UPLOAD_FOLDER, crop_filename)
            cv2.imwrite(local_crop_path, crop)
            
            # 7. GCS 업로드: local_crop_path를 GCS에 업로드 후 URL 반환
            # 수정: 환경 변수 키를 'GCS_BUCKET_NAME'로 변경
            bucket_name = os.environ.get('GCS_BUCKET_NAME')
            crop_url = upload_image_to_gcs(local_crop_path, bucket_name)
            
            # 8. DB에 문제 추가: image_path 컬럼에 crop_url 저장
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO problems 
                (student_id, page_number, problem_number, bbox_x1, bbox_y1, bbox_x2, bbox_y2, image_path)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, (student_id, page_number, best_match, q_x1, q_y1, q_x2, q_y2, crop_url))
            conn.commit()
            conn.close()
            
            os.remove(local_crop_path)

        # 9. 임시 파일 삭제
        os.remove(temp_filename)
        os.remove(processed_file.name)
    except Exception as e:
        print("process_image 에러:", e)





#########################################
# 기존 라우트: 학생 목록, 문제 리스트, 사진 업로드, 문제 삭제, 문제 상태 업데이트 등
#########################################

@app.route('/')
def student_list():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT id, name FROM students')
    students = cursor.fetchall()
    conn.close()
    return render_template('students.html', students=students)

@app.route('/student/<int:student_id>')
def problem_list(student_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM problems WHERE student_id = %s", (student_id,))
    problems = cursor.fetchall()
    problems = sorted(problems, key=lambda p: (
        int(p['page_number']) if p['page_number'].isdigit() else 0, 
        int(p['problem_number']) if p['problem_number'].isdigit() else 0
    ))
    cursor.execute("SELECT * FROM students WHERE id = %s", (student_id,))
    student_row = cursor.fetchone()
    conn.close()
    if student_row:
        student = dict(student_row)
    else:
        student = {}
    return render_template('problems.html', problems=problems, student_id=student_id, student=student)

@app.route('/update_problem_details', methods=['POST'])
def update_problem_details():
    problem_id = request.form.get('problem_id')
    new_page_number = request.form.get('page_number')
    new_problem_number = request.form.get('problem_number')
    if not problem_id:
        return "문제 ID가 필요합니다.", 400

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE problems 
        SET page_number = %s, problem_number = %s
        WHERE id = %s
    """, (new_page_number, new_problem_number, problem_id))
    conn.commit()
    conn.close()
    return "업데이트 완료"


@app.route('/delete_problem/<int:problem_id>')
def delete_problem(problem_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM problems WHERE id = %s", (problem_id,))
    conn.commit()
    conn.close()
    flash("문제가 삭제되었습니다.")
    return redirect(request.referrer)

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        student_id = request.form.get('student_id')
        if not student_id:
            flash("학생을 선택하세요.")
            return redirect(request.url)
        files = request.files.getlist('photos')
        if not files:
            flash("업로드할 사진이 없습니다.")
            return redirect(request.url)
        for file in files:
            if file:
                process_image(file, int(student_id))
        flash("사진 업로드 및 문제 분석 완료!")
        return redirect(url_for('problem_list', student_id=student_id))
    else:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT id, name FROM students')
        students = cursor.fetchall()
        conn.close()
        return render_template('upload.html', students=students)

@app.route('/update_problem', methods=['POST'])
def update_problem():
    problem_id = request.form['problem_id']
    status_field = request.form['status_field']  # X1, Star, X2, Insane, MakeTest
    value = int(request.form['value'])
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(f'UPDATE problems SET status_{status_field} = %s WHERE id = %s', (value, problem_id))
    conn.commit()
    conn.close()
    return 'OK'

@app.route('/update_section', methods=['POST'])
def update_section():
    problem_id = request.form['problem_id']
    field = request.form['field']  # 'major_section' 또는 'minor_section'
    value = request.form['value']
    if field not in ['major_section', 'minor_section']:
        return "Invalid field", 400
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(f'UPDATE problems SET {field} = %s WHERE id = %s', (value, problem_id))
    conn.commit()
    conn.close()
    return 'OK'

# View image 라우트
@app.route('/problem_image/<int:problem_id>')
def problem_image(problem_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM problems WHERE id = %s', (problem_id,))
    problem = cursor.fetchone()
    conn.close()
    if problem and problem['image_path'] and os.path.exists(problem['image_path']):
        return send_file(problem['image_path'])
    else:
        return "No image available for this problem."

#########################################
# PDF 생성: 필터 조건에 따라, 원본 크롭 이미지를 3~4문제씩 배치, 문제 이미지 왼쪽 위에 페이지 번호 작게 표시
#########################################
@app.route('/generate_pdf/<int:student_id>')
def generate_pdf(student_id):
    bottom_filters = {
        'x1': request.args.get('x1'),
        'star': request.args.get('star'),
        'x2': request.args.get('x2'),
        'insane': request.args.get('insane'),
        'make_test': request.args.get('make_test')
    }
    if any(bottom_filters.values()):
        conditions = []
        params = [student_id]
        # 각 필터 조건을 OR로 결합 (중복 없이 한 번씩만 반환)
        for key, val in bottom_filters.items():
            # 값이 존재하고 1로 체크되었다면 조건 추가 (예: '1')
            if val is not None and int(val) == 1:
                col = key.upper() if key != 'make_test' else 'MakeTest'
                conditions.append(f"status_{col} = %s")
                params.append(1)
        # 조건들을 OR로 결합하고, DISTINCT를 사용해 중복 제거
        condition_str = " OR ".join(conditions)
        query = f"SELECT DISTINCT * FROM problems WHERE student_id = %s AND ({condition_str})"
    else:
        query = '''
            SELECT DISTINCT * FROM problems WHERE student_id = %s
            AND (status_X1 = 1 OR status_STAR = 1 OR status_X2 = 1 OR status_INSANE = 1 OR status_MakeTest = 1)
        '''
        params = [student_id]
    query += ' ORDER BY CAST(page_number as INTEGER), CAST(problem_number as INTEGER)'


    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(query, tuple(params))
    problems = cursor.fetchall()
    conn.close()
    # PDF 생성 라우트 내에서, 기존의 문제 이미지 배치 부분을 아래와 같이 수정합니다.
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=10)

    # 여백 설정 (mm)
    x_margin = 10
    y_margin = 10
    vertical_spacing = 10  # 이미지 아래쪽 여백
    y = y_margin  # x는 고정 (세로 배치이므로)
    pdf.add_page()

    for prob in problems:
        # DB에서 크롭 이미지 경로 가져오기 (키가 없으면 None 처리)
        crop_path = prob["image_path"] if "image_path" in prob.keys() else None
        if crop_path and os.path.exists(crop_path):
            # 원본 크기 측정을 위해 크롭 이미지를 읽습니다.
            crop_img = cv2.imread(crop_path)
            if crop_img is not None:
                h_px, w_px = crop_img.shape[:2]
                # FPDF의 기본 단위는 mm입니다. 일반적으로 96 DPI 가정 시,
                # 1 픽셀 ≒ 25.4 / 96 mm (약 0.2646 mm)
                factor = 25.4 / 96.0
                scale = 1/4
                img_w_mm = w_px * factor * scale
                img_h_mm = h_px * factor * scale

                # 원본 크기 그대로 삽입 (resize 없이)
                pdf.image(crop_path, x=x_margin, y=y, w=img_w_mm, h=img_h_mm)
                # 이미지 왼쪽 위에 페이지 번호 (예: "21p")를 작게 표시
                pdf.set_xy(x_margin + 2, y + 2)
                pdf.set_font("Arial", size=8)
                pdf.cell(0, 0, txt=f"{prob['page_number']}p", border=0)
                pdf.set_font("Arial", size=10)
                # 다음 이미지의 y 좌표는 현재 이미지 높이 + vertical_spacing 만큼 증가
                y += img_h_mm + vertical_spacing
            else:
                # 이미지 로드 실패 시
                pdf.set_xy(x_margin, y)
                pdf.cell(0, 10, txt="(No image available)", border=1)
                y += 15
        else:
            pdf.set_xy(x_margin, y)
            pdf.cell(0, 10, txt="(No image available)", border=1)
            y += 15

        # 만약 y 좌표가 페이지 하단을 넘으면 새 페이지 추가
        if y > 250:  # 이 값은 페이지 여백에 맞게 조정
            pdf.add_page()
            y = y_margin

    pdf_path = f"student_{student_id}_filtered_problems.pdf"
    pdf.output(pdf_path)
    return send_file(pdf_path, as_attachment=True)
    

@app.route('/homework/<student_name>', methods=['GET', 'POST'])
def homework(student_name):
    conn = get_db_connection()
    cursor = conn.cursor()
    # 학생 이름으로 조회 (학생 이름이 유일하다는 전제)
    cursor.execute("SELECT * FROM students WHERE name = %s", (student_name,))
    student = cursor.fetchone()
    if student is None:
        conn.close()
        flash("해당 학생을 찾을 수 없습니다.")
        return redirect(url_for('student_list'))
    student = dict(student)  # sqlite3.Row -> dict 변환
    student_id = student['id']
    
    # POST 요청: 새로운 숙제 컨테이너 추가 (숙제 목표는 학생 테이블에서 관리)
    if request.method == 'POST':
        class_date = request.form['class_date']
        homework_completion_date = request.form.get('homework_completion_date', '')
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO homework_containers (student_id, class_date, homework_completion_date)
            VALUES (%s, %s, %s)
        """, (student_id, class_date, homework_completion_date))
        conn.commit()
        flash("숙제 컨테이너 추가 완료!")
        conn.close()
        return redirect(url_for('homework', student_name=student_name))
    
    conn = get_db_connection()
    cursor = conn.cursor()

    # 숙제 컨테이너 조회 (DictCursor를 사용하므로 fetchall()은 이미 dict의 리스트입니다)
    cursor.execute("SELECT * FROM homework_containers WHERE student_id = %s", (student_id,))
    containers = cursor.fetchall()

    # 각 컨테이너 내 숙제 항목 조회 및 완료 항목 계산
    for container in containers:
        cursor.execute("SELECT * FROM homework_items WHERE container_id = %s", (container['id'],))
        items = cursor.fetchall()
        container['items'] = items
        container['completed_count'] = sum(1 for item in items if item.get('completed'))

    # 전체 숙제 완료 수 (모든 컨테이너의 완료 항목 합산)
    total_completed = sum(container['completed_count'] for container in containers)
    conn.close()

    
    return render_template('homework.html', 
                           containers=containers, 
                           student=student, 
                           student_name=student_name, 
                           total_completed=total_completed)

@app.route('/add_homework_item/<int:container_id>', methods=['POST'])
def add_homework_item(container_id):
    content = request.form['content']
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO homework_items (container_id, content) VALUES (%s, %s)", (container_id, content))
    conn.commit()
    conn.close()
    return redirect(request.referrer)

@app.route('/delete_homework_item/<int:item_id>')
def delete_homework_item(item_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM homework_items WHERE id = %s", (item_id,))
    conn.commit()
    conn.close()
    return redirect(request.referrer)

@app.route('/delete_homework_container/<int:container_id>')
def delete_homework_container(container_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM homework_items WHERE container_id = %s", (container_id,))
    cursor.execute("DELETE FROM homework_containers WHERE id = %s", (container_id,))
    conn.commit()
    conn.close()
    return redirect(request.referrer)

@app.route('/update_homework_item', methods=['POST'])
def update_homework_item():
    item_id = request.form['item_id']
    completed = int(request.form['completed'])
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE homework_items SET completed = %s WHERE id = %s", (completed, item_id))
    conn.commit()
    conn.close()
    return 'OK'

@app.route('/update_homework_goal/<student_name>', methods=['POST'])
def update_homework_goal(student_name):
    new_goal = request.form['homework_goal']
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE students SET homework_goal = %s WHERE name = %s", (new_goal, student_name))
    conn.commit()
    conn.close()
    flash("숙제 목표 업데이트 완료!")
    return redirect(url_for('homework', student_name=student_name))

@app.route('/confirm_homework_item', methods=['POST'])
def confirm_homework_item():
    item_id = request.form['item_id']
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE homework_items SET confirmed_by_teacher = 1 WHERE id = %s", (item_id,))
    conn.commit()
    conn.close()
    return 'OK'

@app.route('/unconfirm_homework_item', methods=['POST'])
def unconfirm_homework_item():
    item_id = request.form['item_id']
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE homework_items SET completed = 0, confirmed_by_teacher = 0 WHERE id = %s", (item_id,))
    conn.commit()
    conn.close()
    return 'OK'

#########################################
# 학생 추가 라우트
#########################################
@app.route('/add_student', methods=['GET', 'POST'])
def add_student():
    if request.method == 'POST':
        name = request.form['name']
        progress = request.form.get('progress', '')
        student_class = request.form.get('class_name', '')
        homework_goal = request.form.get('homework_goal', 0)  # 학생 단위 숙제 목표
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("INSERT INTO students (name, progress, class_name, homework_goal) VALUES (%s, %s, %s, %s)", 
                     (name, progress, student_class, homework_goal))
        conn.commit()
        conn.close()
        flash("학생 추가 완료!")
        return redirect(url_for('student_list'))
    return render_template('add_student.html')


if __name__ == '__main__':
    app.run(debug=True)
