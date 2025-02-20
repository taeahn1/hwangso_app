import os
import pymysql

# Cloud SQL(MySQL) 연결 설정을 위한 get_db_connection 함수
def get_db_connection():
    connection = pymysql.connect(
        host=os.environ.get('34.64.211.105'),       # 예: '127.0.0.1' 또는 Cloud SQL 인스턴스 IP/소켓 경로
        user=os.environ.get('root'),       # 예: 'your_mysql_user'
        password=os.environ.get('spongebob1'),   # 예: 'your_mysql_password'
        db=os.environ.get('mysql_hwangso'),         # 예: 'your_database'
        charset='utf8mb4',
        cursorclass=pymysql.cursors.DictCursor
    )
    return connection

# DB 스키마 생성 (MySQL에 맞게 일부 수정)
conn = get_db_connection()
c = conn.cursor()

# 학생 테이블 생성 (name, progress, class, homework_goal)
c.execute('''
CREATE TABLE IF NOT EXISTS students (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    progress TEXT,
    class_name TEXT,
    homework_goal INTEGER DEFAULT 0
)
''')

# 문제 테이블 생성 (image_path, major_section, minor_section 포함)
c.execute('''
CREATE TABLE IF NOT EXISTS problems (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id INTEGER,
    page_number TEXT,
    problem_number TEXT,
    bbox_x1 INTEGER,
    bbox_y1 INTEGER,
    bbox_x2 INTEGER,
    bbox_y2 INTEGER,
    status_X1 INTEGER DEFAULT 0,
    status_Star INTEGER DEFAULT 0,
    status_X2 INTEGER DEFAULT 0,
    status_Insane INTEGER DEFAULT 0,
    status_MakeTest INTEGER DEFAULT 0,
    image_path TEXT,
    major_section TEXT,
    minor_section TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(student_id) REFERENCES students(id)
)
''')

# 숙제 컨테이너 테이블 생성 (학생별 숙제 그룹: 수업 날짜, 숙제 완료 날짜)
c.execute('''
CREATE TABLE IF NOT EXISTS homework_containers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id INTEGER,
    class_date TEXT,
    homework_completion_date TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(student_id) REFERENCES students(id)
)
''')

# 숙제 항목 테이블 생성 (각 컨테이너 내 여러 숙제 항목: 내용, 완료 여부)
c.execute('''
CREATE TABLE IF NOT EXISTS homework_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    container_id INTEGER,
    content TEXT,
    completed INTEGER DEFAULT 0,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(container_id) REFERENCES homework_containers(id)
)
''')

conn.commit()
conn.close()
print("Database setup complete.")
