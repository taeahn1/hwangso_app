import sqlite3

conn = sqlite3.connect('problems.db')
c = conn.cursor()

# 예시: 학생 3명 추가
students = [("김황소",)]
c.executemany("INSERT INTO students (name) VALUES (?)", students)



conn.commit()
conn.close()
print("학생 데이터 추가 완료!")
