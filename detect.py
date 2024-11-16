import cv2

img = cv2.imread("coins.jpg")

#ลดขนาดภาพ
resize = cv2.resize(img, (1000, 900))

#ปรับภาพเพื่อนำเข้าbinary
gray = cv2.cvtColor(resize, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)

#แปลงภาพเป็นBinary
_, binary = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY_INV)

#ปิดขอบเหรียญ มีเหลี่ยมตรงภาพขวาล่าง กะแปลงbinary แล้วกลมไม่หมด
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

#หาขอบเหรีญกลมๆ
contour, hierarchy = cv2.findContours(closing.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# กรองเหรียญ
filtered_contours = [cont for cont in contour if cv2.contourArea(cont) > 500]

# วาดเหรียญและใส่ตัวเลข
for i, cont in enumerate(filtered_contours):
    x, y, w, h = cv2.boundingRect(cont)
    resize = cv2.rectangle(resize, (x, y), (x + w, y + h), (0, 255, 0), 2)
    resize = cv2.putText(resize, str(i + 1), (x, y - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)

# Display the results
print(f"จำนวนเหรียญในภาพ: {len(filtered_contours)}")
cv2.imshow("Original Image", resize)
cv2.imshow("Binary Image", binary)
cv2.imshow("Closing Image", closing)
cv2.waitKey(0)
cv2.destroyAllWindows()
