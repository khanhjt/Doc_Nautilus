# 1. Giới thiệu về HOG
## 1.1 Giới thiệu chung
Có rất nhiều phương pháp khác nhau trong cv. Khi phân loại ảnh chúng ta có thể áp dụng họ các mô hình CNN và để phát hiện vật thể ta dùng YOLO, SSD, F-RCNN,F-RCNN. Các thuật toán kể trên đều là mô hình DL, vậy trc khi DL bùng nổ thì HOG là thuật toán hiệu quả để xử lý ảnh
Thuật toán này tạo ra bộ mô tả đặc trưng(feature descriptor) để phát hiện vật thể. Khi có 1 bức ảnh thì ta lấy ra 2 ma trận quan trọng giúp lưu thông tin ảnh đó là độ lớn gradient và phương của gradient. Bằng cách kết hợp 2 thông tin này vào biểu đồ histogram, trong đó độ lớn gradient được đếm theo các nhóm bins của phương gradient. Cuối cùng chúng ta sẽ thu được vector HOG đại diejn cho histogram. 
## 1.2 Ứng dụng
- Nhận diện người
- Nhận diện khuôn mặt
- Nhận diện các vật thể
- Nhận diện các vật thể khác
- Tạo feature cho các bài toán phân loại ảnh
## 1.3 Thuật ngữ
- Feature Descriptor: Bộ mô tả đặc trưng
- Histogram: LÀ biểu đồ histogram biểu diễn phân phối của các cường độ màu sắc theo khoảng giá trị
- Gradient: Là đạo hàm của vector cường độ màu sắc theo hướng di chuyển của các vật thể trong ảnh
- Local cell: Ô cục bộ. Trong HOG một hình ảnh được chia thành nhiều cell bởi một lưới ô vuông. Mỗi cell được gọi là một ô cục bộ
- Local normalization: Phép chuẩn hóa được thực hiện trên vùng cục bộ. Thuognwf là chia cho norm chuẩn bâc 2 hoặc norm chuẩn bậc 1. Nhằm mục đích chuẩn hóa dữ liệu đồng nhất các giá trj cường độ màu sắc về chung một phân phối
- gradient magnitude: Độ lớn gradient $$|G| = \sqrt{G_x^{2} +G_y^{2}}$$ 
# 2. Lý thuyết
HOG hoạt độg dựa trên hình dạng của một vật thể cục bộ được mô tả theo 2 ma trận kể hotrên, vậy 2 ma trận đó được tính như thế nào? Đầu tiên hình ảnh được chia thành 2 lưới ô vuông và trên đó chúng ta xác định rất nhiều các vùng cục bộ liền kề chồng lấn lên nhau. Các vùng này tuongư tự như những hình nahr cục bộ mà chúng ta tính tích chập trong CNN. Một vùng cục bộ bao gồm nhiều ô cục bộ (HOG là 4) có kt 8x8. Sau đó một biểu đồ histogram thống kê độ lớn gradient được tính toán trên mỗi ô cục bộ . Bộ mô tả HOG được tạo thành bằng cách kết nối 4 vector histogram ứng với mỗi ô thành một vector tổng hợp. Để cải thiện Độ chính xác, mỗi giá trị của vector histogram trên vùng cục bộ sẽ được chuẩn hó thep norm chuẩn bậc một hoawcwjbaawcj  . phép chuẩn hóa này nhằm taoj ra sự bất biến tốt hơn đối với những thay đổi trong chiếu sáng và đổi bóng.  
Bộ mô tả HOG có một vài lời thế chính so với các bộ mô tả khác. Vì nó hoạt động trên các ô cục bộ nó bất biến đối với các biến đổi hình học thay đổi độ sáng.  
## 2.1 Tính toán gradient
Tiền xử lý dữ liệu ảnh gồm chuẩn hóa màu sắc và giá trị gamma. Cái này bị bỏ qua trong bộ mô tả HOG, vì việc chuẩn hóa bộ mô tả ở các bước tiếp theo đã đạt được kết quả tương tự. Thay vào đó tại bước đầu tiên của tính toán mô tả chúng ta sẽ tính các giá trị gradient. Phương pháp phổ biến nhất là áp dụng một mặt nạ đạo hàm rời rạc (discrete derivative mask) theo một hoặc cả hai chiều ngang và dọc. Cụ thể, phương pháp sẽ lọc ma trận cường độ ảnh với các bộ lọc như Sobel mask hoặc scharr.
Để tính bộ lọc sobel, phép tích chập của kernel kích thước $3x3$ được thực hiện với hình ảnh ban đầu. Nếu chúng ta kí hiệu \mathbf{I} à ma trận ảnh gốc và 
 là 2 ma trận ảnh mà mỗi điểm trên nó lần lượt là đạo hàm theo trục 
 trục 
. Chúng ta có thể tính toán được kernel như sau:


$$G_x = \begin{bmatrix} -1 & 0 & 1 \\
-2 & 0 & 2 \\
-1 & 0 & 1 \\ \end{bmatrix} * \mathbf{I}$$

Đạo hàm theo chiều dọc:
$$G_y = \begin{bmatrix} -1 & -2 & -1 \\
0 & 0 & 0 \\
1 & 2 & 1 \\ \end{bmatrix} * \mathbf{I}$$

Kí hiệu * tương tự như phép tích chập giữa bộ lọc bên trái và ảnh đầu vào bên phải.


tính toán:
import numpy as np
import cv2
import matplotlib.pyplot as plt

img = plt.imread('pic.JPEG', cv2.IMREAD_UNCHANGED)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print('image shape:', img.shape)
print('gray shape: ', gray.shape)

plt.figure(figsize = (16, 4))
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title('Original Image')
plt.subplot(1, 2, 2)
plt.imshow(gray)
plt.title('Gray Image')
1
2
image shape: (1120, 2016, 3)
gray shape:  (1120, 2016)
![image](https://github.com/khanhjt/Doc_Nautilus/assets/105477211/ef6707e8-0d74-46bd-85ca-b73a1fb5c78c)

# Calculate gradient gx, gy
gx = cv2.Sobel(gray, cv2.CV_32F, dx=0, dy=1, ksize=3)
gy = cv2.Sobel(gray, cv2.CV_32F, dx=1, dy=0, ksize=3)

print('gray shape: {}'.format(gray.shape))
print('gx shape: {}'.format(gx.shape))
print('gy shape: {}'.format(gy.shape))

gray shape: (1120, 2016)
gx shape: (1120, 2016)
gy shape: (1120, 2016)

g, theta = cv2.cartToPolar(gx, gy, angleInDegrees=True) 
print('gradient format: {}'.format(g.shape))
print('theta format: {}'.format(theta.shape))
gradient format: (1120, 2016)
theta format: (1120, 2016)

w = 20
h = 10

plt.figure(figsize=(w, h))
plt.subplot(1, 4, 1)
plt.title('gradient of x')
plt.imshow(gx)

plt.subplot(1, 4, 2)
plt.title('gradient of y')
plt.imshow(gy)

plt.subplot(1, 4, 3)
plt.title('Magnitute of gradient')
plt.imshow(g)

plt.subplot(1, 4, 4)
plt.title('Direction of gradient')
plt.imshow(theta)
![image](https://github.com/khanhjt/Doc_Nautilus/assets/105477211/d31f6418-a211-4fdb-b25b-efd850fabb10)


