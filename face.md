# 1. Các phương pháp xác thực khuôn mặt
## 1.1. Phương pháp truyền thống
Các phương pháp truyền thống: Xác thực khuôn mặt bằng cách trích xuất ra một land mark cho face. Land mark như là một bản đồ xác định các vị trí cố định trên khuôn mặt của một người như mắt, mũi, miệng, lông mày,….

![image](https://github.com/khanhjt/Doc_Nautilus/assets/105477211/1ce3414e-09fa-42ac-a1a5-5613d2e2a06c)

Như vậy thay land mark face đã loại bỏ những phần thông tin không cần thiết và giữ lại những thông tin chính. Khi đó mỗi khuôn mặt sẽ được nén thành một véc tơ n chiều. Thông thường là 68 chiều.

Sử dụng đầu vào là land mark face, áp dụng các thuật toán cổ điển như SVM, k-NN, Naive Bayes, Random Forest, MLP, … để phân loại khuôn mặt cho một người.
## 1.2 Nhận diện 3D
Kĩ thuật nhận diện 3D sẽ sử dụng không gian 3 chiều để biểu diễn khuôn mặt. Từ thông tin này có thể xác định các đặc trưng khác nhau trên bề mặt khuôn mặt như các đường countour của mắt, mũi, cằm.

Một lợi thế của nhận diện khuôn mặt 3D là không bị ảnh hưởng bởi những thay đổi về ánh sáng như các phương pháp 2D. Dữ liệu 3D đã cải thiện đáng kể độ chính xác của nhận dạng khuôn mặt.
Để tạo ra một ảnh 3D, một cụm ba camera được áp dụng. Mỗi camera sẽ hướng vào một góc khác nhau. Tất cả các camera này phối hợp cùng nhau trong việc theo dõi khuôn mặt của một người trong thời gian thực và có thể nhận diện chúng.

![image](https://github.com/khanhjt/Doc_Nautilus/assets/105477211/7b79b9bf-60ad-4fbc-94a3-a6e51b13c4be)

Nhận diện khuôn mặt của iphone là nhận diện 3D. Bạn sẽ phải quay tròn khuôn mặt của mình khi xác thực nó để thuật toán học các góc độ khác nhau.

## 1.3 Các phương pháp nhận diện khác
Ngoài ra còn có một số phương pháp nhận diện khuôn mặt như nhận diện cảm biến da và phương pháp kết hợp.

Phương pháp kết hợp có thể sử dụng nhiều thông tin từ đồng thời land mark face, nhận diện 3D, nhận diện cảm biến da và mang lại độ chính xác cao hơn vì nó nhận diện tốt được trong các trường hợp khuôn mặt có các biểu cảm như cau mày, chớp mắt, co dãn khi cười, nói và ít nhạy cảm với chiếu sáng.

# 2. Các bài toán khác nhau về face
* Nhận diện khuôn mặt (face identification): Đây là bài toán match one-many. Bài toán này sẽ trả lời cho câu hỏi “người này là ai?” bằng cách nhận input là ảnh khuôn mặt và output là nhãn tên người trong ảnh. Tác vụ này thường được áp dụng trong các hệ thống chấm công, hệ thống giám sát công dân, hệ thống cammera thông minh tại các đô thị.
* Xác thực khuôn mặt (face verification): Đây là bài toán match one-one. Bài toán này trả lời cho câu hỏi “có phải 2 ảnh đầu vào là cùng một người không?” Kết quả output sẽ là yes hoặc no. Bài toán thường được dùng trong các hệ thống bảo mật. Xác thực khuôn mặt trên điện thoại của bạn là một bài toán như vậy
* Tìm kiếm khuôn mặt đại diện (face clustering)
* Tìm kiếm khuôn mặt tương đương (face similarity)
# Các thuật toán
## 3.1 One-shot learning
One-shot learning là thuật toán học có giám sát mà mỗi một người chỉ cần 1 vài, rất ít hoặc thậm chí chỉ 1 bức ảnh duy nhất (để khỏi tạo ra nhiều biến thể).

Từ đầu vào là bức ảnh của một người, chúng ta sử dụng một kiến trúc thuật toán CNN đơn giản để dự báo người đó là ai.

Tuy nhiên nhược điểm của phương pháp này là chúng ta phải huấn luyện lại thuật toán thường xuyên khi xuất hiện thêm một người mới vì shape của output thay đổi tăng lên 1. Rõ ràng là không tốt đối với các hệ thống nhận diện khuôn mặt của một công ty vì số lượng người luôn biến động theo thời gian.

Để khắc phục được vấn đề này, chúng ta sử dụng phương pháp learning similarity.
## 3.2 Learning similarity
Phương pháp này dựa trên một phép đo khoảng cách giữa 2 bức ảnh, thông thường là các norm chuẩn l_1 hoặc l_2 sao cho nếu 2 bức ảnh thuộc cùng một người thì khoảng cách là nhỏ nhất và nếu không thuộc thì khoảng cách sẽ lớn hơn.

$$\begin{equation}
  \begin{cases} d(\text{img1}, \text{img2}) \leq \tau & \rightarrow &\text{same} \\
  d(\text{img1}, \text{img2}) > \tau & \rightarrow & \text{different}
  \end{cases}
\end{equation}$$

![image](https://github.com/khanhjt/Doc_Nautilus/assets/105477211/05cdf192-c4aa-4be8-bec4-4b6a19e69125)

Giống kiểu so sánh 2 bức ảnh thì ảnh nào giống nhất thì lấy với một ngưỡng giá trị gọi là threshold
Phương pháp này ko phụ thuộc số lượng classes -> ko cần huấn luyện lại khi xuất hiện class mới
Xây dụng một model encoding đủ tốt để chiếu các bức ảnh lên một không gian eucledean n  chiều -> sử dụng threshold để quyết định nhãn
Ưu điểm hơn so với one-shot learning là ko phải huấn luyện model
## 3.3 Siam network - biểu diễn ảnh trong không gian euledean
khi bạn đưa vào 2 bức ảnh và mô hình sẽ trả lời chúng thuộc về cùng 1 người hay không được gọi chung là Siam network.
Kiễn trúc mạng dựa trên basenetwork là một CNN và loại bỏ ouputlayer nhằm encoding ảnh thành vector embedding.
Đầu vào : 2 ảnh ngầu nhiên
Output: 2 vector tương ứng với biểu diễn 
Đưa 2vector vào hàm loss để đo lường sự khác biệt giữa chúng. 
![image](https://github.com/khanhjt/Doc_Nautilus/assets/105477211/d6f8e26f-f35f-47ed-bd7a-b1643d2b6b6a)


Khoảng cách d(x1,x2) = $$||f(\mathbf{x_1}) - f(\mathbf{x_2})||_2^{2}$$
hàm f(x) có tác dụng biến đổi layer fully connected trong mạng neural network để tạo tính phi tuyến và giảm chiều dữ liệu của các kích thước nhỏ(128)
Khi x1, x2 là cùng 1 người:
       $$||f(\mathbf{x_1}) - f(\mathbf{x_2})||_2^{2}$$  phải là một giá trị nhỏ.
Khi x1, x2 là 2 người khác thì giá trị hàm fx lớn

không cần phải lo lắng về vấn đề output shape thay đổi vì base network đã được loại bỏ layer cuối.

# 4. Facenet
Facenet chính là một dạng siam network có tác dụng biểu diễn các bức ảnh trong một không gian eucledean n chiều (thường là 128) sao cho khoảng cách giữa các véc tơ embedding càng nhỏ, mức độ tương đồng giữa chúng càng lớn

## 4.1 Khái niệm thuật toán
* Base network áp dụng một mạng convolutional neural network và giảm chiều dữ liệu xuống chỉ còn 128 chiều. Do đó quá trình suy diễn và dự báo nhanh hơn và đồng thời độ chính xác vẫn được đảm bảo.
* Sử dụng loss function là hàm triplot loss có khả năng học được đồng thời sự giống nhau giữa 2 bức ảnh cùng nhóm và phân biệt các bức ảnh không cùng nhóm. Do đó hiệu quả hơn rất nhiều so với các phương pháp trước đây.
## 4.2 Triple loss
Trong facenet, quá trình encoding của mạng convolutional neural network đã giúp ta mã hóa bức ảnh về 128 chiều. Sau đó những véc tơ này sẽ làm đầu vào cho hàm loss function đánh giá khoảng cách giữa các véc tơ.

Để áp dụng triple loss, chúng ta cần lấy ra 3 bức ảnh trong đó có một bức ảnh là anchor. Trong 3 ảnh thì ảnh anchor được cố định trước. Chúng ta sẽ lựa chọn 2 ảnh còn lại sao cho một ảnh là negative (của một người khác với anchor) và một ảnh là positive (cùng một người với anchor).

![image](https://github.com/khanhjt/Doc_Nautilus/assets/105477211/8fd0db67-5773-4607-8d88-9204d77b2b4c)
\mathbf{A}, \mathbf{P}, \mathbf{N}

Mục tiêu của hàm loss function là tối thiểu hóa khoảng cách giữa 2 ảnh khi chúng là negative và tối đa hóa khoảng cách khi chúng là positive. Như vậy chúng ta cần lựa chọn các bộ 3 ảnh sao cho:
* Ảnh Anchor và Positive khác nhau nhất: cần lựa chọn để khoảng cách $$d(\mathbf{A}, \mathbf{P})$$ lớn. Điều này cũng tương tự như bạn lựa chọn một ảnh của mình hồi nhỏ so với hiện tại để thuật toán học khó hơn. Nhưng nếu nhận biết được thì nó sẽ thông minh hơn.
* Ảnh Anchor và Negative giống nhau nhất: cần lựa chọn để khoảng cách $$d(\mathbf{A}, \mathbf{N})$$ nhỏ. Điều này tương tự như việc thuật toán phân biệt được ảnh của một người anh em giống bạn với bạn.
Triplot loss function luôn lấy 3 bức ảnh làm input và trong mọi trường hợp ta kì vọng:
    $$d(\mathbf{A}, \mathbf{P}) < d(\mathbf{A}, \mathbf{N}) \tag{1}$$
Để làm cho khoảng cách giữa vế trái và vế phải lớn hơn, chúng ta sẽ cộng thêm vào vế trái một hệ số không âm rất nhỏ. Khi đó trở thành:
$$\begin{eqnarray}d(\mathbf{A}, \mathbf{P}) + \alpha & \leq & d(\mathbf{A}, \mathbf{N}) \\
\rightarrow||f(\mathbf{A})-f(\mathbf{P})||_2^{2} + \alpha & \leq & ||f(\mathbf{A})-f(\mathbf{N})||_2^{2}\end{eqnarray}$$
Hàm loss:
$$\mathcal{L}(\mathbf{A, P, N}) = \sum_{i=0}^{n}||f(\mathbf{A}_i)-f(\mathbf{P}_i)||_2^{2} - ||f(\mathbf{A}_i)-f(\mathbf{N_i})||_2^{2}+ \alpha$$
n là số lượng bộ 3 hình ảnh
Sẽ không ảnh hưởng gì nếu ta nhận diện đúng ảnh Negative và Positive là cùng cặp hay khác cặp với Anchor. Mục tiêu của chúng ta là giảm thiểu các trường hợp hợp mô hình nhận diện sai ảnh Negative thành Postive nhất có thể. Do đó để loại bỏ ảnh hưởng của các trường hợp nhận diện đúng Negative và Positive lên hàm loss function. Ta sẽ điều chỉnh giá trị đóng góp của nó vào hàm loss function về 0.

Tức là nếu: $$||f(\mathbf{A})-f(\mathbf{P})||_2^{2} - ||f(\mathbf{A})-f(\mathbf{N})||_2^{2}+ \alpha \leq 0$$
sẽ được điều chỉnh về 0:
Hàm loss lúc này
        $$\mathcal{L}(\mathbf{A, P, N}) = \sum_{i=0}^{n}\max(||f(\mathbf{A}_i)-f(\mathbf{P}_i)||_2^{2} - ||f(\mathbf{A}_i)-f(\mathbf{N_i})||_2^{2}+ \alpha, 0)$$
## 4.2 Triple images input
Để mô hình khó học hơn và đồng thời cũng giúp mô hình phân biệt chuẩn xác hơn mức độ giống và khác nhau giữa các khuôn mặt, chúng ta cần lựa chọn các input theo bộ 3 khó học (hard triplets).
Xảy ra dấu "="
tìm ra bộ ba $$(\mathbf{A}, \mathbf{N}, \mathbf{P})$$ sao cho gần đạt được đẳng thức (xảy ra dấu =) nhất. 







