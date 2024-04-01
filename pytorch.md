---
jupyter:
  colab:
  kernelspec:
    display_name: Python 3
    name: python3
  language_info:
    name: python
  nbformat: 4
  nbformat_minor: 0
---

::: {.cell .markdown id="V-zBUqVasNhK"}
**Creat tensor**
:::

::: {.cell .code id="v-glge-Ur9dE"}
``` python
import torch
import numpy as np

# tạo một tensor từ list cho trước dùng torch.Tensor
t = torch.Tensor([[1,2,3],[3,4,5]])
#Tạo một Tensor với kích thước (2, 3) cho trước và có giá trị ngẫu nhiên tuân theo phân phối chuẩn với trung vị bằng 0 và phương sai bằng 1
t = torch.randn(2, 3)
# Tạo một Tensor với kích thước (2, 3) cho trước và tất cả phần tử có giá trị đều bằng 1
t = torch.ones(2, 3)
# Tạo một Tensor với kích thước (2, 3) cho trước và tất cả phần tử có giá trị đều bằng 0
t = torch.zeros(2, 3)
#Tạo một tensor có kích thước (2,3) với giá trị nằm trong khoảng từ 0->10
t = torch.randint(low = 0,high = 10,size = (2,3))
#Sử dụng torch.from_numpy để chuyển đổi từ Numpy array sang Tensor
a = np.array([[1,2,3],[3,4,5]])
t = torch.from_numpy(a)
# Sử dụng .numpy() để chuyển đổi từ Tensor sang Numpy array
t = t.numpy()
```
:::

::: {.cell .markdown id="wwpSW4d0tdu2"}
**các phương thức**
:::

::: {.cell .code id="yULWubwVta2m"}
``` python
A = torch.randn(2,4)
W = torch.randn(4,3)
#Nhân 2 ma trận sử dụng .mm
t = A.mm(W)
print(t)
#Ma trận chuyển vị
t = t.t()
print(t.shape)
#Bình phương mỗi giá trị trong Tensor
t = t**2
print(t)
#Trả về  size của Tensor
t_size = t.size()
print(t_size)
#Duỗi Tensor có kích thước (a,b) thành (1,a*b)
t = t.flatten()
print(t)

#Thêm 1 chiều với dim bằng 0 cho Tensor
t = t.unsqueeze(0)
print(t.shape)
#Giảm 1 chiều với dim bằng 0 cho Tensor
t = t.squeeze(0)
print(t.shape)
#hàm transpose có tác dụng đổi chiều dữ liệu ví dụ dữ liệu đang có shape=(224,224,3), sau khi thực hiện câu lệnh dưới sẽ thành (3,224,224)
t = t.transpose(2,0,1)
#hàm view có tác dụng giảm chiều dữ liệu ví dụ dữ liệu đang có shape = (3,224,224), sau khi thực hiện câu lệnh dưới sẽ thành (3,224*224)
t = t.view(t.size(0),-1)
```
:::

::: {.cell .markdown id="B7D_M1z_t__Y"}
**tạo model bằng pytorch**

1.  thừa kế lớp nn.Model thì tạo một mô hình học sâu dưới dạng một lớp,
    lớp này có 2 hàm int và forward:

-   init: khởi tạo vag nhận các biến. vì class kế thừa từ nn.Model nên
    khi khởi tạo một đối tượng mới của class thì phải tạo lớp kế thừa
    trong hàm init -\> super().\_\_init\_\_().
-   forward: input ban đầu là dữ liệu, sau đó đi qua từng layer của
    model và trả về output của model.
:::

::: {.cell .code id="yQ5PCT3ds6Gd"}
``` python
class Mymodel(nn.Model):
  def __init__(self):
    super().__init__() # định nghĩa các layer
    self.lin1 = nn.Linear(256,128)
    self.lin2 = nn.Linear(128,10)
  def forward(self, x):
    # kết nối các layer lại với nhau
    x = self.lin1(x)
    x = self.lin2(x)
    return x
```
:::

::: {.cell .markdown id="8oyDweLPwFp7"}
gồm 2 lớp Fully Connected với đầu vào là 1 tensor có độ dài 256 và đầu
ra là 10.
:::

::: {.cell .markdown id="im8AIlapwUfW"}
**Custom model**
:::

::: {.cell .code id="PlJMhZruwYIc"}
``` python
class mymodel(nn.Model):
  def __init__(self):
    super().__init__()
    self.layer1 = nn.Linear(256,128)
    self.layer2 = nn.Linear(128,256)
    self.layer3 = nn.Linear(128,10)

  def forward(self,x):
    x_out1 = self.layer1(x);
    x_out2 = x + self.layrt2(x_out1)
    x_out2 = self.layer1(x_out2)
    x = self.layer3(x_out2)
    return x
```
:::

::: {.cell .markdown id="apFYmIcMx2id"}
**Mô hình classification đơn giản dùng transfer learning**
:::

::: {.cell .code id="5poIa_Rlw_W-"}
``` python
import torch.nn as nn
import torchvision.models as models
from torchvision.models import FasterRCNN

class model(nn.Model):
  def __init__(self, num_cls):
    super().__init__()
    self.num_cls = num_cls
    backbone = models.resnet50(pretrained = True)
    # load a model; pre-trained on COCO
    # model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    self.model_backbone = nn.Sequential(*list(backbone.children()[:-1]))
    # cắt bỏ các layer ko cần thiết
    self.clf = nn.Sequential(
        nn.Linear(2048,512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(0.5),
            nn.Linear(512,self.num_cls)
        )
  def forward(self,input):
    output = self.model_backbone(input)
    output = output.view(output.size(0),-1 )
    output = self.clf(output)
    return output
```
:::

::: {.cell .code id="yoyrhy84HnoS"}
``` python
from torchvision.models import FasterRCNN
# load a model; pre-trained on COCO
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
num_classes = 2  # 1 class (wheat) + background

# get number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features

# replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
```
:::

::: {.cell .code id="ShtCfiWAIlEt"}
``` python
torch.nn.Conv2d(in_channels: int, out_channels: int, kernel_size: Union[T, Tuple[T, T]], stride: Union[T, Tuple[T, T]] = 1, padding: Union[T, Tuple[T, T]] = 0, dilation: Union[T, Tuple[T, T]] = 1, groups: int = 1, bias: bool = True, padding_mode: str = 'zeros')
```
:::

::: {.cell .markdown id="0hf2RlnOIpgS"}
**code một layer đơn giản và giải thích**
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="Dh127WevIl7d" outputId="12b5c9a6-a145-44b8-e9a9-fda810c61a74"}
``` python
import torch.nn as nn
import torch
#Tạo một lớp Conv2d với đầu vào channals là 3, đầu ra là 256, với kernel có kích thước (3,3) và stride = 2
layers =  nn.Conv2d(3,256,(3,3),2)
#Mình sẽ tạo ngẫu nhiên một Tensor có chiều (3,256,256) để xem nó thay đổi như thế nào sau khi đi qua lớp Conv2d, kết quả sẽ được một Tensor có chiều (256,127,127)
print(layers(torch.randn(1,3,256,256)).shape)
```

::: {.output .stream .stdout}
    torch.Size([1, 256, 127, 127])
:::
:::

::: {.cell .markdown id="wY2YjlWJJC3T"}
**sử dụng thư viện torchvision.datasets.ImageFolder để tạo dataset:**
:::

::: {.cell .code id="DssVJ5GGIyiF"}
``` python
from torchvision import transforms
from torchvision.datasets import ImageFolder
traindir = "data/train/"
t = transforms.Compose([
    transforms.Resize(size = 256),
    transforms.CenterCrop(size=224),
    transforms.ToTensor()
])
train_dataset = ImageFolder(root=traindir, transform = t)
print("so anh trong dataset: ", len(train_dataset))
print("hinh anh va nhan: ", train_dataset[2])
```
:::

::: {.cell .markdown id="DnY3NQly0TEz"}
**init : Là hàm khởi tạo, nhận vào các tham số và khởi tạo các tham số
tương ứng**

**len : Hàm trả về độ dài của dữ liệu**

**getitem: nhận vào là index, chỉ số này nằm trong độ dài của dữ liệu.
Hàm này mục tiêu để đọc dữ liệu, xử lí dữ liệu, nhãn và trả về dữ liệu
chuẩn để đưa vào model.**
:::

::: {.cell .code id="3i2tAaL1Kw99"}
``` python
class myDataset(Dataset):
    def __init__(self,data_dict,trans = None):
        super().__init__()
        self.data_dict = data_dict
        self.trans = trans

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self,idx):
        image_path = self.data_dict[idx]['data_link']
        label= self.data_dict[idx]['label']
        try:
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(e)
        if self.trans:
            transformed = self.trans(image=image)
            image = transformed["image"]
        return image.float(), label
```
:::

::: {.cell .markdown id="Ayalm_ez04yf"}
# Dataloader
:::

::: {.cell .code id="OBLWL1VV08ZM"}
``` python
torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None, num_workers=0, collate_fn=None, pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None, multiprocessing_context=None, generator=None)
```
:::

::: {.cell .markdown id="WS2TytZ51HCC"}
dataset :nhận vào class Dataset đã khởi tạo ở trên.

batch_size: thể hiện bạn muốn dữ liệu cua bạn được generate theo batch
bao nhiêu.

num_workers: khi bạn muốn chạy nhiều tiến trình cùng một lúc tùy vào
phần cứng của bạn.

collate_fn: Hàm này để định nghĩa cách sắp xếp và kết nối dữ liệu và
nhãn tương ứng theo từng lô dữ liệu.
:::

::: {.cell .code id="IBZBryy81X9F"}
``` python
trans = A.Compose([
            A.RandomCrop(width=256, height=256),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            ToTensorV2(),
        ])
train_dataset = myDataset(data_dict=data_dict,trans=trans)
train_dataloader = DataLoader(train_dataset,batch_size = 64, shuffle=True, num_workers=10)
```
:::

::: {.cell .markdown id="vTrWBXZR1b7Q"}
# Huấn luyện mô hình
:::

::: {.cell .code id="1z9NjOZF1g6c"}
``` python
num_epochs = 10
for epoch in range(num_epochs):
    #Thiết lập trạng thái huấn luyện cho mô hình
    model.train()
    for x_batch,y_batch in train_dataloader:
        #xóa gradients
        optimizer.zero_grad()
        # Cho dữ liệu qua model và trả về output cần tìm
        pred = model(x_batch)
        # Tính toán giá trị lỗi và backpropagation
        loss = loss_criterion(pred, y_batch)
        loss.backward()
        # Cập nhật trọng số
        optimizer.step()
    #Thiết lập trạng thái đánh giá cho mô hình, ở bước này thì mô hình không backward và cập nhật trọng số
    model.eval()
    for x_batch,y_batch in valid_dataloader:
        pred = model(x_batch)
        val_loss = loss_criterion(pred, y_batch)
```
:::

::: {.cell .markdown id="H21VMQP92Yvw"}
**LOSS**
:::

::: {.cell .code id="PyCVydyK2dpJ"}
``` python
def dice_loss(pred, target, smooth = 1.):
    pred = pred.contiguous()
    target = target.contiguous()

    intersection = (pred * target).sum()

    loss = (1 - ((2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)))

    return loss.mean()
```
:::

::: {.cell .markdown id="uOvSTJx12kTa"}
kết hợp nhiều hàm loss lại với nhau
:::

::: {.cell .code id="ANVzeqNE2oeJ"}
``` python
def calc_loss(pred, target, metrics, bce_weight=0.5):

    bce = F.binary_cross_entropy( pred, target)

    dice = dice_loss(pred, target)

    loss = bce * bce_weight + dice * (1 - bce_weight)

    metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
    metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)

    return loss

```
:::

::: {.cell .code id="uDgTwZMu2q9I"}
``` python
```
:::
