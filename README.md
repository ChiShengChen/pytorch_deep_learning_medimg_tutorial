# 醫學影像分類深度學習模型教學

這個專案簡單實現了三種常見不同的深度學習模型（VGG16、ResNet18、ViT）來進行醫學影像分類任務，使用 PathMNIST 資料集。

## 專案結構

```
medical_image_classification/
│
├── models/ # 模型定義
│ ├── vgg.py # VGG16 模型
│ ├── resnet.py # ResNet18 模型
│ ├── vit.py # Vision Transformer 模型
│ └── vit_handmade.py # 純pytorch疊成的Vision Transformer 模型
│
├── data/ # 資料處理
│ └── dataset.py # 資料載入和預處理
│
├── utils/ # 工具函數
│ └── trainer.py # 訓練相關函數
│
├── checkpoints/ # 保存訓練好的模型
│
└── main.py # 主程式
```

## 環境配置

1. 首先確保你已經安裝了 Python 3.7 或更高版本
2. 安裝所需的套件：

```bash
pip install torch torchvision numpy matplotlib
```

## 訓練和測試

```bash
python main.py
```

這將訓練和測試三個模型，並保存訓練好的模型到 `checkpoints` 目錄中。


## 資料處理

### PathMNIST 資料集介紹
PathMNIST 是 MedMNIST 資料集系列的一部分，專門用於病理學圖像分類：
![image](https://github.com/user-attachments/assets/94acae3a-a6cc-41e8-a209-0f0e7fd72062)

- **圖像大小**：28×28 像素，RGB 彩色圖像
- **類別數量**：9 種不同的組織類型
- **數據分布**：
  - 訓練集：89,996 張圖像
  - 驗證集：10,004 張圖像
  - 測試集：7,180 張圖像
- **類別說明**：
  1. 腺癌 (ADI)
  2. 結締組織 (BACK)
  3. 碎片 (DEB)
  4. 淋巴細胞 (LYM)
  5. 黏液 (MUC)
  6. 肌肉 (MUS)
  7. 正常結腸黏膜 (NORM)
  8. 間質 (STR)
  9. 腫瘤上皮 (TUM)

### 資料預處理流程
- 圖像預處理：
  1. 轉換為張量（ToTensor）
  2. 轉換為灰度圖並擴展為3通道
  3. 標準化處理（Normalize）：使用均值 0.5 和標準差 0.5
- 相關程式碼：`data/dataset.py`

### 資料載入
- 使用 PyTorch 的 DataLoader 進行批次載入
- 預設批次大小（batch size）：64
- 訓練集進行隨機打亂（shuffle=True）


## 模型介紹

### 1. VGG16
- 經典的卷積神經網絡
- 特點：結構簡單，層數深，適合圖像分類任務
- 位置：`models/vgg.py`

### 2. ResNet18
- 殘差網絡，解決了深層網絡的梯度消失問題
- 特點：引入跳躍連接，訓練更穩定
- 位置：`models/resnet.py`

### 3. Vision Transformer (ViT)
- 基於 Transformer 架構的視覺模型
- 特點：將圖像分割成小塊進行處理，注意力機制強大，但需要海量資料才訓練的來
- 位置：`models/vit.py`

## 資料處理

- 使用 PathMNIST 資料集
- 圖像預處理：
  1. 轉換為張量
  2. 轉換為灰度圖並擴展為3通道
  3. 標準化處理
- 相關程式碼：`data/dataset.py`

## 訓練過程

訓練器（`utils/trainer.py`）包含：
- 損失函數：CrossEntropyLoss
- 優化器：Adam
- 訓練過程監控：損失值和準確率
- 模型保存功能

## 如何使用

1. 複製下載專案：

```bash
git clone https://github.com/ChiShengChen/pytorch_deep_learning_medimg_tutorial.git
```

2. 運行訓練：

```bash
python main.py
```




## 訓練結果

訓練完成後：
- 模型權重將保存在 `checkpoints` 目錄
- 每個 epoch 都會顯示訓練損失和準確率
- 可以比較三個模型的性能差異

## 擴展建議

1. 添加驗證集評估
2. 實現更多模型架構
3. 添加資料增強
4. 添加學習率調度器
5. 添加早停機制

## 常見問題

1. **顯存不足**
   - 減小 batch size
   - 使用較小的模型配置

2. **訓練時間過長**
   - 可以先用較少的 epoch 測試
   - 考慮使用 GPU 訓練

3. **準確率不理想**
   - 調整學習率
   - 增加訓練輪數
   - 添加資料增強

## 參考資料

- [PyTorch 官方文檔](https://pytorch.org/docs/stable/index.html)
- [MedMNIST 資料集](https://medmnist.com/)
- [VGG 論文: Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)
- [ResNet 論文: Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- [Vision Transformer 論文: An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)

## 執行過程截圖
<img width="675" alt="image" src="https://github.com/user-attachments/assets/e96d5580-f06b-4cf0-b954-732a081d3800" />

<img width="591" alt="image" src="https://github.com/user-attachments/assets/569605b9-e86e-4946-887d-1ebd5d99fcda" />


## 注意事項

- 確保有足夠的硬碟空間存放資料集
- 建議使用 GPU 進行訓練
- 第一次運行時會自動下載資料集，需要等待一段時間

## 其他參考資料
這份以影像辨識為主，若想學其他時序模型可以參考這份：
https://github.com/ChiShengChen/Deep_learning_introducrion
