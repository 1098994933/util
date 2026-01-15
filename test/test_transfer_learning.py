import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# 设备配置（优先GPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# ===================== 1. 构造模拟数据（表格数据，标签负相关） =====================
np.random.seed(42)
n_features = 10  # 特征维度
n_source = 500  # 源域样本量
n_target = 100  # 目标域样本量

# 构造特征（源域/目标域特征分布略有差异）
X_source = np.random.randn(n_source, n_features) + 0.1
X_target = np.random.randn(n_target, n_features)

# 构造标签（源域 = -1 * 目标域 + 低噪声）
true_coef = np.array([1, -0.5, 0.8, -0.3, 0.6, -0.2, 0.4, -0.7, 0.9, -0.1])
y_target = np.dot(X_target, true_coef) + np.random.randn(n_target) * 0.02
y_source = -1.0 * np.dot(X_source, true_coef) + np.random.randn(n_source) * 0.02
y_source_reversed = -1 * y_source
# ===================== 3. 数据预处理（标准化+划分数据集） =====================
# 标准化（表格数据必须做，提升神经网络收敛性）
scaler = StandardScaler()
X_source_scaled = scaler.fit_transform(X_source)
X_target_scaled = scaler.transform(X_target)  # 用源域的scaler，避免目标域信息泄露

# 划分目标域训练/测试集
X_t_train, X_t_test, y_t_train, y_t_test = train_test_split(
    X_target_scaled, y_target, test_size=0.5, random_state=42
)


# ===================== 4. 自定义Dataset（适配PyTorch） =====================
class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32).to(device)
        self.y = torch.tensor(y, dtype=torch.float32).to(device)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class TabularMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[64, 32, 16], output_dim=1):
        super(TabularMLP, self).__init__()
        # 构建隐藏层
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            prev_dim = dim
        # 输出层（回归任务无激活）
        layers.append(nn.Linear(prev_dim, output_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x).squeeze()  # squeeze：[batch,1] → [batch]


# 初始化模型
model = TabularMLP(input_dim=n_features).to(device)
print("模型结构:\n", model)

# 构建DataLoader
batch_size = 32
# 源域数据集（反转标签）
source_dataset = TabularDataset(X_source_scaled, y_source_reversed)
source_loader = DataLoader(source_dataset, batch_size=batch_size, shuffle=True)
# 目标域训练/测试集
target_train_dataset = TabularDataset(X_t_train, y_t_train)
target_train_loader = DataLoader(target_train_dataset, batch_size=batch_size, shuffle=True)
target_test_dataset = TabularDataset(X_t_test, y_t_test)
target_test_loader = DataLoader(target_test_dataset, batch_size=batch_size, shuffle=False)


# ===================== 阶段1：源域预训练（学习通用特征） =====================
def train_pretrain(model, dataloader, epochs=50, lr=1e-3):
    criterion = nn.MSELoss()  # 回归损失
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for X, y in dataloader:
            optimizer.zero_grad()
            pred = model(X)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * X.size(0)

        avg_loss = total_loss / len(dataloader.dataset)
        if (epoch + 1) % 10 == 0:
            print(f"预训练Epoch {epoch + 1:2d} | 平均MSE: {avg_loss:.6f}")

    return model


# 预训练（在反转后的源域数据上）
pretrained_model = train_pretrain(model, source_loader, epochs=50, lr=1e-3)
# 保存预训练权重
torch.save(pretrained_model.state_dict(), "pretrained_model.pth")


# ===================== 阶段2：目标域微调（冻结底层参数） =====================
def freeze_layers(model, freeze_layers_idx=[0, 1, 2, 3]):
    """
    冻结指定层的参数（不更新梯度）
    :param model: 预训练模型
    :param freeze_layers_idx: 要冻结的层索引（model.model是Sequential，可打印查看）
    """
    # 遍历所有层，默认全部解冻
    for param in model.parameters():
        param.requires_grad = True

    # 冻结指定层
    for idx in freeze_layers_idx:
        for param in model.model[idx].parameters():
            param.requires_grad = False

    # 打印冻结状态
    print("\n层冻结状态:")
    for idx, layer in enumerate(model.model):
        freeze_status = "冻结" if not any(p.requires_grad for p in layer.parameters()) else "解冻"
        print(f"层 {idx}: {layer.__class__.__name__} → {freeze_status}")


# 加载预训练权重并冻结底层（前4层：Linear(10→64) + ReLU + Linear(64→32) + ReLU）
finetune_model = TabularMLP(input_dim=n_features).to(device)
finetune_model.load_state_dict(torch.load("pretrained_model.pth"))
# 冻结底层特征层，仅训练顶层（32→16 + 16→1）
freeze_layers(finetune_model, freeze_layers_idx=[0, 1, 2, 3])


# 微调训练（仅更新未冻结层的参数）
def train_finetune(model, dataloader, epochs=30, lr=1e-4):
    criterion = nn.MSELoss()
    # 仅优化未冻结的参数（节省计算）
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for X, y in dataloader:
            optimizer.zero_grad()
            pred = model(X)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * X.size(0)

        avg_loss = total_loss / len(dataloader.dataset)
        if (epoch + 1) % 10 == 0:
            print(f"微调Epoch {epoch + 1:2d} | 平均MSE: {avg_loss:.6f}")

    return model


# 开始微调
finetuned_model = train_finetune(finetune_model, target_train_loader, epochs=30, lr=1e-4)

# ===================== 阶段3：基线模型（仅目标域训练） =====================
# 初始化全新模型，仅用目标域数据训练
baseline_model = TabularMLP(input_dim=n_features).to(device)
baseline_model = train_pretrain(baseline_model, target_train_loader, epochs=50, lr=1e-3)


def evaluate_model(model, dataloader):
    """评估模型MSE"""
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    mse = mean_squared_error(all_labels, all_preds)
    return mse

# 评估结果
baseline_mse = evaluate_model(baseline_model, target_test_loader)
finetune_mse = evaluate_model(finetuned_model, target_test_loader)

# 打印最终结果
print("\n===== 迁移学习结果对比 =====")
print(f"仅目标域训练 MSE: {baseline_mse:.6f}")
print(f"源域预训练+冻结微调 MSE: {finetune_mse:.6f}")
print(f"性能提升: {((baseline_mse - finetune_mse)/baseline_mse)*100:.2f}%")