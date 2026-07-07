import torchvision
import os
import urllib.request
import ssl

# 解决SSL证书问题
ssl._create_default_https_context = ssl._create_unverified_context

def download_cifar10_with_progress(save_path='./data'):
    """
    带进度条下载 CIFAR-10
    """
    # 创建目录
    os.makedirs(save_path, exist_ok=True)
    
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = os.path.join(save_path, "cifar-10-python.tar.gz")
    
    # 如果文件已存在，直接加载
    if os.path.exists(filename):
        print(f"✅ 文件已存在: {filename}")
    else:
        print("📥 开始下载 CIFAR-10 (约163MB)...")
        print("⏳ 请耐心等待，进度条会显示...")
        
        def report_hook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(100, downloaded * 100 / total_size)
            # 动态显示进度
            bar_len = 30
            filled_len = int(bar_len * percent / 100)
            bar = '█' * filled_len + '░' * (bar_len - filled_len)
            print(f'\r进度: |{bar}| {percent:.1f}% ({downloaded/1024/1024:.1f}MB/{total_size/1024/1024:.1f}MB)', end='')
        
        try:
            urllib.request.urlretrieve(url, filename, reporthook=report_hook)
            print("\n✅ 下载完成！")
        except Exception as e:
            print(f"\n❌ 下载失败: {e}")
            return False
    
    # 使用 torchvision 加载
    print("📂 正在加载数据集...")
    trainset = torchvision.datasets.CIFAR10(
        root=save_path,
        train=True,
        download=False,
        transform=None
    )
    testset = torchvision.datasets.CIFAR10(
        root=save_path,
        train=False,
        download=False,
        transform=None
    )
    
    print(f"✅ 加载成功！训练集: {len(trainset)}张, 测试集: {len(testset)}张")
    return trainset, testset

# ============ 使用 ============
if __name__ == "__main__":
    # 下载并加载
    trainset, testset = download_cifar10_with_progress('./data')