#!/usr/bin/env python3
"""
MolEnc Docker客户端示例
展示如何使用Docker化的分子编码器API
"""

import requests
import json
import time
import pandas as pd
from typing import List, Dict, Any

class MolEncClient:
    """MolEnc Docker客户端"""
    
    def __init__(self, base_url: str = "http://localhost"):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        
    def health_check(self, encoder: str = None) -> Dict[str, Any]:
        """健康检查"""
        if encoder:
            url = f"{self.base_url}/api/{encoder}/health"
        else:
            url = f"{self.base_url}/health"
            
        response = self.session.get(url)
        response.raise_for_status()
        return response.json()
    
    def get_info(self, encoder: str) -> Dict[str, Any]:
        """获取编码器信息"""
        url = f"{self.base_url}/api/{encoder}/info"
        response = self.session.get(url)
        response.raise_for_status()
        return response.json()
    
    def encode_morgan(self, smiles: List[str], **kwargs) -> Dict[str, Any]:
        """Morgan指纹编码"""
        url = f"{self.base_url}/api/morgan/encode"
        
        # 默认参数
        params = {
            "smiles": smiles,
            "n_bits": kwargs.get("n_bits", 2048),
            "radius": kwargs.get("radius", 2),
            "use_counts": kwargs.get("use_counts", False),
            "use_features": kwargs.get("use_features", False)
        }
        
        response = self.session.post(url, json=params)
        response.raise_for_status()
        return response.json()
    
    def encode_chemberta(self, smiles: List[str], **kwargs) -> Dict[str, Any]:
        """ChemBERTa编码"""
        url = f"{self.base_url}/api/chemberta/encode"
        
        # 默认参数
        params = {
            "smiles": smiles,
            "model_name": kwargs.get("model_name", "seyonec/ChemBERTa-zinc-base-v1"),
            "pooling_strategy": kwargs.get("pooling_strategy", "mean"),
            "max_length": kwargs.get("max_length", 512)
        }
        
        response = self.session.post(url, json=params)
        response.raise_for_status()
        return response.json()
    
    def encode_batch(self, encoder: str, smiles: List[str], batch_size: int = 100) -> List[List[float]]:
        """批量编码，自动分批处理"""
        all_results = []
        
        for i in range(0, len(smiles), batch_size):
            batch = smiles[i:i+batch_size]
            
            if encoder == "morgan":
                result = self.encode_morgan(batch)
                all_results.extend(result["fingerprints"])
            elif encoder == "chemberta":
                result = self.encode_chemberta(batch)
                all_results.extend(result["embeddings"])
            else:
                raise ValueError(f"Unknown encoder: {encoder}")
            
            # 避免过快请求
            time.sleep(0.1)
        
        return all_results

def demo_basic_usage():
    """基本使用示例"""
    print("=== 基本使用示例 ===")
    
    client = MolEncClient()
    
    # 健康检查
    print("健康检查...")
    health = client.health_check()
    print(f"✅ 系统状态: {health}")
    
    # 获取编码器信息
    print("\n获取Morgan信息...")
    morgan_info = client.get_info("morgan")
    print(f"Morgan描述: {morgan_info['description']}")
    
    print("\n获取ChemBERTa信息...")
    chemberta_info = client.get_info("chemberta")
    print(f"ChemBERTa描述: {chemberta_info['description']}")
    
    # 编码单个分子
    smiles = ["CCO", "CCCO", "CCCCO"]
    
    print("\nMorgan指纹编码...")
    morgan_result = client.encode_morgan(smiles, n_bits=1024)
    print(f"指纹形状: {morgan_result['shape']}")
    print(f"前3个指纹的前10位: {[fp[:10] for fp in morgan_result['fingerprints'][:3]]}")
    
    print("\nChemBERTa编码...")
    chemberta_result = client.encode_chemberta(smiles)
    print(f"嵌入形状: {chemberta_result['shape']}")
    print(f"前3个嵌入的前10维: {[emb[:10] for emb in chemberta_result['embeddings'][:3]]}")

def demo_batch_processing():
    """批量处理示例"""
    print("\n=== 批量处理示例 ===")
    
    client = MolEncClient()
    
    # 生成分子列表
    smiles_list = [
        "CCO", "CCCO", "CCCCO", "CCCCCO", "CCCCCCO",
        "c1ccccc1", "c1ccc(C)cc1", "c1ccc(CC)cc1",
        "CC(=O)O", "CC(=O)N", "CC(=O)C",
        "NC(C)=O", "NC(CC)=O", "NC(CCC)=O"
    ] * 10  # 扩展到130个分子
    
    print(f"处理 {len(smiles_list)} 个分子...")
    
    # Morgan批量编码
    start_time = time.time()
    morgan_fps = client.encode_batch("morgan", smiles_list, batch_size=50)
    morgan_time = time.time() - start_time
    print(f"Morgan编码完成: {len(morgan_fps)} 个分子, 耗时 {morgan_time:.2f}s")
    print(f"指纹维度: {len(morgan_fps[0])}")
    
    # ChemBERTa批量编码
    start_time = time.time()
    chemberta_embs = client.encode_batch("chemberta", smiles_list, batch_size=32)
    chemberta_time = time.time() - start_time
    print(f"ChemBERTa编码完成: {len(chemberta_embs)} 个分子, 耗时 {chemberta_time:.2f}s")
    print(f"嵌入维度: {len(chemberta_embs[0])}")

def demo_dataframe_processing():
    """DataFrame处理示例"""
    print("\n=== DataFrame处理示例 ===")
    
    client = MolEncClient()
    
    # 创建示例数据
    data = {
        'smiles': [
            'CCO', 'CCCO', 'CCCCO', 'c1ccccc1', 'c1ccc(C)cc1',
            'CC(=O)O', 'CC(=O)N', 'NC(C)=O', 'CCN', 'CCCN'
        ],
        'name': [
            'Ethanol', 'Propanol', 'Butanol', 'Benzene', 'Toluene',
            'Acetic acid', 'Acetamide', 'Acetamide', 'Ethylamine', 'Propylamine'
        ]
    }
    
    df = pd.DataFrame(data)
    print(f"原始数据: {len(df)} 行")
    print(df.head())
    
    # 添加Morgan指纹
    print("\n添加Morgan指纹...")
    morgan_fps = client.encode_batch("morgan", df['smiles'].tolist())
    df['morgan_fp'] = morgan_fps
    
    # 添加ChemBERTa嵌入
    print("添加ChemBERTa嵌入...")
    chemberta_embs = client.encode_batch("chemberta", df['smiles'].tolist())
    df['chemberta_emb'] = chemberta_embs
    
    print("\n处理后的数据:")
    print(df.head())
    print(f"\n数据形状: {df.shape}")
    print(f"Morgan指纹维度: {len(df['morgan_fp'].iloc[0])}")
    print(f"ChemBERTa嵌入维度: {len(df['chemberta_emb'].iloc[0])}")

def demo_error_handling():
    """错误处理示例"""
    print("\n=== 错误处理示例 ===")
    
    client = MolEncClient()
    
    # 测试无效SMILES
    print("测试无效SMILES...")
    try:
        result = client.encode_morgan(["invalid_smiles"])
        print("❌ 应该抛出异常")
    except requests.exceptions.HTTPError as e:
        print(f"✅ 正确捕获异常: {e.response.status_code}")
    
    # 测试空列表
    print("\n测试空列表...")
    try:
        result = client.encode_morgan([])
        print("结果:", result['metadata'])
    except Exception as e:
        print(f"异常: {e}")
    
    # 测试服务不可用
    print("\n测试服务不可用...")
    bad_client = MolEncClient("http://localhost:9999")
    try:
        bad_client.health_check()
    except requests.exceptions.ConnectionError:
        print("✅ 正确捕获连接异常")

def demo_performance_comparison():
    """性能比较示例"""
    print("\n=== 性能比较示例 ===")
    
    client = MolEncClient()
    
    # 不同大小的分子集合
    test_sizes = [10, 50, 100, 200]
    
    results = []
    
    for size in test_sizes:
        # 生成测试数据
        test_smiles = ["CCO", "CCCO", "CCCCO"] * (size // 3 + 1)[:size]
        
        # 测试Morgan
        start_time = time.time()
        morgan_result = client.encode_morgan(test_smiles)
        morgan_time = time.time() - start_time
        
        # 测试ChemBERTa
        start_time = time.time()
        chemberta_result = client.encode_chemberta(test_smiles)
        chemberta_time = time.time() - start_time
        
        results.append({
            'n_molecules': size,
            'morgan_time': morgan_time,
            'chemberta_time': chemberta_time,
            'morgan_mol_per_sec': size / morgan_time,
            'chemberta_mol_per_sec': size / chemberta_time
        })
        
        print(f"分子数: {size:3d} | Morgan: {morgan_time:.3f}s ({size/morgan_time:.1f} mol/s) | "
              f"ChemBERTa: {chemberta_time:.3f}s ({size/chemberta_time:.1f} mol/s)")
    
    # 总结
    df_results = pd.DataFrame(results)
    print(f"\n平均处理速度:")
    print(f"Morgan: {df_results['morgan_mol_per_sec'].mean():.1f} 分子/秒")
    print(f"ChemBERTa: {df_results['chemberta_mol_per_sec'].mean():.1f} 分子/秒")

if __name__ == "__main__":
    print("🧬 MolEnc Docker客户端示例")
    print("=" * 50)
    
    # 运行所有示例
    try:
        demo_basic_usage()
        demo_batch_processing()
        demo_dataframe_processing()
        demo_error_handling()
        demo_performance_comparison()
        
        print("\n🎉 所有示例运行完成！")
        
    except Exception as e:
        print(f"\n❌ 运行失败: {e}")
        print("请确保Docker服务已启动并运行正确")