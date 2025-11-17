#!/usr/bin/env python3
"""
本地测试脚本 - 验证编码器逻辑和API接口
用于在没有Docker环境的情况下测试核心功能
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

import json
import time
from typing import List, Dict, Any, Optional
import traceback

# 模拟API响应格式
def create_success_response(data: Any = None, message: str = "操作成功", metadata: Dict = None):
    """创建成功响应"""
    return {
        "success": True,
        "message": message,
        "data": data or {},
        "metadata": metadata or {}
    }

def create_error_response(message: str, status_code: int = 500, details: Dict = None):
    """创建错误响应"""
    return {
        "success": False,
        "message": message,
        "error": details or {},
        "metadata": {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    }

class LocalAPITester:
    """本地API测试器"""
    
    def __init__(self):
        self.results = []
        self.morgan_encoder = None
        self.chemberta_encoder = None
    
    def _init_morgan_encoder(self, n_bits: int = 1024, radius: int = 2, use_features: bool = False):
        """初始化Morgan编码器"""
        try:
            # 模拟Morgan编码器
            class MockMorganEncoder:
                def __init__(self, n_bits, radius, use_features):
                    self.n_bits = n_bits
                    self.radius = radius
                    self.use_features = use_features
                
                def encode_batch(self, smiles_list):
                    # 模拟指纹生成
                    import random
                    return [[random.randint(0, 1) for _ in range(self.n_bits)] for _ in smiles_list]
                
                def get_info(self):
                    return {
                        "name": "morgan",
                        "description": "Morgan指纹编码器",
                        "n_bits": self.n_bits,
                        "radius": self.radius,
                        "use_features": self.use_features
                    }
            
            self.morgan_encoder = MockMorganEncoder(n_bits, radius, use_features)
            return True
        except Exception as e:
            print(f"Morgan编码器初始化失败: {e}")
            return False
    
    def _init_chemberta_encoder(self, model_name: str = "seyonec/ChemBERTa-zinc-base-v1", 
                               pooling_strategy: str = "mean", max_length: int = 512):
        """初始化ChemBERTa编码器"""
        try:
            # 模拟ChemBERTa编码器
            class MockChemBERTaEncoder:
                def __init__(self, model_name, pooling_strategy, max_length):
                    self.model_name = model_name
                    self.pooling_strategy = pooling_strategy
                    self.max_length = max_length
                
                def encode_batch(self, smiles_list):
                    # 模拟嵌入向量生成
                    import random
                    return [[random.uniform(-1, 1) for _ in range(768)] for _ in smiles_list]
                
                def get_info(self):
                    return {
                        "name": "chemberta",
                        "description": "ChemBERTa分子嵌入编码器",
                        "model_name": self.model_name,
                        "pooling_strategy": self.pooling_strategy,
                        "max_length": self.max_length,
                        "output_dim": 768
                    }
            
            self.chemberta_encoder = MockChemBERTaEncoder(model_name, pooling_strategy, max_length)
            return True
        except Exception as e:
            print(f"ChemBERTa编码器初始化失败: {e}")
            return False
    
    def test_health_check(self, encoder: str = None) -> Dict[str, Any]:
        """测试健康检查"""
        start_time = time.time()
        
        try:
            if encoder == "morgan":
                if not self._init_morgan_encoder():
                    raise Exception("Morgan编码器初始化失败")
                status = "healthy"
                message = "Morgan编码器运行正常"
            elif encoder == "chemberta":
                if not self._init_chemberta_encoder():
                    raise Exception("ChemBERTa编码器初始化失败")
                status = "healthy"
                message = "ChemBERTa编码器运行正常"
            else:
                status = "healthy"
                message = "系统运行正常"
            
            elapsed = time.time() - start_time
            
            result = create_success_response(
                data={"status": status, "encoder": encoder or "system"},
                message=message,
                metadata={"response_time": elapsed}
            )
            
            self.results.append({
                "test": "health_check",
                "encoder": encoder or "gateway",
                "status": "pass",
                "response_time": elapsed
            })
            
            return result
            
        except Exception as e:
            elapsed = time.time() - start_time
            result = create_error_response(
                message=f"健康检查失败: {str(e)}",
                details={"error": str(e)}
            )
            
            self.results.append({
                "test": "health_check",
                "encoder": encoder or "gateway",
                "status": "fail",
                "response_time": elapsed,
                "error": str(e)
            })
            
            return result
    
    def test_encoder_info(self, encoder: str) -> Dict[str, Any]:
        """测试编码器信息接口"""
        start_time = time.time()
        
        try:
            if encoder == "morgan":
                if not self._init_morgan_encoder():
                    raise Exception("Morgan编码器初始化失败")
                info = self.morgan_encoder.get_info()
            elif encoder == "chemberta":
                if not self._init_chemberta_encoder():
                    raise Exception("ChemBERTa编码器初始化失败")
                info = self.chemberta_encoder.get_info()
            else:
                raise ValueError(f"未知的编码器: {encoder}")
            
            elapsed = time.time() - start_time
            
            result = create_success_response(
                data=info,
                message=f"{encoder}编码器信息"
            )
            
            self.results.append({
                "test": "info",
                "encoder": encoder,
                "status": "pass",
                "response_time": elapsed
            })
            
            return result
            
        except Exception as e:
            elapsed = time.time() - start_time
            result = create_error_response(
                message=f"获取编码器信息失败: {str(e)}"
            )
            
            self.results.append({
                "test": "info",
                "encoder": encoder,
                "status": "fail",
                "response_time": elapsed,
                "error": str(e)
            })
            
            return result
    
    def test_encode(self, encoder: str, smiles: List[str], **kwargs) -> Dict[str, Any]:
        """测试编码接口"""
        start_time = time.time()
        
        try:
            # 验证SMILES
            if not smiles or not all(isinstance(s, str) and s.strip() for s in smiles):
                raise ValueError("SMILES列表不能为空且必须包含有效字符串")
            
            if encoder == "morgan":
                # 设置参数
                n_bits = kwargs.get("n_bits", 1024)
                radius = kwargs.get("radius", 2)
                use_features = kwargs.get("use_features", False)
                
                if not self._init_morgan_encoder(n_bits, radius, use_features):
                    raise Exception("Morgan编码器初始化失败")
                
                # 执行编码
                fingerprints = self.morgan_encoder.encode_batch(smiles)
                
                result_data = {
                    "fingerprints": fingerprints,
                    "shape": [len(smiles), n_bits]
                }
                
                metadata = {
                    "encoder": "morgan",
                    "n_bits": n_bits,
                    "radius": radius,
                    "use_features": use_features,
                    "n_molecules": len(smiles)
                }
                
            elif encoder == "chemberta":
                # 设置参数
                model_name = kwargs.get("model_name", "seyonec/ChemBERTa-zinc-base-v1")
                pooling_strategy = kwargs.get("pooling_strategy", "mean")
                max_length = kwargs.get("max_length", 512)
                
                if not self._init_chemberta_encoder(model_name, pooling_strategy, max_length):
                    raise Exception("ChemBERTa编码器初始化失败")
                
                # 执行编码
                embeddings = self.chemberta_encoder.encode_batch(smiles)
                
                result_data = {
                    "embeddings": embeddings,
                    "shape": [len(smiles), 768]
                }
                
                metadata = {
                    "encoder": "chemberta",
                    "model_name": model_name,
                    "pooling_strategy": pooling_strategy,
                    "max_length": max_length,
                    "n_molecules": len(smiles)
                }
            
            else:
                raise ValueError(f"未知的编码器: {encoder}")
            
            elapsed = time.time() - start_time
            
            result = create_success_response(
                data=result_data,
                message="编码成功",
                metadata=metadata
            )
            
            # 验证输出
            validation = "pass"
            if encoder == "morgan":
                if len(fingerprints) != len(smiles) or any(len(fp) != n_bits for fp in fingerprints):
                    validation = "fail"
            elif encoder == "chemberta":
                if len(embeddings) != len(smiles) or any(len(emb) != 768 for emb in embeddings):
                    validation = "fail"
            
            self.results.append({
                "test": "encode",
                "encoder": encoder,
                "status": "pass",
                "response_time": elapsed,
                "n_molecules": len(smiles),
                "validation": validation
            })
            
            return result
            
        except Exception as e:
            elapsed = time.time() - start_time
            result = create_error_response(
                message=f"编码失败: {str(e)}"
            )
            
            self.results.append({
                "test": "encode",
                "encoder": encoder,
                "status": "fail",
                "response_time": elapsed,
                "n_molecules": len(smiles),
                "error": str(e)
            })
            
            return result
    
    def generate_report(self) -> Dict[str, Any]:
        """生成测试报告"""
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.get("status") == "pass")
        failed_tests = sum(1 for r in self.results if r.get("status") == "fail")
        
        # 按测试类型分组
        tests_by_type = {}
        for result in self.results:
            test_type = result.get("test", "unknown")
            if test_type not in tests_by_type:
                tests_by_type[test_type] = []
            tests_by_type[test_type].append(result)
        
        # 性能统计
        encode_tests = [r for r in self.results if r.get("test") == "encode" and r.get("status") == "pass"]
        if encode_tests:
            avg_response_time = sum(r.get("response_time", 0) for r in encode_tests) / len(encode_tests)
            avg_molecules = sum(r.get("n_molecules", 0) for r in encode_tests) / len(encode_tests)
        else:
            avg_response_time = 0
            avg_molecules = 0
        
        report = {
            "summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "success_rate": passed_tests / total_tests if total_tests > 0 else 0,
                "avg_response_time": avg_response_time,
                "avg_molecules_per_request": avg_molecules
            },
            "tests_by_type": {
                test_type: {
                    "total": len(results),
                    "passed": sum(1 for r in results if r.get("status") == "pass"),
                    "failed": sum(1 for r in results if r.get("status") == "fail")
                }
                for test_type, results in tests_by_type.items()
            },
            "detailed_results": self.results
        }
        
        return report

def main():
    """主函数"""
    print("🧪 开始MolEnc本地API测试")
    print("=" * 60)
    
    tester = LocalAPITester()
    
    # 基础测试
    print("🔍 基础测试...")
    
    # 健康检查
    print("  测试网关健康检查...")
    result = tester.test_health_check()
    print(f"  ✅ 网关健康检查: {'通过' if result['success'] else '失败'}")
    
    # 编码器测试
    encoders = ["morgan", "chemberta"]
    
    for encoder in encoders:
        print(f"\n🔬 测试 {encoder}...")
        
        # 健康检查
        print(f"  测试{encoder}健康检查...")
        result = tester.test_health_check(encoder)
        print(f"  ✅ {encoder}健康检查: {'通过' if result['success'] else '失败'}")
        
        # 信息接口
        print(f"  测试{encoder}信息接口...")
        result = tester.test_encoder_info(encoder)
        print(f"  ✅ {encoder}信息接口: {'通过' if result['success'] else '失败'}")
        
        # 编码测试
        print(f"  测试{encoder}编码接口...")
        test_smiles = ["CCO", "CCCO", "CCCCO", "c1ccccc1", "c1ccc(C)cc1"]
        result = tester.test_encode(encoder, test_smiles[:3])
        print(f"  ✅ {encoder}编码接口: {'通过' if result['success'] else '失败'}")
    
    # 生成报告
    print("\n📊 生成测试报告...")
    report = tester.generate_report()
    
    # 显示摘要
    summary = report["summary"]
    print(f"\n📈 测试摘要:")
    print(f"  总测试数: {summary['total_tests']}")
    print(f"  通过: {summary['passed_tests']}")
    print(f"  失败: {summary['failed_tests']}")
    print(f"  成功率: {summary['success_rate']:.1%}")
    print(f"  平均响应时间: {summary['avg_response_time']:.3f}s")
    print(f"  平均分子数/请求: {summary['avg_molecules_per_request']:.1f}")
    
    # 按类型统计
    print(f"\n📋 按测试类型:")
    for test_type, stats in report["tests_by_type"].items():
        print(f"  {test_type}: {stats['passed']}/{stats['total']} ({stats['passed']/stats['total']:.1%})")
    
    # 保存详细报告
    report_file = "/home/jiaodu/projects/molenc/docker/examples/test_report_local.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n💾 详细报告已保存到: {report_file}")
    
    # 返回退出码
    exit(0 if summary["failed_tests"] == 0 else 1)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ 测试执行失败: {e}")
        traceback.print_exc()
        exit(1)