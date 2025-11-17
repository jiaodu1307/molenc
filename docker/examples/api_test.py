#!/usr/bin/env python3
"""
MolEnc Docker API测试工具
用于验证API服务的正确性和性能
"""

import requests
import json
import time
import concurrent.futures
from typing import List, Dict, Any, Optional
import argparse

class APITester:
    """API测试器"""
    
    def __init__(self, base_url: str = "http://localhost", timeout: int = 30):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.results = []
    
    def test_health(self, encoder: str = None) -> Dict[str, Any]:
        """测试健康检查"""
        if encoder:
            url = f"{self.base_url}/api/{encoder}/health"
        else:
            url = f"{self.base_url}/health"
        
        start_time = time.time()
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            elapsed = time.time() - start_time
            
            result = {
                "test": "health_check",
                "encoder": encoder or "gateway",
                "status": "pass",
                "response_time": elapsed,
                "response": response.json()
            }
            self.results.append(result)
            return result
            
        except Exception as e:
            elapsed = time.time() - start_time
            result = {
                "test": "health_check",
                "encoder": encoder or "gateway",
                "status": "fail",
                "response_time": elapsed,
                "error": str(e)
            }
            self.results.append(result)
            return result
    
    def test_info(self, encoder: str) -> Dict[str, Any]:
        """测试信息接口"""
        url = f"{self.base_url}/api/{encoder}/info"
        
        start_time = time.time()
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            elapsed = time.time() - start_time
            
            result = {
                "test": "info",
                "encoder": encoder,
                "status": "pass",
                "response_time": elapsed,
                "response": response.json()
            }
            self.results.append(result)
            return result
            
        except Exception as e:
            elapsed = time.time() - start_time
            result = {
                "test": "info",
                "encoder": encoder,
                "status": "fail",
                "response_time": elapsed,
                "error": str(e)
            }
            self.results.append(result)
            return result
    
    def test_encode(self, encoder: str, smiles: List[str], **kwargs) -> Dict[str, Any]:
        """测试编码接口"""
        url = f"{self.base_url}/api/{encoder}/encode"
        
        # 构建请求参数
        params = {"smiles": smiles}
        if encoder == "morgan":
            params.update({
                "n_bits": kwargs.get("n_bits", 1024),
                "radius": kwargs.get("radius", 2),
                "use_features": kwargs.get("use_features", False)
            })
        elif encoder == "chemberta":
            params.update({
                "model_name": kwargs.get("model_name", "seyonec/ChemBERTa-zinc-base-v1"),
                "pooling_strategy": kwargs.get("pooling_strategy", "mean"),
                "max_length": kwargs.get("max_length", 512)
            })
        
        start_time = time.time()
        try:
            response = self.session.post(url, json=params, timeout=self.timeout)
            response.raise_for_status()
            elapsed = time.time() - start_time
            
            result_data = response.json()
            
            result = {
                "test": "encode",
                "encoder": encoder,
                "status": "pass",
                "response_time": elapsed,
                "n_molecules": len(smiles),
                "output_shape": result_data.get("shape", []),
                "metadata": result_data.get("metadata", {})
            }
            
            # 验证输出
            if encoder == "morgan":
                fingerprints = result_data.get("fingerprints", [])
                if len(fingerprints) == len(smiles) and all(len(fp) > 0 for fp in fingerprints):
                    result["validation"] = "pass"
                else:
                    result["validation"] = "fail"
            elif encoder == "chemberta":
                embeddings = result_data.get("embeddings", [])
                if len(embeddings) == len(smiles) and all(len(emb) > 0 for emb in embeddings):
                    result["validation"] = "pass"
                else:
                    result["validation"] = "fail"
            
            self.results.append(result)
            return result
            
        except Exception as e:
            elapsed = time.time() - start_time
            result = {
                "test": "encode",
                "encoder": encoder,
                "status": "fail",
                "response_time": elapsed,
                "n_molecules": len(smiles),
                "error": str(e)
            }
            self.results.append(result)
            return result
    
    def test_concurrent(self, encoder: str, n_requests: int = 10, n_molecules: int = 10) -> Dict[str, Any]:
        """并发测试"""
        print(f"🔄 并发测试: {encoder}, {n_requests} 请求, {n_molecules} 分子/请求")
        
        # 生成测试分子
        test_smiles = ["CCO", "CCCO", "CCCCO", "c1ccccc1", "c1ccc(C)cc1"] * (n_molecules // 5 + 1)
        test_smiles = test_smiles[:n_molecules]
        
        def make_request():
            return self.test_encode(encoder, test_smiles)
        
        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_requests) as executor:
            futures = [executor.submit(make_request) for _ in range(n_requests)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
        
        total_time = time.time() - start_time
        
        # 分析结果
        successful = sum(1 for r in results if r["status"] == "pass")
        failed = sum(1 for r in results if r["status"] == "fail")
        avg_response_time = sum(r.get("response_time", 0) for r in results) / len(results)
        
        result = {
            "test": "concurrent",
            "encoder": encoder,
            "n_requests": n_requests,
            "n_molecules_per_request": n_molecules,
            "total_time": total_time,
            "successful_requests": successful,
            "failed_requests": failed,
            "success_rate": successful / n_requests,
            "avg_response_time": avg_response_time,
            "requests_per_second": n_requests / total_time
        }
        
        self.results.append(result)
        return result
    
    def test_load(self, encoder: str, duration: int = 30, n_molecules: int = 10) -> Dict[str, Any]:
        """负载测试"""
        print(f"⚡ 负载测试: {encoder}, {duration}秒, {n_molecules} 分子/请求")
        
        # 生成测试分子
        test_smiles = ["CCO", "CCCO", "CCCCO", "c1ccccc1", "c1ccc(C)cc1"] * (n_molecules // 5 + 1)
        test_smiles = test_smiles[:n_molecules]
        
        results = []
        start_time = time.time()
        end_time = start_time + duration
        
        while time.time() < end_time:
            result = self.test_encode(encoder, test_smiles)
            results.append(result)
            time.sleep(0.1)  # 避免过快请求
        
        # 分析结果
        successful = sum(1 for r in results if r["status"] == "pass")
        failed = sum(1 for r in results if r["status"] == "fail")
        avg_response_time = sum(r.get("response_time", 0) for r in results) / len(results)
        total_molecules = sum(r.get("n_molecules", 0) for r in results if r["status"] == "pass")
        
        result = {
            "test": "load",
            "encoder": encoder,
            "duration": duration,
            "n_molecules_per_request": n_molecules,
            "total_requests": len(results),
            "successful_requests": successful,
            "failed_requests": failed,
            "success_rate": successful / len(results),
            "avg_response_time": avg_response_time,
            "total_molecules_processed": total_molecules,
            "molecules_per_second": total_molecules / duration
        }
        
        self.results.append(result)
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
    parser = argparse.ArgumentParser(description="MolEnc Docker API测试工具")
    parser.add_argument("--url", default="http://localhost", help="API基础URL")
    parser.add_argument("--timeout", type=int, default=30, help="请求超时时间")
    parser.add_argument("--encoder", choices=["morgan", "chemberta"], help="指定测试的编码器")
    parser.add_argument("--concurrent", type=int, help="并发测试的请求数量")
    parser.add_argument("--load", type=int, help="负载测试的持续时间（秒）")
    parser.add_argument("--molecules", type=int, default=10, help="每个请求的分子数量")
    parser.add_argument("--output", help="输出报告文件")
    
    args = parser.parse_args()
    
    tester = APITester(args.url, args.timeout)
    
    print("🧪 开始MolEnc Docker API测试")
    print("=" * 50)
    
    # 基础测试
    print("🔍 基础测试...")
    
    # 健康检查
    tester.test_health()
    
    # 编码器测试
    encoders = [args.encoder] if args.encoder else ["morgan", "chemberta"]
    
    for encoder in encoders:
        print(f"\n🔬 测试 {encoder}...")
        
        # 信息接口
        tester.test_info(encoder)
        
        # 编码测试
        test_smiles = ["CCO", "CCCO", "CCCCO", "c1ccccc1", "c1ccc(C)cc1"]
        tester.test_encode(encoder, test_smiles[:args.molecules])
    
    # 并发测试
    if args.concurrent:
        for encoder in encoders:
            tester.test_concurrent(encoder, args.concurrent, args.molecules)
    
    # 负载测试
    if args.load:
        for encoder in encoders:
            tester.test_load(encoder, args.load, args.molecules)
    
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
    
    # 输出报告
    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\n💾 详细报告已保存到: {args.output}")
    
    # 返回退出码
    exit(0 if summary["failed_tests"] == 0 else 1)

if __name__ == "__main__":
    main()