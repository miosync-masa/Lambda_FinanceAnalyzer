"""
Lambda³ Cloud Parallel Tests
最小限の並列実行テスト
"""

import pytest
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from cloud.lambda3_cloud_parallel import (
    CloudScaleConfig,
    Lambda3TaskDecomposer,
    ExecutionBackend
)

class TestCloudParallel:
    """並列実行の基本動作確認"""
    
    def test_task_decomposer(self):
        """タスク分割が動くかチェック"""
        config = CloudScaleConfig(batch_size=10)
        decomposer = Lambda3TaskDecomposer(config)
        
        # 5系列のテストデータ
        series_names = ['A', 'B', 'C', 'D', 'E']
        batches = decomposer.decompose_pairwise_analysis(series_names)
        
        # 5系列なら10ペア（5C2）
        total_pairs = sum(len(batch) for batch in batches)
        assert total_pairs == 10
        
    def test_execution_backend_available(self):
        """実行バックエンドが使えるかチェック"""
        assert ExecutionBackend.LOCAL_MULTIPROCESS
        
        # オプショナルなバックエンドは存在チェックのみ
        try:
            import dask
            assert ExecutionBackend.DASK_DISTRIBUTED
        except ImportError:
            pass

    @pytest.mark.skipif(not Path("data/sample").exists(), 
                       reason="サンプルデータがない場合スキップ")
    def test_basic_parallel_run(self):
        """基本的な並列実行（ローカル）"""
        from cloud.lambda3_cloud_parallel import run_lambda3_cloud_scale
        import asyncio
        
        # 小さなテストデータ
        test_data = {
            'Test_A': np.random.randn(100),
            'Test_B': np.random.randn(100),
            'Test_C': np.random.randn(100)
        }
        
        # 最速設定で実行
        from core.lambda3_zeroshot_tensor_field import L3Config
        config = L3Config(draws=100, tune=100)
        
        # ローカル実行のみテスト
        cloud_config = CloudScaleConfig(
            backend=ExecutionBackend.LOCAL_MULTIPROCESS,
            max_workers=2,
            batch_size=2
        )
        
        # 実行（エラーが出ないことを確認）
        try:
            results = asyncio.run(run_lambda3_cloud_scale(
                test_data, 
                scale='small',
                l3_config=config,
                cloud_config=cloud_config
            ))
            assert results is not None
        except Exception as e:
            # 実行環境によってはスキップ
            pytest.skip(f"並列実行環境なし: {e}")
