# Lambda³ Analytics Framework

<p align="center">
  <img src="docs/images/lambda3_logo.png" alt="Lambda³" width="400"/>
</p>

<p align="center">
  <strong>Universal Structural Tensor Field Analytics</strong><br>
  Beyond time, beyond causality - pure structural dynamics in semantic space
</p>

<p align="center">
  <a href="#"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+"></a>
  <a href="#"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="MIT License"></a>
  <a href="#"><img src="https://img.shields.io/badge/cloud-GCP-orange.svg" alt="GCP Ready"></a>
</p>

## 🌟 Overview

Lambda³ (Λ³) is a revolutionary framework for analyzing complex systems through structural tensor dynamics. Unlike traditional time-series analysis, Lambda³ operates in a time-independent structural space where:

- **Λ** (Lambda): Structural tensor representing system state
- **ΛF**: Progression vector (structural evolution)
- **ρT**: Tension scalar (structural stress)
- **ΔΛC**: Structural change pulsations (events)

## 🚀 Key Features

- **Time-Independent Analysis**: No causality assumptions, pure structural dynamics
- **Massive Parallelization**: Leverage 10,000+ cloud instances simultaneously
- **90% Cost Reduction**: Smart spot instance utilization across 30+ regions
- **Regime-Aware**: Automatic market regime detection and adaptation
- **Fault Tolerant**: Automatic checkpointing and preemption handling

## 📦 Installation

### Quick Start with Docker

```bash
# Clone the repository
git clone https://github.com/yourusername/lambda3-analytics.git
cd lambda3-analytics

# Build and run with Docker
docker-compose up lambda3-main
```

### Local Installation

```bash
# Install base requirements
pip install -r requirements/base.txt

# For GCP features
pip install -r requirements/gcp.txt

# Development setup
pip install -r requirements/dev.txt
```

## 🎯 Quick Examples

### Basic Analysis

```python
from lambda3_zeroshot_tensor_field import run_lambda3_analysis

# Analyze financial data
results = run_lambda3_analysis(
    data_source="data/sample/financial_sample.csv",
    verbose=True
)

# View structural interactions
print(results['pairwise_results']['summary'])
```

### Cloud-Scale Analysis

```python
from lambda3_gcp_ultimate import run_lambda3_gcp_ultimate
import asyncio

# Analyze 1000 time series with 500,000 pairwise interactions
results = asyncio.run(run_lambda3_gcp_ultimate(
    data_source=massive_dataset,
    gcp_config=GCPUltimateConfig(
        target_total_instances=10000,  # Use 10,000 instances!
        max_price_per_hour=0.04
    )
))
```

### Regime-Aware Analysis

```python
from lambda3_regime_aware_extension import run_lambda3_regime_aware_analysis

# Detect and analyze different market regimes
results = run_lambda3_regime_aware_analysis(
    data_source="financial_data.csv",
    hierarchical_config=HierarchicalRegimeConfig(
        n_global_regimes=3,  # Bull/Neutral/Bear
        regime_specific_priors=True
    )
)
```

## 📊 Use Cases

- **Financial Markets**: Detect structural relationships between assets
- **Climate Science**: Analyze global climate system interactions  
- **Neuroscience**: Map brain network dynamics
- **Social Networks**: Identify structural influence patterns
- **Gene Expression**: Discover regulatory relationships

## 🏗️ Architecture

```
Lambda³ Framework
├── Core Theory (Λ³)
│   ├── Structural Tensor (Λ)
│   ├── Progression Vector (ΛF)
│   └── Tension Scalar (ρT)
│
├── Analytical Engines
│   ├── Bayesian Inference (PyMC)
│   ├── Regime Detection
│   └── Hierarchical Analysis
│
└── Execution Backends
    ├── Local Multiprocessing
    ├── Cloud Batch (GCP)
    └── Distributed (Ray/Dask)
```

## 🌍 GCP Integration

Lambda³ can utilize Google Cloud Platform's global infrastructure:

- **30+ Regions**: Automatic resource discovery
- **Spot Instances**: 70-90% cost savings
- **Cloud Batch**: Managed job orchestration
- **Preemption Handling**: Automatic task migration

### Setup GCP

```bash
# Configure project
python scripts/setup_gcp_project.py

# Deploy code to GCS
./scripts/deploy_to_gcs.sh

# Launch analysis
python scripts/launch_ultimate_analysis.py
```

## 📖 Documentation

- [Lambda³ Theory](docs/lambda3_theory.md) - Mathematical foundations
- [API Reference](docs/api_reference.md) - Detailed API documentation
- [Deployment Guide](docs/deployment_guide.md) - Cloud deployment instructions
- [Examples](notebooks/) - Jupyter notebook tutorials

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run specific test suite
pytest tests/test_core.py -v

# With coverage
pytest --cov=src tests/
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## 📝 Citation

If you use Lambda³ in your research, please cite:

```bibtex
@software{lambda3_analytics,
  author = {Iizumi, Masamichi},
  title = {Lambda³: Universal Structural Tensor Field Analytics},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/yourusername/lambda3-analytics}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Dr. Masamichi Iizumi for the Lambda³ theoretical framework
- The PyMC development team for probabilistic programming tools
- Google Cloud Platform for massive compute resources

---

<p align="center">
  <strong>Lambda³ - Revealing the hidden structural dynamics of our universe</strong>
</p>
