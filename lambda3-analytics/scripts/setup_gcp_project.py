#!/usr/bin/env python3
"""
GCP Project Setup Script for Lambda³

Sets up Google Cloud Platform project for Lambda³ ultimate parallel processing.
"""

import subprocess
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Optional

class GCPProjectSetup:
    """Setup GCP project for Lambda³"""
    
    def __init__(self, project_id: Optional[str] = None):
        self.project_id = project_id or self._get_current_project()
        self.required_apis = [
            "batch.googleapis.com",
            "compute.googleapis.com",
            "storage-component.googleapis.com",
            "storage-api.googleapis.com",
            "monitoring.googleapis.com",
            "logging.googleapis.com",
            "cloudresourcemanager.googleapis.com",
            "iam.googleapis.com",
        ]
        
        self.regions = [
            "us-central1", "us-east1", "us-west1",
            "europe-west1", "europe-west4",
            "asia-northeast1", "asia-southeast1"
        ]
        
    def _get_current_project(self) -> str:
        """Get current GCP project ID"""
        try:
            result = subprocess.run(
                ["gcloud", "config", "get-value", "project"],
                capture_output=True, text=True, check=True
            )
            project_id = result.stdout.strip()
            if not project_id:
                raise ValueError("No active GCP project")
            return project_id
        except Exception as e:
            print(f"Error getting project ID: {e}")
            print("Please set a project with: gcloud config set project PROJECT_ID")
            sys.exit(1)
    
    def check_gcloud_installed(self) -> bool:
        """Check if gcloud CLI is installed"""
        try:
            subprocess.run(["gcloud", "--version"], capture_output=True, check=True)
            return True
        except:
            print("Error: gcloud CLI not found!")
            print("Install from: https://cloud.google.com/sdk/docs/install")
            return False
    
    def authenticate(self):
        """Authenticate with GCP"""
        print("\n🔐 Authenticating with GCP...")
        try:
            subprocess.run(["gcloud", "auth", "application-default", "login"], check=True)
            print("✓ Authentication successful")
        except Exception as e:
            print(f"✗ Authentication failed: {e}")
            sys.exit(1)
    
    def enable_apis(self):
        """Enable required GCP APIs"""
        print(f"\n🔧 Enabling APIs for project: {self.project_id}")
        
        for api in self.required_apis:
            print(f"  Enabling {api}...", end="", flush=True)
            try:
                subprocess.run(
                    ["gcloud", "services", "enable", api, "--project", self.project_id],
                    capture_output=True, check=True
                )
                print(" ✓")
            except subprocess.CalledProcessError as e:
                print(f" ✗ Failed: {e.stderr.decode()}")
    
    def create_storage_buckets(self):
        """Create regional storage buckets"""
        print("\n📦 Creating storage buckets...")
        
        base_bucket = f"lambda3-{self.project_id}"
        
        # Create main bucket
        self._create_bucket(base_bucket, "us-central1", is_main=True)
        
        # Create regional buckets for results
        for region in self.regions[:3]:  # Create in first 3 regions
            bucket_name = f"{base_bucket}-{region}"
            self._create_bucket(bucket_name, region)
    
    def _create_bucket(self, bucket_name: str, location: str, is_main: bool = False):
        """Create a single bucket"""
        print(f"  Creating gs://{bucket_name} in {location}...", end="", flush=True)
        
        try:
            # Check if bucket exists
            check_cmd = ["gsutil", "ls", f"gs://{bucket_name}"]
            result = subprocess.run(check_cmd, capture_output=True)
            
            if result.returncode == 0:
                print(" (exists)")
                return
            
            # Create bucket
            create_cmd = [
                "gsutil", "mb",
                "-p", self.project_id,
                "-l", location,
                "-c", "STANDARD",
                f"gs://{bucket_name}"
            ]
            subprocess.run(create_cmd, check=True, capture_output=True)
            print(" ✓")
            
            # Set lifecycle for temporary results (delete after 7 days)
            if not is_main:
                self._set_bucket_lifecycle(bucket_name)
                
        except Exception as e:
            print(f" ✗ Failed: {e}")
    
    def _set_bucket_lifecycle(self, bucket_name: str):
        """Set lifecycle rules for bucket"""
        lifecycle_config = {
            "lifecycle": {
                "rule": [{
                    "action": {"type": "Delete"},
                    "condition": {
                        "age": 7,
                        "matchesPrefix": ["checkpoints/", "temp/"]
                    }
                }]
            }
        }
        
        # Save config to temp file
        config_file = Path("/tmp/lifecycle.json")
        with open(config_file, 'w') as f:
            json.dump(lifecycle_config, f)
        
        # Apply lifecycle
        subprocess.run(
            ["gsutil", "lifecycle", "set", str(config_file), f"gs://{bucket_name}"],
            capture_output=True
        )
        
        config_file.unlink()
    
    def create_service_account(self):
        """Create service account for Lambda³"""
        print("\n👤 Creating service account...")
        
        sa_name = "lambda3-worker"
        sa_email = f"{sa_name}@{self.project_id}.iam.gserviceaccount.com"
        
        # Check if exists
        check_cmd = [
            "gcloud", "iam", "service-accounts", "describe",
            sa_email, "--project", self.project_id
        ]
        result = subprocess.run(check_cmd, capture_output=True)
        
        if result.returncode == 0:
            print(f"  Service account {sa_email} already exists")
        else:
            # Create service account
            create_cmd = [
                "gcloud", "iam", "service-accounts", "create", sa_name,
                "--display-name", "Lambda³ Worker Account",
                "--project", self.project_id
            ]
            subprocess.run(create_cmd, check=True)
            print(f"  ✓ Created service account: {sa_email}")
        
        # Grant necessary roles
        roles = [
            "roles/batch.jobsEditor",
            "roles/compute.instanceAdmin",
            "roles/storage.objectAdmin",
            "roles/logging.logWriter",
            "roles/monitoring.metricWriter"
        ]
        
        print("  Granting roles...")
        for role in roles:
            grant_cmd = [
                "gcloud", "projects", "add-iam-policy-binding", self.project_id,
                "--member", f"serviceAccount:{sa_email}",
                "--role", role,
                "--quiet"
            ]
            subprocess.run(grant_cmd, capture_output=True)
        
        print("  ✓ Roles granted")
        
        # Create and download key
        self._create_service_account_key(sa_email)
    
    def _create_service_account_key(self, sa_email: str):
        """Create and download service account key"""
        key_dir = Path("credentials")
        key_dir.mkdir(exist_ok=True)
        key_file = key_dir / "gcp-lambda3-key.json"
        
        if key_file.exists():
            print(f"  Key already exists: {key_file}")
            return
        
        print("  Creating service account key...")
        create_key_cmd = [
            "gcloud", "iam", "service-accounts", "keys", "create",
            str(key_file),
            "--iam-account", sa_email,
            "--project", self.project_id
        ]
        subprocess.run(create_key_cmd, check=True)
        print(f"  ✓ Key saved to: {key_file}")
        
        # Set permissions
        key_file.chmod(0o600)
    
    def set_quotas(self):
        """Request quota increases for massive parallel processing"""
        print("\n📊 Checking quotas...")
        
        quotas_to_check = [
            ("CPUS", 50000),
            ("CPUS_ALL_REGIONS", 100000),
            ("PREEMPTIBLE_CPUS", 100000),
            ("SSD_TOTAL_GB", 50000),
            ("IN_USE_ADDRESSES", 1000)
        ]
        
        print("\n  Recommended quota increases for ultimate parallelization:")
        print("  (Request these in the GCP Console)")
        print("  " + "-"*50)
        
        for quota_name, recommended in quotas_to_check:
            print(f"  {quota_name}: {recommended:,}")
        
        print("\n  Request at: https://console.cloud.google.com/iam-admin/quotas")
    
    def create_firewall_rules(self):
        """Create firewall rules for internal communication"""
        print("\n🔥 Setting up firewall rules...")
        
        rule_name = "lambda3-internal"
        
        # Check if rule exists
        check_cmd = ["gcloud", "compute", "firewall-rules", "describe", rule_name, 
                    "--project", self.project_id]
        result = subprocess.run(check_cmd, capture_output=True)
        
        if result.returncode == 0:
            print(f"  Firewall rule '{rule_name}' already exists")
        else:
            # Create rule
            create_cmd = [
                "gcloud", "compute", "firewall-rules", "create", rule_name,
                "--allow", "tcp:22,tcp:80,tcp:443,tcp:8080-8090",
                "--source-tags", "lambda3-worker",
                "--target-tags", "lambda3-worker",
                "--project", self.project_id
            ]
            subprocess.run(create_cmd, check=True)
            print(f"  ✓ Created firewall rule: {rule_name}")
    
    def upload_code_to_gcs(self):
        """Upload Lambda³ code to GCS"""
        print("\n📤 Uploading Lambda³ code to GCS...")
        
        bucket = f"lambda3-{self.project_id}"
        
        # Core modules to upload
        modules = [
            "src/core/lambda3_zeroshot_tensor_field.py",
            "src/core/lambda3_regime_aware_extension.py",
            "src/cloud/lambda3_cloud_parallel.py",
            "src/cloud/lambda3_gcp_ultimate.py",
            "src/cloud/lambda3_cloud_worker.py",
            "src/cloud/lambda3_result_aggregator.py"
        ]
        
        for module in modules:
            if Path(module).exists():
                print(f"  Uploading {module}...", end="", flush=True)
                subprocess.run(
                    ["gsutil", "cp", module, f"gs://{bucket}/code/"],
                    capture_output=True
                )
                print(" ✓")
            else:
                print(f"  Skipping {module} (not found)")
    
    def create_sample_data(self):
        """Create sample data for testing"""
        print("\n📊 Creating sample data...")
        
        sample_dir = Path("data/sample")
        sample_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a small sample CSV
        import pandas as pd
        import numpy as np
        
        dates = pd.date_range('2023-01-01', periods=200, freq='D')
        data = {
            'Date': dates,
            'Series_A': 100 + np.cumsum(np.random.randn(200) * 0.5),
            'Series_B': 50 + np.cumsum(np.random.randn(200) * 0.3),
            'Series_C': 75 + np.cumsum(np.random.randn(200) * 0.4)
        }
        
        df = pd.DataFrame(data)
        sample_file = sample_dir / "sample_data.csv"
        df.to_csv(sample_file, index=False)
        
        print(f"  ✓ Created sample data: {sample_file}")
        
        # Upload to GCS
        bucket = f"lambda3-{self.project_id}"
        subprocess.run(
            ["gsutil", "cp", str(sample_file), f"gs://{bucket}/data/"],
            capture_output=True
        )
        print(f"  ✓ Uploaded to gs://{bucket}/data/")
    
    def print_summary(self):
        """Print setup summary"""
        print("\n" + "="*80)
        print("LAMBDA³ GCP SETUP COMPLETE!")
        print("="*80)
        
        print(f"\nProject ID: {self.project_id}")
        print(f"Main bucket: gs://lambda3-{self.project_id}")
        print(f"Service account: lambda3-worker@{self.project_id}.iam.gserviceaccount.com")
        
        print("\n📋 Next Steps:")
        print("1. Request quota increases (see recommendations above)")
        print("2. Set up billing alerts in GCP Console")
        print("3. Run: python scripts/launch_ultimate_analysis.py")
        
        print("\n⚡ Quick Test:")
        print(f"gsutil ls gs://lambda3-{self.project_id}/")
        
        print("\n🚀 Ready for planet-scale Lambda³ analysis!")

def main():
    """Main setup execution"""
    print("╔═══════════════════════════════════════════════════════════╗")
    print("║         LAMBDA³ GCP PROJECT SETUP                         ║")
    print("║                                                           ║")
    print("║    Preparing for planet-scale structural tensor analysis  ║")
    print("╚═══════════════════════════════════════════════════════════╝")
    
    # Get project ID
    project_id = input("\nEnter GCP project ID (or press Enter for current): ").strip()
    
    setup = GCPProjectSetup(project_id if project_id else None)
    
    # Check prerequisites
    if not setup.check_gcloud_installed():
        return
    
    print(f"\nSetting up project: {setup.project_id}")
    proceed = input("Continue? (y/N): ").strip().lower()
    
    if proceed != 'y':
        print("Setup cancelled")
        return
    
    # Run setup steps
    try:
        setup.authenticate()
        setup.enable_apis()
        setup.create_storage_buckets()
        setup.create_service_account()
        setup.create_firewall_rules()
        setup.set_quotas()
        setup.upload_code_to_gcs()
        setup.create_sample_data()
        setup.print_summary()
        
    except KeyboardInterrupt:
        print("\n\nSetup interrupted")
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
